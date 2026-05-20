# PASO 4: Inferencia local + subida de resultados a S3
# Las cuentas nuevas tienen quota 0 para Transform Jobs.
# Este script ejecuta la inferencia localmente y sube los resultados
# a S3 outputs/ en el mismo formato que generaria un Batch Transform.

import boto3
import json
import os
import tempfile
import pickle
import tarfile
import numpy as np
import pandas as pd
from io import BytesIO
from config import REGION, S3_PROCESSED, S3_MODELS, S3_OUTPUTS, get_bucket_name

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()

FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]

FAULT_LABELS = {
    0: 'High Amplitude Variability - holgura/impacto',
    1: 'Mid-Frequency Energy - desalineacion',
    2: 'Low-Frequency Energy - desbalance',
    3: 'Distributed Energy - degradacion general',
}


def load_models():
    print("Descargando model.tar.gz desde S3...")
    tar_buffer = BytesIO()
    s3.download_fileobj(bucket, f"{S3_MODELS}/model.tar.gz", tar_buffer)
    tar_buffer.seek(0)

    models = {}
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".pkl"):
                f = tar.extractfile(member)
                key = member.name.replace(".pkl", "")
                models[key] = pickle.load(f)
    print(f"[OK] Modelos cargados: {list(models.keys())}")
    return models


def load_features():
    print("Descargando features.csv desde S3...")
    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", tmp)
    df = pd.read_csv(tmp)
    print(f"[OK] {len(df)} registros")
    return df


def run_inference(df, models):
    X = df[FEATURE_COLS].fillna(0).values
    X_scaled = models["scaler"].transform(X)

    anomaly_preds = models["isolation_forest"].predict(X_scaled)
    is_anomaly    = anomaly_preds == -1

    fault_types = np.full(len(X), -1, dtype=int)
    severity    = np.zeros(len(X), dtype=float)

    if np.any(is_anomaly):
        X_anom  = X_scaled[is_anomaly]
        scores  = models["isolation_forest"].score_samples(X_anom)
        ft      = models["gradient_boosting"].predict(X_anom)
        fault_types[is_anomaly] = ft
        severity[is_anomaly]    = np.abs(scores)

    results = []
    for i in range(len(X)):
        results.append({
            "is_anomaly":  bool(is_anomaly[i]),
            "fault_type":  int(fault_types[i]),
            "fault_label": FAULT_LABELS.get(int(fault_types[i]), "Normal"),
            "severity":    round(float(severity[i]), 4),
        })

    n_anom = is_anomaly.sum()
    print(f"Anomalias detectadas: {n_anom} ({100*n_anom/len(X):.1f}%)")
    return results, df


def upload_results(results, df):
    # Subir predictions.json (formato equivalente al Batch Transform output)
    predictions_bytes = "\n".join(json.dumps(r) for r in results).encode()
    s3.put_object(
        Bucket=bucket,
        Key=f"{S3_OUTPUTS}/predictions/features.csv.out",
        Body=predictions_bytes,
    )
    print(f"[OK] Predicciones subidas: s3://{bucket}/{S3_OUTPUTS}/predictions/")

    # Construir CSV de alertas
    df_results = pd.DataFrame(results)
    meta_cols  = [c for c in ['dev_eui', 'mode', 'axis', 'timestamp'] if c in df.columns]
    df_meta    = df[meta_cols].reset_index(drop=True)
    df_full    = pd.concat([df_meta, df_results], axis=1)
    df_alerts  = df_full[df_full["is_anomaly"]].copy()

    def severity_level(s):
        if s >= 0.5: return "HIGH"
        if s >= 0.3: return "MEDIUM"
        return "LOW"

    df_alerts["severity_level"] = df_alerts["severity"].apply(severity_level)

    # Subir CSVs a S3
    for key, frame in [
        (f"{S3_OUTPUTS}/all_predictions.csv", df_full),
        (f"{S3_OUTPUTS}/alerts.csv",           df_alerts),
    ]:
        buf = BytesIO()
        frame.to_csv(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
        print(f"[OK] Subido: s3://{bucket}/{key}  ({len(frame)} filas)")

    return df_full, df_alerts


if __name__ == "__main__":
    print("=" * 60)
    print("INFERENCIA LOCAL + SUBIDA A S3")
    print("=" * 60)

    models            = load_models()
    df                = load_features()
    results, df       = run_inference(df, models)
    df_full, df_alerts = upload_results(results, df)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total registros:     {len(df_full)}")
    print(f"Anomalias:           {len(df_alerts)} ({100*len(df_alerts)/len(df_full):.1f}%)")
    print()
    for ft, label in FAULT_LABELS.items():
        n = (df_alerts["fault_type"] == ft).sum()
        print(f"  {label}: {n}")
    print()
    print(f"Resultados en S3:")
    print(f"  s3://{bucket}/{S3_OUTPUTS}/alerts.csv")
    print(f"  s3://{bucket}/{S3_OUTPUTS}/all_predictions.csv")
    print("\nAhora ejecuta: python 5_generate_alerts.py")
