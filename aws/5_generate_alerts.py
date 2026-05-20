# PASO 5: Descargar predicciones de S3 y generar CSV de alertas industriales
# Output: outputs/alerts.csv  (listo para importar en QuickSight)

import boto3
import json
import os
import pandas as pd
from config import REGION, S3_PROCESSED, S3_OUTPUTS, get_bucket_name

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()

FAULT_LABELS = {
    0: 'High Amplitude Variability - holgura/impacto',
    1: 'Mid-Frequency Energy - desalineacion',
    2: 'Low-Frequency Energy - desbalance',
    3: 'Distributed Energy - degradacion general',
    -1: 'Normal',
}

SEVERITY_THRESHOLDS = {
    'HIGH':   0.5,
    'MEDIUM': 0.3,
}


def download_predictions(bucket, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = os.path.basename(key)
            local_path = os.path.join(local_dir, fname)
            s3.download_file(bucket, key, local_path)
            files.append(local_path)
            print(f"  Descargado: {key}")

    return files


def load_features_meta(bucket, prefix):
    local = "/tmp/features.csv"
    try:
        s3.download_file(bucket, f"{prefix}/features.csv", local)
        df = pd.read_csv(local)
        meta_cols = ['dev_eui', 'mode', 'axis', 'timestamp']
        available = [c for c in meta_cols if c in df.columns]
        return df[available].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def parse_predictions(files):
    rows = []
    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, list):
                        rows.extend(item)
                    elif isinstance(item, dict):
                        rows.append(item)
                except json.JSONDecodeError:
                    pass
    return rows


def classify_severity(severity_score):
    if severity_score >= SEVERITY_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif severity_score >= SEVERITY_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    return 'LOW'


def main():
    print(f"Descargando predicciones de s3://{bucket}/{S3_OUTPUTS}/predictions/")
    local_dir = os.path.join(os.path.dirname(__file__), "../outputs/predictions")
    files = download_predictions(bucket, f"{S3_OUTPUTS}/predictions", local_dir)

    if not files:
        print("[ERROR] No se encontraron predicciones. Ejecuta el paso 4 primero.")
        return

    print(f"\nParsing {len(files)} fichero(s)...")
    preds = parse_predictions(files)
    print(f"  Total predicciones: {len(preds)}")

    # Cargar metadatos (dev_eui, mode, axis, timestamp) del CSV procesado
    meta = load_features_meta(bucket, S3_PROCESSED)

    # Construir DataFrame de alertas
    records = []
    for i, pred in enumerate(preds):
        fault_type = pred.get('fault_type', -1)
        is_anomaly = pred.get('is_anomaly', False)
        severity   = pred.get('severity', 0.0)

        row = {
            'is_anomaly':   is_anomaly,
            'fault_type':   fault_type,
            'fault_label':  FAULT_LABELS.get(fault_type, 'Desconocido'),
            'severity':     round(severity, 4),
            'severity_level': classify_severity(severity) if is_anomaly else 'NONE',
        }

        if not meta.empty and i < len(meta):
            for col in meta.columns:
                row[col] = meta.iloc[i][col]

        records.append(row)

    df_alerts = pd.DataFrame(records)

    # Solo alertas (anomalias)
    df_anomalies = df_alerts[df_alerts['is_anomaly']].copy()

    # Guardar CSVs
    out_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(out_dir, exist_ok=True)

    full_path   = os.path.join(out_dir, "all_predictions.csv")
    alerts_path = os.path.join(out_dir, "alerts.csv")

    df_alerts.to_csv(full_path, index=False)
    df_anomalies.to_csv(alerts_path, index=False)

    # Subir alertas a S3
    s3.upload_file(alerts_path, bucket, f"{S3_OUTPUTS}/alerts.csv")
    s3.upload_file(full_path,   bucket, f"{S3_OUTPUTS}/all_predictions.csv")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE ALERTAS")
    print("=" * 60)
    print(f"Total registros procesados: {len(df_alerts)}")
    print(f"Anomalias detectadas:       {len(df_anomalies)} ({100*len(df_anomalies)/len(df_alerts):.1f}%)")
    print()

    if len(df_anomalies) > 0:
        print("Distribucion por tipo de fallo:")
        for ft, label in FAULT_LABELS.items():
            if ft == -1:
                continue
            n = len(df_anomalies[df_anomalies['fault_type'] == ft])
            print(f"  {label}: {n}")
        print()
        print("Distribucion por severidad:")
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            n = len(df_anomalies[df_anomalies['severity_level'] == level])
            print(f"  {level}: {n}")

    print()
    print(f"Archivos generados:")
    print(f"  Local:  {alerts_path}")
    print(f"  S3:     s3://{bucket}/{S3_OUTPUTS}/alerts.csv")
    print(f"  S3:     s3://{bucket}/{S3_OUTPUTS}/all_predictions.csv")
    print("\nListo para conectar con Amazon QuickSight.")


if __name__ == "__main__":
    main()
