# Executa la inferència en local i puja els resultats a S3.
# Les comptes noves d'AWS tenen quota 0 per a Transform Jobs, per tant s'executa
# localment amb el mateix codi que s'executaria al contenidor de SageMaker.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import boto3
import json
import tarfile
import tempfile
import numpy as np
import pandas as pd
from io import BytesIO
from config import REGION, S3_PROCESSED, S3_MODELS, S3_OUTPUTS, get_bucket_name
from inference import load_models, predict_batch, FEATURE_COLS, FAULT_TYPES

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()

# Llindars de severitat basats en el score d'anomalia
SEVERITY_THRESHOLDS = {'HIGH': 0.5, 'MEDIUM': 0.3}


def download_models():
    # Descarrega el paquet de models des de S3 i el descomprimeix en un directori temporal
    print("Descarregant model.tar.gz des de S3...")
    tar_buffer = BytesIO()
    s3.download_fileobj(bucket, f"{S3_MODELS}/model.tar.gz", tar_buffer)
    tar_buffer.seek(0)

    model_dir = tempfile.mkdtemp()
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        tar.extractall(model_dir)

    return load_models(model_dir)


def download_features():
    # Descarrega el CSV de característiques per fer la inferència
    print("Descarregant features.csv des de S3...")
    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", tmp)
    df = pd.read_csv(tmp)
    print(f"Features carregades: {len(df)} registres")
    return df


def classify_severity(confidence):
    # Assigna un nivell de severitat discret a partir del score continu
    if confidence >= SEVERITY_THRESHOLDS['HIGH']:   return 'HIGH'
    if confidence >= SEVERITY_THRESHOLDS['MEDIUM']: return 'MEDIUM'
    return 'LOW'


def run_inference(df, models):
    # Aplica el pipeline de predicció sobre totes les característiques
    X       = df[FEATURE_COLS].fillna(0).values
    results = predict_batch(X, models)

    n_anom = sum(r['is_anomaly'] for r in results)
    print(f"Anomalies detectades: {n_anom} ({100 * n_anom / len(X):.1f}%)")
    return results


def upload_results(results, df):
    # Puja les prediccions raw en format JSON (equivalent a la sortida del Batch Transform)
    raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in results).encode()
    s3.put_object(Bucket=bucket, Key=f"{S3_OUTPUTS}/predictions/features.csv.out", Body=raw)
    print(f"Prediccions pujades: s3://{bucket}/{S3_OUTPUTS}/predictions/")

    # Construeix els CSV de resultats afegint les metadades del dispositiu
    df_results = pd.DataFrame([{
        'is_anomaly':    r['is_anomaly'],
        'fault_type':    r['fault_type'],
        'fault_label':   r['fault_description'],
        'confidence':    r['confidence'],
        'severity_level': classify_severity(r['confidence']) if r['is_anomaly'] else 'NONE',
    } for r in results])

    meta_cols = [c for c in ['dev_eui', 'mode', 'axis', 'timestamp'] if c in df.columns]
    df_full   = pd.concat([df[meta_cols].reset_index(drop=True), df_results], axis=1)
    df_alerts = df_full[df_full['is_anomaly']].copy()

    # Puja el CSV complet i el CSV filtrat d'alertes a S3
    for key, frame in [
        (f"{S3_OUTPUTS}/all_predictions.csv", df_full),
        (f"{S3_OUTPUTS}/alerts.csv",           df_alerts),
    ]:
        buf = BytesIO()
        frame.to_csv(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
        print(f"Pujat: s3://{bucket}/{key}  ({len(frame)} files)")

    return df_full, df_alerts


if __name__ == "__main__":
    models             = download_models()
    df                 = download_features()
    results            = run_inference(df, models)
    df_full, df_alerts = upload_results(results, df)

    # Resum final de les prediccions i distribució per tipus de fallo
    print(f"\nTotal registres:  {len(df_full)}")
    print(f"Anomalies:        {len(df_alerts)} ({100 * len(df_alerts) / len(df_full):.1f}%)")
    for ft, label in FAULT_TYPES.items():
        n = (df_alerts['fault_type'] == ft).sum()
        print(f"  {label}: {n}")

    print(f"\nResultats a S3: s3://{bucket}/{S3_OUTPUTS}/")
    print("Ara executa: python 5_generate_alerts.py")
