# Descarrega les prediccions de S3 i genera el CSV d'alertes industrials en local.
# La sortida alerts.csv és compatible per importar a Amazon QuickSight.

import boto3
import json
import os
import pandas as pd
from config import REGION, S3_PROCESSED, S3_OUTPUTS, get_bucket_name

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()

# Descripció de cada tipus de fallo i llindars de severitat
FAULT_LABELS = {
    0:  'Type 0 - High Std Amplitude - holgura / impacto mecánico',
    1:  'Type 1 - High Energy Mid-Frequency - desalineación',
    2:  'Type 2 - High Energy Low-Frequency - desbalance mecánico',
    3:  'Type 3 - Balanced Multi-Band Energy - degradación general',
    -1: 'Normal',
}

SEVERITY_THRESHOLDS = {'HIGH': 0.5, 'MEDIUM': 0.3}


def download_predictions(bucket, prefix, local_dir):
    # Descarrega tots els fitxers de prediccions del prefix S3 indicat
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    files     = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key        = obj["Key"]
            local_path = os.path.join(local_dir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)
            files.append(local_path)
            print(f"  Descarregat: {key}")

    return files


def load_features_meta(bucket, prefix):
    # Carrega les metadades del CSV de característiques (dev_eui, mode, axis, timestamp)
    local = "/tmp/features.csv"
    try:
        s3.download_file(bucket, f"{prefix}/features.csv", local)
        df   = pd.read_csv(local)
        cols = [c for c in ['dev_eui', 'mode', 'axis', 'timestamp'] if c in df.columns]
        return df[cols].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def parse_predictions(files):
    # Llegeix i parseja les prediccions JSON dels fitxers descarregats
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


def classify_severity(score):
    # Assigna un nivell de severitat discret a partir del score continu
    if score >= SEVERITY_THRESHOLDS['HIGH']:   return 'HIGH'
    if score >= SEVERITY_THRESHOLDS['MEDIUM']: return 'MEDIUM'
    return 'LOW'


def main():
    # Descarrega i parseja les prediccions de S3
    print(f"Descarregant prediccions de s3://{bucket}/{S3_OUTPUTS}/predictions/")
    local_dir = os.path.join(os.path.dirname(__file__), "../outputs/predictions")
    files     = download_predictions(bucket, f"{S3_OUTPUTS}/predictions", local_dir)

    if not files:
        print("Error: no hi ha prediccions. Executa el pas 4 primer.")
        return

    print(f"\nParsejant {len(files)} fitxer(s)...")
    preds = parse_predictions(files)
    print(f"Total prediccions: {len(preds)}")

    # Carrega les metadades del dispositiu per enriquir les alertes
    meta = load_features_meta(bucket, S3_PROCESSED)

    # Construeix el DataFrame d'alertes combinant prediccions i metadades
    records = []
    for i, pred in enumerate(preds):
        fault_type = pred.get('fault_type', -1)
        is_anomaly = pred.get('is_anomaly', False)
        confidence = pred.get('confidence', 0.0)

        row = {
            'is_anomaly':     is_anomaly,
            'fault_type':     fault_type,
            'fault_label':    FAULT_LABELS.get(fault_type, 'Desconegut'),
            'confidence':     round(confidence, 4),
            'severity_level': classify_severity(confidence) if is_anomaly else 'NONE',
        }

        if not meta.empty and i < len(meta):
            for col in meta.columns:
                row[col] = meta.iloc[i][col]

        records.append(row)

    df_all    = pd.DataFrame(records)
    df_alerts = df_all[df_all['is_anomaly']].copy()

    # Desa els CSVs en local i els puja a S3
    out_dir     = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    full_path   = os.path.join(out_dir, "all_predictions.csv")
    alerts_path = os.path.join(out_dir, "alerts.csv")

    df_all.to_csv(full_path,   index=False)
    df_alerts.to_csv(alerts_path, index=False)

    s3.upload_file(alerts_path, bucket, f"{S3_OUTPUTS}/alerts.csv")
    s3.upload_file(full_path,   bucket, f"{S3_OUTPUTS}/all_predictions.csv")

    # Resum de les alertes generades
    print(f"\nTotal registres processats: {len(df_all)}")
    print(f"Anomalies detectades:       {len(df_alerts)} ({100 * len(df_alerts) / len(df_all):.1f}%)")

    print("\nDistribució per tipus de fallo:")
    for ft, label in FAULT_LABELS.items():
        if ft == -1:
            continue
        n = len(df_alerts[df_alerts['fault_type'] == ft])
        print(f"  {label}: {n}")

    print("\nDistribució per severitat:")
    for level in ['HIGH', 'MEDIUM', 'LOW']:
        n = len(df_alerts[df_alerts['severity_level'] == level])
        print(f"  {level}: {n}")

    print(f"\nFitxers generats:")
    print(f"  Local: {alerts_path}")
    print(f"  S3:    s3://{bucket}/{S3_OUTPUTS}/alerts.csv")


if __name__ == "__main__":
    main()
