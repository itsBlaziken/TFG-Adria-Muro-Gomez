# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE_LOCAL_UPLOAD.PY — Genera prediccions i puja els resultats a S3
#
# QUÈ FA:
#   1. Descarrega el paquet de models (model.tar.gz) de S3 i el descomprimeix
#   2. Descarrega el features.csv (característiques espectrals) de S3
#   3. Aplica el pipeline de predicció sobre cada registre:
#        - Isolation Forest  → decideix si és normal o anòmal
#        - Gradient Boosting → si és anòmal, classifica la tipologia de fallo
#   4. Assigna un nivell de severitat (HIGH/MEDIUM/LOW) basat en la confiança
#   5. Puja tres fitxers de resultats a S3:
#        - predictions/features.csv.out → prediccions raw en format JSON
#        - outputs/all_predictions.csv  → tots els registres amb la seva predicció
#        - outputs/alerts.csv           → únicament els registres anòmals
#
# PER QUÈ S'EXECUTA EN LOCAL?
#   Igual que l'entrenament, els comptes nous d'AWS tenen quota 0 per a
#   SageMaker Batch Transform Jobs. El codi és compatible amb Batch Transform
#   i es podria executar al núvol un cop s'obtinguin les quotes.
#
# ÚS:
#   python inference_local_upload.py
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# LLINDARS DE SEVERITAT
#
# El classificador Gradient Boosting retorna una probabilitat (confidence)
# entre 0 i 1 per a cada predicció. A partir d'aquesta probabilitat, assignem
# un nivell de severitat que el personal tècnic pot interpretar fàcilment:
#
#   HIGH   (≥ 0.5) → el model té alta confiança en la classificació de fallo
#   MEDIUM (≥ 0.3) → confiança moderada, cal revisar
#   LOW    (< 0.3) → confiança baixa, possible fals positiu
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {'HIGH': 0.5, 'MEDIUM': 0.3}


def download_models():
    """Descarrega el paquet de models de S3 i el descomprimeix en un directori temporal.

    El model.tar.gz conté tots els fitxers .pkl (models serialitzats) i el
    script d'inferència. Es descomprimeix en un directori temporal per poder
    carregar els models amb pickle.
    """
    print("Descarregant model.tar.gz des de S3...")

    # Descarrega el fitxer directament a memòria (BytesIO) sense desar-lo al disc
    tar_buffer = BytesIO()
    s3.download_fileobj(bucket, f"{S3_MODELS}/model.tar.gz", tar_buffer)
    tar_buffer.seek(0)  # Torna al principi del buffer per poder llegir-lo

    # Descomprimeix el paquet en un directori temporal
    model_dir = tempfile.mkdtemp()
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        tar.extractall(model_dir)

    # load_models() carrega tots els .pkl del directori i els retorna en un diccionari
    return load_models(model_dir)


def download_features():
    """Descarrega el CSV de característiques de S3 per fer la inferència."""
    print("Descarregant features.csv des de S3...")
    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", tmp)
    df = pd.read_csv(tmp)
    print(f"Features carregades: {len(df)} registres")
    return df


def classify_severity(confidence):
    """Converteix un score de confiança numèric en un nivell de severitat llegible."""
    if confidence >= SEVERITY_THRESHOLDS['HIGH']:   return 'HIGH'
    if confidence >= SEVERITY_THRESHOLDS['MEDIUM']: return 'MEDIUM'
    return 'LOW'


def run_inference(df, models):
    """Aplica el pipeline de predicció sobre totes les característiques.

    Extreu la matriu de features del DataFrame, la passa per predict_batch()
    (definit a scripts/inference.py) i retorna una llista de diccionaris,
    un per cada registre, amb el resultat de la predicció.
    """
    # Convertim les columnes de features a una matriu numpy per als models
    X       = df[FEATURE_COLS].fillna(0).values
    results = predict_batch(X, models)

    n_anom = sum(r['is_anomaly'] for r in results)
    print(f"Anomalies detectades: {n_anom} ({100 * n_anom / len(X):.1f}%)")
    return results


def upload_results(results, df):
    """Construeix els DataFrames de resultats i els puja a S3.

    Genera tres fitxers:
      1. predictions/features.csv.out → format raw JSON (compatible amb Batch Transform)
      2. all_predictions.csv          → tots els registres amb metadades + predicció
      3. alerts.csv                   → únicament els registres anòmals (per al dashboard)
    """
    # ─── Fitxer 1: prediccions raw en format JSON (una per línia) ────────────
    # Aquest format és el que generaria un SageMaker Batch Transform Job,
    # per mantenir la compatibilitat si s'executa al núvol en el futur.
    raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in results).encode()
    s3.put_object(Bucket=bucket, Key=f"{S3_OUTPUTS}/predictions/features.csv.out", Body=raw)
    print(f"Prediccions pujades: s3://{bucket}/{S3_OUTPUTS}/predictions/")

    # ─── Fitxer 2 i 3: CSVs estructurats amb metadades del dispositiu ────────
    # Construïm un DataFrame combinant les prediccions amb les metadades
    # originals del dispositiu (identificador, eix, timestamp, mode)
    df_results = pd.DataFrame([{
        'is_anomaly':     r['is_anomaly'],
        'fault_type':     r['fault_type'],
        'fault_label':    r['fault_description'],
        'confidence':     r['confidence'],
        # Assignem severitat NONE als registres normals (no anòmals)
        'severity_level': classify_severity(r['confidence']) if r['is_anomaly'] else 'NONE',
    } for r in results])

    # Agafem les columnes de metadades del dispositiu que existeixin al CSV
    meta_cols = [c for c in ['dev_eui', 'mode', 'axis', 'timestamp'] if c in df.columns]
    df_full   = pd.concat([df[meta_cols].reset_index(drop=True), df_results], axis=1)

    # Filtrem únicament les anomalies per generar el fitxer d'alertes del dashboard
    df_alerts = df_full[df_full['is_anomaly']].copy()

    # Puja ambdós CSVs a S3 usant un buffer en memòria per evitar fitxers temporals
    for key, frame in [
        (f"{S3_OUTPUTS}/all_predictions.csv", df_full),    # tots els registres
        (f"{S3_OUTPUTS}/alerts.csv",           df_alerts), # únicament anomalies
    ]:
        buf = BytesIO()
        frame.to_csv(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
        print(f"Pujat: s3://{bucket}/{key}  ({len(frame)} files)")

    return df_full, df_alerts


if __name__ == "__main__":
    # Pas 1: Descarregar i carregar els models des de S3
    models = download_models()

    # Pas 2: Descarregar les característiques espectrals des de S3
    df = download_features()

    # Pas 3: Executar la inferència sobre tots els registres
    results = run_inference(df, models)

    # Pas 4: Pujar els resultats a S3
    df_full, df_alerts = upload_results(results, df)

    # Resum final per consola
    print(f"\nTotal registres:  {len(df_full)}")
    print(f"Anomalies:        {len(df_alerts)} ({100 * len(df_alerts) / len(df_full):.1f}%)")
    print("\nDistribució per tipologia de fallo:")
    for ft, label in FAULT_TYPES.items():
        n = (df_alerts['fault_type'] == ft).sum()
        print(f"  {label}: {n}")

    print(f"\nResultats a S3: s3://{bucket}/{S3_OUTPUTS}/")
    print("Fet")
