# PASO 3: Entrenamiento local + subida de modelos a S3
# Las cuentas nuevas de AWS tienen quota 0 para Training Jobs.
# Este script entrena localmente (mismo pipeline validado) y sube
# los artefactos a S3 en el formato que espera SageMaker.

import boto3
import tarfile
import os
import json
import tempfile
import numpy as np
import pandas as pd
import pickle
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import REGION, S3_PROCESSED, S3_MODELS, get_bucket_name

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()

FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]


def download_features():
    print("Descargando features.csv desde S3...")
    tmp = tempfile.mktemp(suffix=".csv")
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", tmp)
    df = pd.read_csv(tmp)
    print(f"[OK] {len(df)} registros cargados")
    return df


def train(df):
    X = df[FEATURE_COLS].fillna(0).values

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    contamination = min(0.20, max(0.10, len(X) / 10000))
    iso = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
    preds = iso.fit_predict(X_scaled)
    anomaly_mask = preds == -1
    n_anom = anomaly_mask.sum()
    print(f"Anomalias detectadas: {n_anom} ({100*n_anom/len(X):.1f}%)")

    X_anom = X_scaled[anomaly_mask]

    # K-Means
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_anom)

    # Gradient Boosting + CV
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=5, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(gb, X_anom, labels, cv=skf, scoring='accuracy')
    print(f"CV Accuracy: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
    gb.fit(X_anom, labels)

    return scaler, iso, kmeans, gb


def package_and_upload(scaler, iso, kmeans, gb):
    # Crear model.tar.gz en memoria con todos los artefactos
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for name, obj in [("scaler.pkl", scaler), ("isolation_forest.pkl", iso),
                          ("kmeans.pkl", kmeans), ("gradient_boosting.pkl", gb)]:
            pkl_bytes = pickle.dumps(obj)
            info = tarfile.TarInfo(name=name)
            info.size = len(pkl_bytes)
            tar.addfile(info, BytesIO(pkl_bytes))

        # feature_cols.txt
        cols_bytes = "\n".join(FEATURE_COLS).encode()
        info = tarfile.TarInfo(name="feature_cols.txt")
        info.size = len(cols_bytes)
        tar.addfile(info, BytesIO(cols_bytes))

        # inference.py dentro del paquete (necesario para Batch Transform)
        with open("scripts/inference.py", "rb") as f:
            inf_bytes = f.read()
        info = tarfile.TarInfo(name="inference.py")
        info.size = len(inf_bytes)
        tar.addfile(info, BytesIO(inf_bytes))

    tar_buffer.seek(0)
    s3_key = f"{S3_MODELS}/model.tar.gz"
    s3.upload_fileobj(tar_buffer, bucket, s3_key)
    model_uri = f"s3://{bucket}/{s3_key}"
    print(f"[OK] Modelo subido: {model_uri}")
    return model_uri


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRENAMIENTO LOCAL + SUBIDA A S3")
    print("=" * 60)

    df        = download_features()
    scaler, iso, kmeans, gb = train(df)
    model_uri = package_and_upload(scaler, iso, kmeans, gb)

    with open(".last_model_uri", "w") as f:
        f.write(model_uri)

    print(f"\n[OK] Modelos en: {model_uri}")
    print("Ahora ejecuta: python 4_run_batch_inference.py")
