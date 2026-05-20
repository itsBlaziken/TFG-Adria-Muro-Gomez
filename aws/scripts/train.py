# Script que corre DENTRO del SageMaker Training Job (contenedor sklearn)
# Input:  SM_CHANNEL_TRAIN / /opt/ml/input/data/train/features.csv
# Output: SM_MODEL_DIR    / /opt/ml/model/  (SageMaker lo comprime a model.tar.gz)

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]


def train(train_dir, model_dir):
    csv_path = os.path.join(train_dir, 'features.csv')
    print(f"Cargando datos: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Registros cargados: {len(df)}")

    X = df[FEATURE_COLS].fillna(0).values

    # --- Normalización ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Isolation Forest ---
    contamination = min(0.20, max(0.10, len(X) / 10000))
    iso = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
    preds = iso.fit_predict(X_scaled)

    anomaly_mask = preds == -1
    n_anom = anomaly_mask.sum()
    print(f"Anomalias detectadas: {n_anom} ({100*n_anom/len(X):.1f}%)")

    X_anom = X_scaled[anomaly_mask]

    # --- K-Means sobre anomalias ---
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_anom)

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} muestras ({100*c/len(labels):.1f}%)")

    # --- Gradient Boosting + validacion cruzada ---
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=5, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb, X_anom, labels, cv=skf, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    gb.fit(X_anom, labels)

    # --- Guardar modelos ---
    os.makedirs(model_dir, exist_ok=True)

    artifacts = {
        'isolation_forest.pkl': iso,
        'kmeans.pkl':           kmeans,
        'gradient_boosting.pkl': gb,
        'scaler.pkl':           scaler,
    }
    for fname, obj in artifacts.items():
        path = os.path.join(model_dir, fname)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"[OK] Guardado: {path}")

    with open(os.path.join(model_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(FEATURE_COLS))

    print(f"\nEntrenamiento completado. Modelos en: {model_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train',     type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    args = parser.parse_args()
    train(args.train, args.model_dir)
