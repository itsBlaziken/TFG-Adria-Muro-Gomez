# SageMaker Training Job
# Input:  SM_CHANNEL_TRAIN / /opt/ml/input/data/train/features.csv
# Output: SM_MODEL_DIR     / /opt/ml/model/

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Característiques espectrals + mode operatiu i eix físic.
# Mode i eix s'inclouen perquè el rang de freqüències mesurat varia amb el mode;
# sense ells el model confon patrons de freqüència de modes diferents.
FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
    'mode', 'axis',
]


def train(df, model_dir):
    X = df[FEATURE_COLS].fillna(0).values

    # Normalització de les característiques per donar la mateixa escala a totes les variables
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Detecció d'anomalies amb Isolation Forest: contaminació adaptativa entre el 10% i el 20%
    contamination = min(0.20, max(0.10, len(X) / 10000))
    iso   = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
    preds = iso.fit_predict(X_scaled)

    mask   = preds == -1
    n_anom = int(mask.sum())
    print(f"Anomalies detectades: {n_anom} ({100 * n_anom / len(X):.1f}%)")

    X_anom = X_scaled[mask]

    # Agrupació de les anomalies en 4 tipologies de fallo amb K-Means
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_anom)

    for u, c in zip(*np.unique(labels, return_counts=True)):
        print(f"  Cluster {u}: {c} mostres ({100 * c / len(labels):.1f}%)")

    # Validació creuada de 5 folds del classificador Gradient Boosting
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (tr, te) in enumerate(skf.split(X_anom, labels), 1):
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        max_depth=5, random_state=42)
        gb.fit(X_anom[tr], labels[tr])
        score = accuracy_score(labels[te], gb.predict(X_anom[te]))
        cv_scores.append(score)
        print(f"  Fold {fold}: {score * 100:.2f}%")

    print(f"CV Accuracy: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%")

    # Entrenament final del classificador amb totes les anomalies
    gb_final = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                          max_depth=5, random_state=42)
    gb_final.fit(X_anom, labels)

    # Avaluació sobre el conjunt d'entrenament i importància de les característiques
    y_pred = gb_final.predict(X_anom)
    print(f"Train Accuracy: {accuracy_score(labels, y_pred) * 100:.2f}%")
    print(f"F1-Score (macro): {f1_score(labels, y_pred, average='macro') * 100:.2f}%")
    print(f"\nConfusion Matrix:\n{confusion_matrix(labels, y_pred)}")
    print(f"\nClassification Report:\n{classification_report(labels, y_pred)}")

    top_idx = np.argsort(gb_final.feature_importances_)[-5:][::-1]
    print("Top 5 features per importància:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {FEATURE_COLS[idx]}: {gb_final.feature_importances_[idx] * 100:.1f}%")

    # Serialització dels quatre models i la llista de features al directori de sortida
    os.makedirs(model_dir, exist_ok=True)
    for fname, obj in [
        ('aeinnova_anomaly_detector.pkl',     iso),
        ('aeinnova_fault_type_classifier.pkl', gb_final),
        ('aeinnova_scaler_anomaly.pkl',        scaler),
        ('aeinnova_kmeans.pkl',                kmeans),
    ]:
        with open(os.path.join(model_dir, fname), 'wb') as f:
            pickle.dump(obj, f)

    with open(os.path.join(model_dir, 'aeinnova_feature_names.txt'), 'w') as f:
        f.write('\n'.join(FEATURE_COLS))

    print(f"Models guardats a: {model_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train',     type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    args = parser.parse_args()

    # Llegeix el CSV de característiques del canal d'entrada de SageMaker
    df = pd.read_csv(os.path.join(args.train, 'features.csv'))
    print(f"Característiques carregades: {len(df)} registres")

    train(df, args.model_dir)
