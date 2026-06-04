import json
import os
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

# Rutes locals del dataset i del directori on es desaran els models
DATA_PATH = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'
MODEL_DIR = r'C:/Users/adria/OneDrive/Escritorio/TFG/models'

# Característiques espectrals + mode operatiu i eix físic.
# Mode i eix s'inclouen perquè el rang de freqüències mesurat varia amb el mode.
FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
    'mode', 'axis',
]


def extract_features(record):
    values = record['values']
    freqs  = record['frequencies']

    # Definició de les bandes de freqüència: baixa (<30 Hz), mitjana (30-70 Hz), alta (>70 Hz)
    low  = freqs < 30
    mid  = (freqs >= 30) & (freqs < 70)
    high = freqs >= 70

    energy = float(np.sum(values ** 2))
    rms    = float(np.sqrt(np.mean(values ** 2)))

    # Estadístics globals del senyal i factor de cresta
    feats = {
        'rms':            rms,
        'energy':         energy,
        'max_amplitude':  float(np.max(values)),
        'mean_amplitude': float(np.mean(values)),
        'std_amplitude':  float(np.std(values)),
        'peak_to_peak':   float(np.max(values) - np.min(values)),
        'crest_factor':   float(np.max(values) / (rms + 1e-8)),
        'temp':           record['temp'],
    }

    # Energia absoluta per banda i ràtio relatiu sobre l'energia total
    feats['energy_low']  = float(np.sum(values[low]  ** 2)) if np.any(low)  else 0.0
    feats['energy_mid']  = float(np.sum(values[mid]  ** 2)) if np.any(mid)  else 0.0
    feats['energy_high'] = float(np.sum(values[high] ** 2)) if np.any(high) else 0.0
    feats['ratio_low']   = feats['energy_low']  / (energy + 1e-8)
    feats['ratio_mid']   = feats['energy_mid']  / (energy + 1e-8)
    feats['ratio_high']  = feats['energy_high'] / (energy + 1e-8)

    # Freqüència dominant: la freqüència amb major amplitud
    feats['dominant_frequency'] = float(freqs[np.argmax(values)])

    return feats


def load_dataset(json_path):
    records = []
    skipped = 0

    # Llegeix el dataset JSON línia per línia (format DynamoDB Export)
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'Item' not in obj:
                    skipped += 1
                    continue
                item = obj['Item']

                # Descarta els registres sense lectures espectrals
                if 'readings' not in item:
                    skipped += 1
                    continue

                rd = item['readings']
                if 'L' not in rd or not rd['L']:
                    skipped += 1
                    continue

                # Extreu les parelles freqüència-amplitud de cada lectura
                freqs, vals = [], []
                for entry in rd['L']:
                    if 'M' in entry:
                        m = entry['M']
                        if 'frequency' in m and 'value' in m:
                            try:
                                freqs.append(float(m['frequency']['N']))
                                vals.append(float(m['value']['N']))
                            except (KeyError, ValueError):
                                pass

                if not vals:
                    skipped += 1
                    continue

                # Afegeix el registre amb les metadades del dispositiu
                records.append({
                    'frequencies': np.array(freqs),
                    'values':      np.array(vals),
                    'dev_eui':     item.get('dev_eui',   {}).get('S', 'unknown'),
                    'mode':        float(item.get('mode',      {}).get('N', -1)),
                    'axis':        float(item.get('axis',      {}).get('N', -1)),
                    'timestamp':   float(item.get('timestamp', {}).get('N',  0)),
                    'temp':        float(item.get('T1',        {}).get('N',  0)),
                })
            except Exception:
                skipped += 1

    print(f"Dataset carregat: {len(records)} registres vàlids, {skipped} descartats")
    return records


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
    # Càrrega del dataset i extracció de característiques espectrals
    print("Carregant dataset...")
    records = load_dataset(DATA_PATH)

    if not records:
        raise RuntimeError("Cap registre vàlid al dataset")

    print("Extraient característiques espectrals...")
    rows = []
    for r in records:
        try:
            feat = extract_features(r)
            feat.update({'dev_eui': r['dev_eui'], 'mode': r['mode'],
                         'axis': r['axis'], 'timestamp': r['timestamp']})
            rows.append(feat)
        except Exception:
            pass

    df = pd.DataFrame(rows).fillna(0).replace([np.inf, -np.inf], 0)
    print(f"Característiques extretes: {len(df)} registres, {len(FEATURE_COLS)} features")

    # Entrenament del pipeline complet i desament dels models
    train(df, MODEL_DIR)
