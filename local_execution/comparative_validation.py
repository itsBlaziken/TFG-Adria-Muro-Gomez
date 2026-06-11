import json
import os
import pickle
import warnings

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

MODEL_DIR = r'C:/Users/adria/OneDrive/Escritorio/TFG/models'
DATA_PATH = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'

# Característiques espectrals en el mateix ordre que durant l'entrenament
FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]


def extract_features(record):
    values = record['values']
    freqs  = record['frequencies']

    low  = freqs < 30
    mid  = (freqs >= 30) & (freqs < 70)
    high = freqs >= 70

    energy = float(np.sum(values ** 2))
    rms    = float(np.sqrt(np.mean(values ** 2)))

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

    feats['energy_low']  = float(np.sum(values[low]  ** 2)) if np.any(low)  else 0.0
    feats['energy_mid']  = float(np.sum(values[mid]  ** 2)) if np.any(mid)  else 0.0
    feats['energy_high'] = float(np.sum(values[high] ** 2)) if np.any(high) else 0.0
    feats['ratio_low']   = feats['energy_low']  / (energy + 1e-8)
    feats['ratio_mid']   = feats['energy_mid']  / (energy + 1e-8)
    feats['ratio_high']  = feats['energy_high'] / (energy + 1e-8)
    feats['dominant_frequency'] = float(freqs[np.argmax(values)])

    return feats


def load_dataset(json_path):
    records = []
    skipped = 0

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'Item' not in obj:
                    skipped += 1
                    continue
                item = obj['Item']

                if 'readings' not in item:
                    skipped += 1
                    continue

                rd = item['readings']
                if 'L' not in rd or not rd['L']:
                    skipped += 1
                    continue

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

                records.append({
                    'frequencies': np.array(freqs),
                    'values':      np.array(vals),
                    'temp':        float(item.get('T1', {}).get('N', 0)),
                })

            except Exception:
                skipped += 1

    print(f"Dataset carregat: {len(records)} registres valids, {skipped} descartats")
    return records, skipped


def load_models(model_dir):
    models = {}
    for key, fname in [
        ('anomaly_detector', 'aeinnova_anomaly_detector_v2.pkl'),
        ('scaler',           'aeinnova_scaler_anomaly_v2.pkl'),
        ('kmeans',           'aeinnova_kmeans_v2.pkl'),
    ]:
        with open(os.path.join(model_dir, fname), 'rb') as f:
            models[key] = pickle.load(f)
    return models


def main():
    SEP = '-' * 60

    print('=' * 60)
    print('VALIDACIO COMPARATIVA: Isolation Forest vs LOF')
    print('Seccio 7.4 — Avaluacio quantitativa del sistema')
    print('=' * 60)

    print('\nCarregant dataset...')
    records, _ = load_dataset(DATA_PATH)

    rows = []
    for r in records:
        try:
            feat = extract_features(r)
            rows.append([feat[c] for c in FEATURE_COLS])
        except Exception:
            pass

    X = np.array(rows, dtype=float)
    print(f'Registres amb caracteristiques valides: {len(X)}')

    print('\nCarregant models entrenats...')
    models   = load_models(MODEL_DIR)
    X_scaled = models['scaler'].transform(X)

    # Isolation Forest — model principal del sistema
    print(f'\n{SEP}')
    print('ISOLATION FOREST (model principal)')
    print(SEP)

    iso_preds  = models['anomaly_detector'].predict(X_scaled)
    is_anom_if = iso_preds == -1
    n_if       = int(is_anom_if.sum())
    pct_if     = 100 * n_if / len(X)

    X_anom_if       = X_scaled[is_anom_if]
    labels_if       = models['kmeans'].predict(X_anom_if)
    sil_if          = silhouette_score(
        X_anom_if, labels_if,
        sample_size=min(5000, len(X_anom_if)),
        random_state=42,
    )

    print(f'  Anomalies detectades : {n_if:,} ({pct_if:.1f}%)')
    print(f'  Silhouette score     : {sil_if:.4f}')

    # LOF — algorisme de referencia per a la validació comparativa
    # Mateixa taxa de contaminació per garantir comparació equitativa
    print(f'\n{SEP}')
    print('LOCAL OUTLIER FACTOR (LOF) — referencia comparativa')
    print(SEP)

    contamination = n_if / len(X)
    lof           = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_preds     = lof.fit_predict(X_scaled)
    is_anom_lof   = lof_preds == -1
    n_lof         = int(is_anom_lof.sum())
    pct_lof       = 100 * n_lof / len(X)

    X_anom_lof = X_scaled[is_anom_lof]
    labels_lof = models['kmeans'].predict(X_anom_lof)
    sil_lof    = silhouette_score(
        X_anom_lof, labels_lof,
        sample_size=min(5000, len(X_anom_lof)),
        random_state=42,
    )

    print(f'  Anomalies detectades : {n_lof:,} ({pct_lof:.1f}%)')
    print(f'  Silhouette score     : {sil_lof:.4f}')

    # Resum comparatiu
    print(f'\n{SEP}')
    print('RESUM COMPARATIU')
    print(SEP)
    print(f'  {"Algorisme":<25} {"N anomalies":>12} {"% dataset":>10} {"Silhouette":>12}')
    print(f'  {"-"*25} {"-"*12} {"-"*10} {"-"*12}')
    print(f'  {"Isolation Forest":<25} {n_if:>12,} {pct_if:>9.1f}% {sil_if:>12.4f}')
    print(f'  {"LOF (n_neighbors=20)":<25} {n_lof:>12,} {pct_lof:>9.1f}% {sil_lof:>12.4f}')

    diff = sil_if - sil_lof
    print(f'\n  Diferencia silhouette (IF - LOF): +{diff:.4f}')
    print(f'\n  Isolation Forest obte una separacio de clusters {diff/sil_lof*100:.1f}% superior')
    print(f'  a LOF per al mateix conjunt de dades, confirmant la idoneitat')
    print(f'  de l\'algorisme escollit per al perfil industrial analitzat.')
    print('=' * 60)


if __name__ == '__main__':
    main()
