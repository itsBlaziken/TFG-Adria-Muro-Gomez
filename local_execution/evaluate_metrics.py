import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

MODEL_DIR = r'C:/Users/adria/OneDrive/Escritorio/TFG/models'
DATA_PATH = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'

FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]

FAULT_LABELS = {
    0: 'Inestabilitat estructural',
    1: 'Desalineacio',
    2: 'Desbalanceig mecanic',
    3: 'Degradacio general',
}


def load_models_v2(model_dir):
    models = {}
    for key, fname in [
        ('anomaly_detector', 'aeinnova_anomaly_detector_v2.pkl'),
        ('classifier',       'aeinnova_fault_type_classifier_v2.pkl'),
        ('scaler',           'aeinnova_scaler_anomaly_v2.pkl'),
        ('kmeans',           'aeinnova_kmeans_v2.pkl'),
    ]:
        with open(os.path.join(model_dir, fname), 'rb') as f:
            models[key] = pickle.load(f)

    with open(os.path.join(model_dir, 'aeinnova_feature_names_v2.txt'), 'r') as f:
        models['feature_names'] = [line.strip() for line in f]

    return models


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
                    'dev_eui':     item.get('dev_eui', {}).get('S', 'unknown'),
                    'temp':        float(item.get('T1', {}).get('N', 0)),
                })
            except Exception:
                skipped += 1
    return records, skipped


def main():
    SEP = "-" * 50
    print("=" * 60)
    print("AVALUACIO QUANTITATIVA DEL PIPELINE - seccio 7.4")
    print("=" * 60)

    print("\nCarregant dataset...")
    records, skipped = load_dataset(DATA_PATH)
    rows = []
    for r in records:
        try:
            feat = extract_features(r)
            rows.append([feat[c] for c in FEATURE_COLS])
        except Exception:
            pass

    X = np.array(rows, dtype=float)
    n_total = len(X)
    print(f"  Registres valids carregats : {n_total}")
    print(f"  Registres descartats       : {skipped}")

    print("\nCarregant models (v2)...")
    models = load_models_v2(MODEL_DIR)

    # 1. Anomaly ratio
    X_scaled = models['scaler'].transform(X)
    anomaly_preds = models['anomaly_detector'].predict(X_scaled)
    is_anomaly = anomaly_preds == -1
    n_anom = int(is_anomaly.sum())
    pct_anom = 100 * n_anom / n_total

    print("\n" + SEP)
    print("1. TAXA D'ANOMALIES DETECTADES")
    print(SEP)
    print(f"  Total registres analitzats : {n_total:,}")
    print(f"  Anomalies detectades       : {n_anom:,}")
    print(f"  Percentatge d'anomalies    : {pct_anom:.1f}%")
    print(f"  Registres normals          : {n_total - n_anom:,} ({100 - pct_anom:.1f}%)")

    # 2. Silhouette score
    X_anom = X_scaled[is_anomaly]
    cluster_labels = models['kmeans'].predict(X_anom)
    sil = silhouette_score(X_anom, cluster_labels,
                           sample_size=min(5000, len(X_anom)), random_state=42)

    print("\n" + SEP)
    print("2. QUALITAT DEL CLUSTERING (K-Means, K=4)")
    print(SEP)
    print(f"  Silhouette Score : {sil:.4f}")
    if sil >= 0.5:
        print("  Interpretacio    : bona separacio entre clusters")
    elif sil >= 0.3:
        print("  Interpretacio    : separacio moderada pero consistent")
    else:
        print("  Interpretacio    : separacio baixa (esperable en dades industrials reals)")

    print("\n  Distribucio de mostres per cluster:")
    for u, c in zip(*np.unique(cluster_labels, return_counts=True)):
        print(f"    Cluster {u}: {c:,} mostres ({100*c/len(cluster_labels):.1f}%)")

    # 3. Distribucio de tipologies de fallo
    fault_preds  = models['classifier'].predict(X_anom)
    fault_probas = models['classifier'].predict_proba(X_anom)
    mean_confidence = np.mean(np.max(fault_probas, axis=1))

    print("\n" + SEP)
    print("3. DISTRIBUCIO DE TIPOLOGIES DE FALLO")
    print(SEP)
    for ft, label in FAULT_LABELS.items():
        count = int((fault_preds == ft).sum())
        pct   = 100 * count / n_anom if n_anom > 0 else 0
        print(f"  Tipus {ft} - {label:<30}: {count:>5} ({pct:.1f}%)")
    print(f"\n  Confianca mitjana de classificacio: {mean_confidence * 100:.1f}%")

    # 4. Feature importance
    importances = models['classifier'].feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    print("\n" + SEP)
    print("4. IMPORTANCIA DE LES CARACTERISTIQUES (top 10)")
    print(SEP)
    for rank, idx in enumerate(sorted_idx[:10], 1):
        bar = '#' * int(importances[idx] * 100 / 2)
        print(f"  {rank:>2}. {FEATURE_COLS[idx]:<22}: {importances[idx]*100:>5.2f}%  {bar}")

    top3 = [FEATURE_COLS[i] for i in sorted_idx[:3]]

    print("\n" + "=" * 60)
    print("RESUM PER A LA SECCIO 7.4 DEL TFG")
    print("=" * 60)
    print(f"  * Total registres analitzats : {n_total:,}")
    print(f"  * Anomalies detectades       : {n_anom:,} ({pct_anom:.1f}%)")
    print(f"  * Silhouette score (K-Means) : {sil:.4f}")
    print(f"  * Confianca mitjana classif. : {mean_confidence * 100:.1f}%")
    print(f"  * Top 3 features             : {', '.join(top3)}")


if __name__ == '__main__':
    main()
