import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATASET_PATH = r'data\DatasetI.json'

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
    low  = freqs < 30
    mid  = (freqs >= 30) & (freqs < 70)
    high = freqs >= 70
    energy = float(np.sum(values ** 2))
    rms    = float(np.sqrt(np.mean(values ** 2)))
    feats = {
        'rms': rms, 'energy': energy,
        'max_amplitude': float(np.max(values)),
        'mean_amplitude': float(np.mean(values)),
        'std_amplitude': float(np.std(values)),
        'peak_to_peak': float(np.max(values) - np.min(values)),
        'crest_factor': float(np.max(values) / (rms + 1e-8)),
        'temp': record['temp'],
    }
    feats['energy_low']  = float(np.sum(values[low]  ** 2)) if np.any(low)  else 0.0
    feats['energy_mid']  = float(np.sum(values[mid]  ** 2)) if np.any(mid)  else 0.0
    feats['energy_high'] = float(np.sum(values[high] ** 2)) if np.any(high) else 0.0
    total = energy + 1e-8
    feats['ratio_low']  = feats['energy_low']  / total
    feats['ratio_mid']  = feats['energy_mid']  / total
    feats['ratio_high'] = feats['energy_high'] / total
    feats['dominant_frequency'] = float(freqs[np.argmax(values)])
    return feats

print("=" * 60)
print("BENCHMARK PIPELINE AEINNOVA")
print("=" * 60)

# ─── ETAPA 1: Preprocessament i extracció de característiques ───
t0 = time.perf_counter()

records = []
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            if 'Item' not in obj:
                continue
            item = obj['Item']
            if 'readings' not in item:
                continue
            rd = item['readings']
            if 'L' not in rd or not rd['L']:
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
                continue
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
            pass

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
n_records = len(df)

t1 = time.perf_counter()
time_preprocess = t1 - t0
print(f"[1] Preprocessament i extracció de característiques: {time_preprocess:.2f} s  ({n_records} registres)")

# ─── ETAPA 2: Detecció d'anomalies (Isolation Forest) ───
X = df[FEATURE_COLS].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

t2_start = time.perf_counter()
contamination = min(0.20, max(0.10, n_records / 10000))
iso   = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
preds = iso.fit_predict(X_scaled)
mask  = preds == -1
n_anom = int(mask.sum())
t2_end = time.perf_counter()
time_iso = t2_end - t2_start
print(f"[2] Detecció d'anomalies (Isolation Forest):         {time_iso:.2f} s  ({n_anom} anomalies, {100*n_anom/n_records:.1f}%)")

# ─── ETAPA 3: Clustering K-Means ───
X_anom = X_scaled[mask]

t3_start = time.perf_counter()
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_anom)
t3_end = time.perf_counter()
time_kmeans = t3_end - t3_start
print(f"[3] Clustering K-Means (k=4):                        {time_kmeans:.2f} s")

# ─── ETAPA 4: Classificació (Gradient Boosting) ───
t4_start = time.perf_counter()
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=5, random_state=42)
gb.fit(X_anom, labels)
t4_end = time.perf_counter()
time_gb = t4_end - t4_start
print(f"[4] Classificació Gradient Boosting:                 {time_gb:.2f} s")

time_total = t1 - t0 + time_iso + time_kmeans + time_gb
print("-" * 60)
print(f"    Pipeline complet:                                {time_total:.2f} s")
print("=" * 60)

print("\n=== TAULA PER A L'INFORME ===")
print(f"Preprocessament i extracció de característiques | {time_preprocess:.1f} s")
print(f"Detecció d'anomalies (Isolation Forest)         | {time_iso:.1f} s")
print(f"Clustering (K-Means)                            | {time_kmeans:.1f} s")
print(f"Classificació (Gradient Boosting)               | {time_gb:.1f} s")
print(f"Pipeline complet                                | {time_total:.1f} s")
