import json
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Qualitat de les imatges generades
matplotlib.rcParams['figure.dpi']  = 150
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['font.size']   = 10

# Rutes locals del dataset i del directori de sortida de figures
DATA_PATH  = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'
OUTPUT_DIR = Path(r'C:/Users/adria/OneDrive/Escritorio/TFG/doc/figuras_validacion')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_spectral_features(frequencies, values):
    # Extreu el mateix vector de 15 característiques que utilitza el model d'entrenament
    frequencies = np.array(frequencies, dtype=float)
    values      = np.array(values,      dtype=float)

    low  = frequencies < 30
    mid  = (frequencies >= 30) & (frequencies < 70)
    high = frequencies >= 70
    energy = np.sum(values ** 2) + 1e-10

    return [
        np.sqrt(np.mean(values ** 2)),
        np.sum(values ** 2),
        np.max(values),
        np.mean(values),
        np.std(values),
        np.max(values) - np.min(values),
        np.max(values) / (np.sqrt(np.mean(values ** 2)) + 1e-10),
        20.0,
        np.sum(values[low]  ** 2),
        np.sum(values[mid]  ** 2),
        np.sum(values[high] ** 2),
        np.sum(values[low]  ** 2) / energy,
        np.sum(values[mid]  ** 2) / energy,
        np.sum(values[high] ** 2) / energy,
        frequencies[np.argmax(values)] if len(frequencies) > 0 else 0,
    ]


# Carrega els primers 5000 registres del dataset per a l'anàlisi exploratòria
print("Carregant dataset...")
readings_data = []
with open(DATA_PATH, 'r') as f:
    for line in f.readlines()[:5000]:
        try:
            item = json.loads(line.strip())
            if 'Item' not in item or 'readings' not in item['Item']:
                continue
            rd = item['Item']['readings'].get('L', [])
            if not rd:
                continue
            freqs, vals = [], []
            for reading in rd:
                if 'M' in reading:
                    freqs.append(float(reading['M'].get('frequency', {}).get('N', 0)))
                    vals.append(float(reading['M'].get('value',     {}).get('N', 0)))
            if len(freqs) > 10:
                readings_data.append({'frequencies': np.array(freqs), 'values': np.array(vals)})
        except Exception:
            continue

print(f"Registres carregats: {len(readings_data)}")


# Figura 1 — Senyal de vibració en el domini temporal
# Mostra un registre individual i la comparativa de cinc registres superposats
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

sample = readings_data[np.random.randint(0, len(readings_data))]
ax1.plot(np.arange(len(sample['values'])), sample['values'],
         linewidth=1.5, color='steelblue', alpha=0.8)
ax1.fill_between(np.arange(len(sample['values'])), sample['values'], alpha=0.3, color='steelblue')
ax1.set_xlabel('Temps (mostres)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Amplitud (g)',    fontsize=11, fontweight='bold')
ax1.set_title('Senyal de vibració en el domini temporal', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

for i in range(min(5, len(readings_data))):
    s = readings_data[i]
    ax2.plot(np.arange(len(s['values'])), s['values'], alpha=0.6, linewidth=1, label=f'Registre {i+1}')
ax2.set_xlabel('Temps (mostres)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Amplitud (g)',    fontsize=11, fontweight='bold')
ax2.set_title('Comparativa de múltiples senyals de vibració', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_2_senal_temporal.png', bbox_inches='tight')
plt.close()
print("Guardada: Fig_2_senal_temporal.png")


# Figura 2 — Espectres de freqüència de quatre registres aleatoris
# Destaca la freqüència dominant de cada espectre amb una estrella vermella
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, ax in enumerate(axes.flatten()):
    s       = readings_data[np.random.randint(0, len(readings_data))]
    dom_idx = np.argmax(s['values'])
    ax.bar(s['frequencies'], s['values'], width=0.5, color='darkgreen', alpha=0.7)
    ax.fill_between(s['frequencies'], s['values'], alpha=0.3, color='green')
    ax.plot(s['frequencies'][dom_idx], s['values'][dom_idx], 'r*', markersize=15,
            label=f"Dominant: {s['frequencies'][dom_idx]:.1f} Hz")
    ax.set_xlabel('Freqüència (Hz)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Amplitud (g)',    fontsize=10, fontweight='bold')
    ax.set_title(f'Espectre de freqüència - Registre {idx+1}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_3_espectro_frecuencia.png', bbox_inches='tight')
plt.close()
print("Guardada: Fig_3_espectro_frecuencia.png")


# Extracció de característiques i entrenament de l'Isolation Forest per a les figures 3 i 4
print("Extraient característiques per al model...")
n_samples = min(1000, len(readings_data))
indices   = np.random.choice(len(readings_data), n_samples, replace=False)
feats     = [extract_spectral_features(readings_data[i]['frequencies'], readings_data[i]['values'])
             for i in indices]

X      = StandardScaler().fit_transform(np.array(feats))
iso    = IsolationForest(contamination=0.20, random_state=42)
preds  = iso.fit_predict(X)
scores = iso.score_samples(X)

n_anom = int((preds == -1).sum())
print(f"Anomalies detectades: {n_anom} ({100 * n_anom / len(preds):.1f}%)")


# Figura 3 — Distribució de l'anomaly score i separació entre classes
# Scatter dels scores per índex (superior) i histograma de les dues classes (inferior)
normal_sc = scores[preds ==  1]
anom_sc   = scores[preds == -1]
threshold = (np.max(anom_sc) + np.min(normal_sc)) / 2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

colors = ['red' if p == -1 else 'blue' for p in preds]
ax1.scatter(np.arange(len(scores)), scores, c=colors, alpha=0.6, s=30,
            edgecolors='black', linewidth=0.5)
ax1.axhline(y=threshold, color='green', linestyle='--', linewidth=2.5,
            label=f'Threshold: {threshold:.3f}')
ax1.set_xlabel('Índex de registre', fontsize=11, fontweight='bold')
ax1.set_ylabel('Anomaly Score',     fontsize=11, fontweight='bold')
ax1.set_title("Evolució del score d'anomalia - Isolation Forest", fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.hist(normal_sc, bins=30, alpha=0.7, color='blue',  label='Normal',   edgecolor='black')
ax2.hist(anom_sc,   bins=30, alpha=0.7, color='red',   label='Anomalia', edgecolor='black')
ax2.axvline(x=threshold, color='green', linestyle='--', linewidth=2.5,
            label=f'Threshold: {threshold:.3f}')
ax2.set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Freqüència',    fontsize=11, fontweight='bold')
ax2.set_title("Distribució de scores d'anomalia", fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_4_anomaly_score.png', bbox_inches='tight')
plt.close()
print("Guardada: Fig_4_anomaly_score.png")


# Figura 4 — Exemples de senyals normals i anòmals
# Selecciona els dos casos més clars de cada classe per màxima separació visual
normal_idx  = np.where(preds ==  1)[0]
anom_idx    = np.where(preds == -1)[0]
best_normal = normal_idx[np.argsort(scores[normal_idx])[::-1]][:2]
best_anom   = anom_idx[np.argsort(scores[anom_idx])][:2]
selected    = list(best_normal) + list(best_anom)
labels      = ['Normal', 'Normal', 'Anomalia', 'Anomalia']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for plot_idx, (data_idx, label) in enumerate(zip(selected, labels)):
    orig_idx = indices[data_idx] if data_idx < len(indices) else data_idx
    if orig_idx >= len(readings_data):
        continue
    s     = readings_data[orig_idx]
    color = 'red' if 'Anomalia' in label else 'green'
    ax    = axes.flatten()[plot_idx]
    ax.bar(s['frequencies'], s['values'], width=0.5, color=color,
           alpha=0.8 if color == 'red' else 0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Freqüència (Hz)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Amplitud (g)',    fontsize=10, fontweight='bold')
    ax.set_title(f'{label} - Score: {scores[data_idx]:.3f}', fontsize=11,
                 fontweight='bold', color=color)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig_5_deteccion_anomalias.png', bbox_inches='tight')
plt.close()
print("Guardada: Fig_5_deteccion_anomalias.png")

print(f"\nTotes les figures guardades a: {OUTPUT_DIR}")
