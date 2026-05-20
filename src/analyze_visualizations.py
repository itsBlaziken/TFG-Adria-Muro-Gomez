"""
Script de análisis exploratorio con visualizaciones
Genera gráficos para documentación de defensa (Sección 6.3)

Figuras generadas:
- Fig 2: Señal de vibración en dominio temporal
- Fig 3: Espectro de frecuencia del señal
- Fig 4: Evolución del anomaly score
- Fig 5: Detección de anomalías en el señal
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.ensemble import IsolationForest
import sys
import io

# Fijar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configurar matplotlib para mejor calidad
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['font.size'] = 10

# Crear directorio de salida
output_dir = Path('C:/Users/adria/OneDrive/Escritorio/TFG/doc/figuras_validacion')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("ANÁLISIS EXPLORATORIO - GENERACIÓN DE VISUALIZACIONES")
print("="*70)

# ==============================================================================
# PASO 1: Cargar datos
# ==============================================================================
print("\n[PASO 1] Cargando datos del dataset...")

try:
    with open('C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json', 'r') as f:
        lines = f.readlines()
    print(f"[OK] Archivo cargado: {len(lines)} líneas")
except Exception as e:
    print(f"[ERROR] Error: {e}")
    exit(1)

# Parsear datos (formato DynamoDB)
readings_data = []
count = 0

for line in lines[:5000]:  # Procesar primeros 5000 para análisis
    try:
        item = json.loads(line.strip())
        if 'Item' in item and 'readings' in item['Item']:
            readings_list = item['Item']['readings'].get('L', [])
            
            # Extraer frequency-value pairs
            if readings_list:
                frequencies = []
                values = []
                
                for reading in readings_list:
                    if 'M' in reading:
                        freq = float(reading['M'].get('frequency', {}).get('N', 0))
                        val = float(reading['M'].get('value', {}).get('N', 0))
                        frequencies.append(freq)
                        values.append(val)
                
                if len(frequencies) > 10:  # Al menos 10 puntos
                    readings_data.append({
                        'frequencies': np.array(frequencies),
                        'values': np.array(values),
                        'timestamp': item['Item'].get('timestamp', {}).get('N', 0)
                    })
                    count += 1
    except:
        continue

print(f"[OK] Registros con readings extraidos: {count}")

if not readings_data:
    print("[ERROR] No hay datos para visualizar")
    exit(1)

# ==============================================================================
# FIGURA 2: Señal en dominio temporal
# ==============================================================================
print("\n[PASO 2] Generando Fig 2: Señal en dominio temporal...")

# Tomar varios registros y concatenarlos para simular serie temporal
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Subplot 1: Señal temporal simple
sample_idx = np.random.randint(0, len(readings_data))
sample = readings_data[sample_idx]
frequencies = sample['frequencies']
time_indices = np.arange(len(sample['values']))

ax1.plot(time_indices, sample['values'], linewidth=1.5, color='steelblue', alpha=0.8)
ax1.fill_between(time_indices, sample['values'], alpha=0.3, color='steelblue')
ax1.set_xlabel('Temps (mostres)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Amplitud (g)', fontsize=11, fontweight='bold')
ax1.set_title('Senyal de vibració en el domini temporal', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(sample['values']) * 1.1])

# Subplot 2: Múltiples registros superpuestos
for i in range(min(5, len(readings_data))):
    sample = readings_data[i]
    ax2.plot(np.arange(len(sample['values'])), sample['values'], 
             alpha=0.6, linewidth=1, label=f'Registre {i+1}')

ax2.set_xlabel('Temps (mostres)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Amplitud (g)', fontsize=11, fontweight='bold')
ax2.set_title('Comparativa de múltiples senyals de vibració', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'Fig_2_senal_temporal.png', dpi=150, bbox_inches='tight')
print(f"[OK] Guardada: Fig_2_senal_temporal.png")
plt.close()

# ==============================================================================
# FIGURA 3: Espectro de frecuencia
# ==============================================================================
print("\n[PASO 3] Generando Fig 3: Espectro de frecuencia...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx in range(4):
    sample_idx = np.random.randint(0, len(readings_data))
    sample = readings_data[sample_idx]
    frequencies = sample['frequencies']
    values = sample['values']
    
    ax = axes[idx]
    ax.bar(frequencies, values, width=0.5, color='darkgreen', alpha=0.7, edgecolor='darkgreen')
    ax.fill_between(frequencies, values, alpha=0.3, color='green')
    
    ax.set_xlabel('Freqüència (Hz)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Amplitud (g)', fontsize=10, fontweight='bold')
    ax.set_title(f'Espectre de freqüència - Registre {idx+1}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    # Destacar frecuencia dominante
    if len(values) > 0:
        dominant_idx = np.argmax(values)
        dominant_freq = frequencies[dominant_idx]
        dominant_val = values[dominant_idx]
        ax.plot(dominant_freq, dominant_val, 'r*', markersize=15, label=f'Dominante: {dominant_freq:.1f} Hz')
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'Fig_3_espectro_frecuencia.png', dpi=150, bbox_inches='tight')
print(f"[OK] Guardada: Fig_3_espectro_frecuencia.png")
plt.close()

# ==============================================================================
# FIGURA 4 & 5: Anomaly scores y detección
# ==============================================================================
print("\n[PASO 4] Extrayendo características espectrales...")

# Función para extraer features (igual que en train_fault_types_v2.py)
def extract_spectral_features(frequencies, values):
    """Extrae 15 características espectrales"""
    values = np.array(values, dtype=float)
    frequencies = np.array(frequencies, dtype=float)
    
    # Bandas de frecuencia
    low_band = (frequencies < 20)
    mid_band = (frequencies >= 20) & (frequencies < 50)
    high_band = (frequencies >= 50)
    
    features = [
        np.sqrt(np.mean(values ** 2)),  # rms
        np.sum(values ** 2),  # energy
        np.max(values),  # max_amplitude
        np.mean(values),  # mean_amplitude
        np.std(values),  # std_amplitude
        np.max(values) - np.min(values),  # peak_to_peak
        np.max(values) / (np.sqrt(np.mean(values ** 2)) + 1e-10),  # crest_factor
        20.0,  # temp (default)
        np.sum(values[low_band] ** 2),  # energy_low
        np.sum(values[mid_band] ** 2),  # energy_mid
        np.sum(values[high_band] ** 2),  # energy_high
        np.sum(values[low_band] ** 2) / (np.sum(values ** 2) + 1e-10),  # ratio_low
        np.sum(values[mid_band] ** 2) / (np.sum(values ** 2) + 1e-10),  # ratio_mid
        np.sum(values[high_band] ** 2) / (np.sum(values ** 2) + 1e-10),  # ratio_high
        frequencies[np.argmax(values)] if len(frequencies) > 0 else 0  # dominant_frequency
    ]
    
    return np.array(features)

# Extraer features de todos los registros (muestreo aleatorio)
print("Extrayendo características de todos los registros...")
features_list = []
indices_for_features = []

# Usar muestreo aleatorio en lugar de solo los primeros
max_samples = min(1000, len(readings_data))
random_indices = np.random.choice(len(readings_data), max_samples, replace=False)

for idx in random_indices:
    sample = readings_data[idx]
    features = extract_spectral_features(sample['frequencies'], sample['values'])
    features_list.append(features)
    indices_for_features.append(idx)

features_array = np.array(features_list)
print(f"[OK] Features extraidas: shape {features_array.shape}")

# Normalizar para Isolation Forest
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

# Entrenar Isolation Forest
print("Entrenando Isolation Forest...")
iso_forest = IsolationForest(contamination=0.20, random_state=42)
anomaly_predictions = iso_forest.fit_predict(features_scaled)
anomaly_scores = iso_forest.score_samples(features_scaled)

print(f"[OK] Anomalias detectadas: {sum(anomaly_predictions == -1)} ({100*sum(anomaly_predictions == -1)/len(anomaly_predictions):.1f}%)")

# ==============================================================================
# FIGURA 4: Evolución del anomaly score
# ==============================================================================
print("\n[PASO 5] Generando Fig 4: Evolución del anomaly score...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Calcular threshold dinámico (mediana entre normal y anomalía)
normal_scores = anomaly_scores[anomaly_predictions == 1]
anomaly_scores_list = anomaly_scores[anomaly_predictions == -1]
dynamic_threshold = (np.max(anomaly_scores_list) + np.min(normal_scores)) / 2

# Subplot 1: Anomaly score vs índice
sample_indices = np.arange(len(anomaly_scores))
colors = ['red' if x == -1 else 'blue' for x in anomaly_predictions]

ax1.scatter(sample_indices, anomaly_scores, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax1.axhline(y=dynamic_threshold, color='green', linestyle='--', linewidth=2.5, label=f'Threshold: {dynamic_threshold:.3f}')
ax1.set_xlabel('Índex de registre', fontsize=11, fontweight='bold')
ax1.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
ax1.set_title('Evolució del score d\'anomalia - Isolation Forest', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(['Normal (blau)', 'Anomalia (vermell)', f'Threshold: {dynamic_threshold:.3f}'], fontsize=10)

# Subplot 2: Histograma
ax2.hist(anomaly_scores[anomaly_predictions == 1], bins=30, alpha=0.7, 
         color='blue', label='Normal', edgecolor='black')
ax2.hist(anomaly_scores[anomaly_predictions == -1], bins=30, alpha=0.7, 
         color='red', label='Anomalia', edgecolor='black')
ax2.axvline(x=dynamic_threshold, color='green', linestyle='--', linewidth=2.5, label=f'Threshold: {dynamic_threshold:.3f}')
ax2.set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Freqüència', fontsize=11, fontweight='bold')
ax2.set_title('Distribució de scores d\'anomalia', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'Fig_4_anomaly_score.png', dpi=150, bbox_inches='tight')
print(f"[OK] Guardada: Fig_4_anomaly_score.png")
plt.close()

# ==============================================================================
# FIGURA 5: Detección de anomalías
# ==============================================================================
print("\n[PASO 6] Generando Fig 5: Detección de anomalías...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Seleccionar ejemplos COHERENTES:
# - Normales: los con scores MÁS altos (menos anómalos)
# - Anómalos: los con scores MÁS bajos (más anómalos)

normal_indices = np.where(anomaly_predictions == 1)[0]
anomaly_indices = np.where(anomaly_predictions == -1)[0]

# Ordenar por score para obtener los más claros
normal_sorted = normal_indices[np.argsort(anomaly_scores[normal_indices])[::-1]]  # Descendente (más altos)
anomaly_sorted = anomaly_indices[np.argsort(anomaly_scores[anomaly_indices])]     # Ascendente (más bajos)

# Tomar los 2 más claros de cada grupo
selected_normal = normal_sorted[:2]
selected_anomaly = anomaly_sorted[:2]
selected_indices = list(selected_normal) + list(selected_anomaly)
labels = ['Normal', 'Normal', 'Anomalia', 'Anomalia']

for plot_idx, (data_idx, label) in enumerate(zip(selected_indices, labels)):
    # Mapear índice de features a índice de datos originales
    original_idx = indices_for_features[data_idx] if data_idx < len(indices_for_features) else data_idx
    
    if original_idx < len(readings_data):
        sample = readings_data[original_idx]
        frequencies = sample['frequencies']
        values = sample['values']
        
        ax = axes[plot_idx]
        
        if 'Anomalia' in label:
            color = 'red'
            alpha = 0.8
        else:
            color = 'green'
            alpha = 0.6
        
        ax.bar(frequencies, values, width=0.5, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Freqüència (Hz)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Amplitud (g)', fontsize=10, fontweight='bold')
        ax.set_title(f'{label} - Score: {anomaly_scores[data_idx]:.3f}', 
                     fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'Fig_5_deteccion_anomalias.png', dpi=150, bbox_inches='tight')
print(f"[OK] Guardada: Fig_5_deteccion_anomalias.png")
plt.close()

# ==============================================================================
# RESUMEN
# ==============================================================================
print("\n" + "="*70)
print("RESUMEN - ANÁLISIS COMPLETADO")
print("="*70)
print(f"""
Registros procesados:     {len(readings_data)}
Features extraídas:       {features_array.shape}
Anomalías detectadas:     {sum(anomaly_predictions == -1)} ({100*sum(anomaly_predictions == -1)/len(anomaly_predictions):.1f}%)
Normales clasificadas:    {sum(anomaly_predictions == 1)} ({100*sum(anomaly_predictions == 1)/len(anomaly_predictions):.1f}%)

FIGURAS GENERADAS:
  ✓ Fig_2_senal_temporal.png       - Señal en dominio temporal
  ✓ Fig_3_espectro_frecuencia.png  - Espectros de frecuencia
  ✓ Fig_4_anomaly_score.png        - Evolución de anomaly score
  ✓ Fig_5_deteccion_anomalias.png  - Detección de anomalías

Ubicación: {output_dir}

INSTRUCCIONES PARA USAR EN DEFENSA:
1. Usa estas figuras en la Sección 6.3 del informe
2. Fig 2: Explica variaciones temporales
3. Fig 3: Muestra distribución de energía en frecuencia
4. Fig 4: Justifica el threshold del modelo
5. Fig 5: Ejemplos visuales de normal vs anomalia
""")

print("✓ Script completado exitosamente")
print("="*70 + "\n")
