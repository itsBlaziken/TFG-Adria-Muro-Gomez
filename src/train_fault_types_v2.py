# Sistema de clasificacion de fallos en AEInnova - VERSION 2
# CORRECCIÓN: Ahora usa el formato "readings" (espectros de frecuencia)
# Entrada: 24,939 registros con readings (en lugar de solo 4,121 con px/py)
# Salida: 4 tipos de fallo con accuracy mejorado

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CLASIFICADOR DE FALLOS AEINNOVA")
print("=" * 80)

# PASO 1: CARGAR DATOS CON FORMATO "READINGS"
print("\n[PASO 1] Cargando datos de sensores con format 'readings'...")
json_path = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'

records = []
with open(json_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            if 'Item' in obj:
                item = obj['Item']
                
                # Solo toma registros con "readings"
                if 'readings' not in item:
                    continue
                
                readings_data = item['readings']
                if 'L' not in readings_data or len(readings_data['L']) == 0:
                    continue
                
                # Parsea los readings (frequency-value pairs)
                frequencies = []
                values = []
                for entry in readings_data['L']:
                    if 'M' in entry:
                        m = entry['M']
                        if 'frequency' in m and 'value' in m:
                            try:
                                freq = float(m['frequency']['N'])
                                val = float(m['value']['N'])
                                frequencies.append(freq)
                                values.append(val)
                            except:
                                pass
                
                if len(values) > 0:
                    record = {
                        'frequencies': np.array(frequencies),
                        'values': np.array(values),
                        'dev_eui': item.get('dev_eui', {}).get('S', 'unknown'),
                        'mode': float(item.get('mode', {}).get('N', -1)),
                        'axis': float(item.get('axis', {}).get('N', -1)),
                        'timestamp': float(item.get('timestamp', {}).get('N', 0)),
                    }
                    if 'T1' in item:
                        record['temp'] = float(item['T1']['N'])
                    else:
                        record['temp'] = 0
                    
                    records.append(record)
        except Exception as e:
            pass

print(f"[OK] Cargados {len(records)} registros con 'readings'")

if len(records) == 0:
    print("ERROR: No se cargaron registros con 'readings'")
    exit(1)

# PASO 2: EXTRAER CARACTERÍSTICAS DE LOS ESPECTROS
print("\n[PASO 2] Extrayendo características espectrales...")

features_list = []
valid_indices = []

for idx, record in enumerate(records):
    values = record['values']
    
    try:
        # Características espectrales
        feature_dict = {
            'rms': np.sqrt(np.mean(values ** 2)),
            'energy': np.sum(values ** 2),
            'max_amplitude': np.max(values),
            'mean_amplitude': np.mean(values),
            'std_amplitude': np.std(values),
            'peak_to_peak': np.max(values) - np.min(values),
            'crest_factor': np.max(values) / (np.sqrt(np.mean(values ** 2)) + 1e-8) if np.sqrt(np.mean(values ** 2)) > 0 else 0,
            'temp': record['temp'],
        }
        
        # Distribución de energía en bandas de frecuencia
        # Asume que frequencies van de ~13 Hz a ~99 Hz
        freq_array = record['frequencies']
        
        # Baja: < 30 Hz
        low_mask = freq_array < 30
        feature_dict['energy_low'] = np.sum(values[low_mask] ** 2) if np.any(low_mask) else 0
        
        # Media: 30-70 Hz
        mid_mask = (freq_array >= 30) & (freq_array < 70)
        feature_dict['energy_mid'] = np.sum(values[mid_mask] ** 2) if np.any(mid_mask) else 0
        
        # Alta: > 70 Hz
        high_mask = freq_array >= 70
        feature_dict['energy_high'] = np.sum(values[high_mask] ** 2) if np.any(high_mask) else 0
        
        # Ratios de energía
        total_energy = feature_dict['energy'] + 1e-8
        feature_dict['ratio_low'] = feature_dict['energy_low'] / total_energy
        feature_dict['ratio_mid'] = feature_dict['energy_mid'] / total_energy
        feature_dict['ratio_high'] = feature_dict['energy_high'] / total_energy
        
        # Frecuencia dominante
        if len(values) > 0:
            dominant_idx = np.argmax(values)
            feature_dict['dominant_frequency'] = freq_array[dominant_idx]
        else:
            feature_dict['dominant_frequency'] = 0
        
        features_list.append(feature_dict)
        valid_indices.append(idx)
    except Exception as e:
        pass

X_all = pd.DataFrame(features_list).fillna(0).replace([np.inf, -np.inf], 0)

print(f"[OK] Extrajeron {len(X_all)} características espectrales")
print(f"     Features: {list(X_all.columns)}")

# PASO 3: DETECTAR ANOMALÍAS
print("\n[PASO 3] Detectando anomalías con Isolation Forest...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Contamination adaptativa: espera encontrar entre 10-20% anomalías
contamination_rate = min(0.20, max(0.10, len(X_all) / 10000))
iso_forest = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
predictions = iso_forest.fit_predict(X_scaled)

anomaly_mask = predictions == -1
n_anomalies = np.sum(anomaly_mask)
print(f"[OK] Detectadas {n_anomalies} anomalías ({100*n_anomalies/len(X_all):.1f}%)")

X_anomalies = X_all[anomaly_mask].copy()
X_anomalies_scaled = X_scaled[anomaly_mask].copy()

# PASO 4: AGRUPAR ANOMALÍAS CON K-MEANS
print("\n[PASO 4] Agrupando anomalías con K-Means...")

n_types = 4
kmeans = KMeans(n_clusters=n_types, random_state=42, n_init=10)
anomaly_types = kmeans.fit_predict(X_anomalies_scaled)

# Cuenta muestras por tipo
unique, counts = np.unique(anomaly_types, return_counts=True)
for utype, count in zip(unique, counts):
    print(f"     Type {utype}: {count} muestras ({100*count/len(anomaly_types):.1f}%)")

# PASO 5: ENTRENAR CLASIFICADOR GRADIENT BOOSTING
print("\n[PASO 5] Entrenando Gradient Boosting...")

# Validación cruzada 5-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_anomalies_scaled, anomaly_types), 1):
    X_train = X_anomalies_scaled[train_idx]
    y_train = anomaly_types[train_idx]
    X_test = X_anomalies_scaled[test_idx]
    y_test = anomaly_types[test_idx]
    
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                     max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    fold_accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(fold_accuracy)
    print(f"     Fold {fold}: {fold_accuracy*100:.2f}%")

mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
print(f"\n[RESULTADO] Validación Cruzada: {mean_cv*100:.2f}% ± {std_cv*100:.2f}%")

# Entrenar modelo final con todos los datos
gb_final = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                       max_depth=5, random_state=42)
gb_final.fit(X_anomalies_scaled, anomaly_types)

# PASO 6: EVALUACIÓN FINAL
print("\n[PASO 6] Evaluación final...")

y_pred_final = gb_final.predict(X_anomalies_scaled)
final_accuracy = accuracy_score(anomaly_types, y_pred_final)
f1_final = f1_score(anomaly_types, y_pred_final, average='macro')

print(f"     Train Accuracy: {final_accuracy*100:.2f}%")
print(f"     F1-Score (macro): {f1_final*100:.2f}%")
print(f"\n     Confusion Matrix:")
cm = confusion_matrix(anomaly_types, y_pred_final)
print(cm)

print(f"\n     Classification Report:")
print(classification_report(anomaly_types, y_pred_final))

# Feature importance
feature_importance = gb_final.feature_importances_
top_features_idx = np.argsort(feature_importance)[-5:][::-1]
print(f"\n     Top 5 Features:")
for i, idx in enumerate(top_features_idx, 1):
    feat_name = X_all.columns[idx]
    importance = feature_importance[idx]
    print(f"       {i}. {feat_name}: {importance*100:.1f}%")

# PASO 7: GUARDAR MODELOS
print("\n[PASO 7] Guardando modelos entrenados...")

model_dir = 'C:/Users/adria/OneDrive/Escritorio/TFG/models'
import os
os.makedirs(model_dir, exist_ok=True)

# Guardar modelos
with open(f'{model_dir}/aeinnova_anomaly_detector_v2.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)

with open(f'{model_dir}/aeinnova_fault_type_classifier_v2.pkl', 'wb') as f:
    pickle.dump(gb_final, f)

with open(f'{model_dir}/aeinnova_scaler_anomaly_v2.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(f'{model_dir}/aeinnova_kmeans_v2.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Guardar feature names
feature_names = X_all.columns.tolist()
with open(f'{model_dir}/aeinnova_feature_names_v2.txt', 'w') as f:
    for name in feature_names:
        f.write(name + '\n')

print(f"[OK] Modelos guardados en {model_dir}/")

# RESUMEN FINAL
print("\n" + "=" * 80)
print("RESUMEN - ENTRENAMIENTO COMPLETADO")
print("=" * 80)
print(f"Registros procesados:        {len(X_all)} (formato 'readings')")
print(f"Anomalías detectadas:        {n_anomalies} ({100*n_anomalies/len(X_all):.1f}%)")
print(f"Tipos de fallo identificados: {n_types}")
print(f"Accuracy (5-fold CV):        {mean_cv*100:.2f}% ± {std_cv*100:.2f}%")
print(f"F1-Score (macro):            {f1_final*100:.2f}%")
print("=" * 80)
