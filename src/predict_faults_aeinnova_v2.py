# Sistema de predicción de fallos AEInnova - VERSION 2
# Usa el formato "readings" (espectros de frecuencia)

import numpy as np
import pickle
import json

class AEInnovaFaultClassifier:
    """Clasificador de fallos basado en espectros de frecuencia"""
    
    def __init__(self, model_dir='models'):
        """Carga los modelos entrenados"""
        try:
            with open(f'{model_dir}/aeinnova_anomaly_detector_v2.pkl', 'rb') as f:
                self.anomaly_detector = pickle.load(f)
            
            with open(f'{model_dir}/aeinnova_fault_type_classifier_v2.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            
            with open(f'{model_dir}/aeinnova_scaler_anomaly_v2.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f'{model_dir}/aeinnova_feature_names_v2.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            self.model_dir = model_dir
            print(f"✓ Modelos cargados exitosamente desde {model_dir}/")
        except FileNotFoundError as e:
            print(f"✗ Error: No se encontraron modelos. Asegúrate de entrenar primero.")
            print(f"  Ejecuta: python train_fault_types_v2.py")
            raise
    
    def extract_features_from_readings(self, frequencies, values, temperature=0):
        """
        Extrae características espectrales de un espectro de frecuencia
        
        Args:
            frequencies: array de frecuencias
            values: array de amplitudes/energías
            temperature: temperatura del sensor
        
        Returns:
            dict con características espectrales
        """
        values = np.array(values, dtype=float)
        frequencies = np.array(frequencies, dtype=float)
        
        features = {
            'rms': np.sqrt(np.mean(values ** 2)),
            'energy': np.sum(values ** 2),
            'max_amplitude': np.max(values),
            'mean_amplitude': np.mean(values),
            'std_amplitude': np.std(values),
            'peak_to_peak': np.max(values) - np.min(values),
            'crest_factor': np.max(values) / (np.sqrt(np.mean(values ** 2)) + 1e-8),
            'temp': temperature,
        }
        
        # Energía en bandas de frecuencia
        low_mask = frequencies < 30
        mid_mask = (frequencies >= 30) & (frequencies < 70)
        high_mask = frequencies >= 70
        
        features['energy_low'] = np.sum(values[low_mask] ** 2) if np.any(low_mask) else 0
        features['energy_mid'] = np.sum(values[mid_mask] ** 2) if np.any(mid_mask) else 0
        features['energy_high'] = np.sum(values[high_mask] ** 2) if np.any(high_mask) else 0
        
        total_energy = features['energy'] + 1e-8
        features['ratio_low'] = features['energy_low'] / total_energy
        features['ratio_mid'] = features['energy_mid'] / total_energy
        features['ratio_high'] = features['energy_high'] / total_energy
        
        if len(values) > 0:
            dominant_idx = np.argmax(values)
            features['dominant_frequency'] = frequencies[dominant_idx]
        else:
            features['dominant_frequency'] = 0
        
        return features
    
    def predict(self, frequencies, values, temperature=0):
        """
        Predice el tipo de fallo a partir de un espectro de frecuencia
        
        Args:
            frequencies: array de frecuencias
            values: array de amplitudes
            temperature: temperatura (opcional)
        
        Returns:
            dict con predicción y confianza
        """
        # Extrae características
        features = self.extract_features_from_readings(frequencies, values, temperature)
        
        # Ordena características en el mismo orden que el entrenamiento
        X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Normaliza
        X_scaled = self.scaler.transform(X)
        
        # Detecta si es anomalía
        anomaly_pred = self.anomaly_detector.predict(X_scaled)[0]
        is_anomaly = (anomaly_pred == -1)
        
        if not is_anomaly:
            return {
                'is_anomaly': False,
                'fault_type': None,
                'confidence': 0.0,
                'details': 'Comportamiento normal - sin anomalía detectada'
            }
        
        # Clasifica tipo de fallo
        fault_type = self.classifier.predict(X_scaled)[0]
        probabilities = self.classifier.predict_proba(X_scaled)[0]
        confidence = np.max(probabilities)
        
        # Interpretación según tipo
        fault_types = {
            0: 'Type 0 - High Std Amplitude (Vibración de alta variabilidad) - Posible fallo: holgura / impacto mecánico',
            1: 'Type 1 - High Energy Mid-Frequency (Energía en frecuencia media) - Posible fallo: desalineación',
            2: 'Type 2 - High Energy Low-Frequency (Energía en baja frecuencia) - Posible fallo: desbalance mecánico',
            3: 'Type 3 - Balanced Multi-Band Energy (Energía distribuida) - Posible fallo: degradación general o mezcla de fallos'
        }
        
        return {
            'is_anomaly': True,
            'fault_type': fault_type,
            'fault_description': fault_types[fault_type],
            'confidence': float(confidence),
            'probabilities': {i: float(p) for i, p in enumerate(probabilities)},
            'features': features
        }


if __name__ == '__main__':
    print("=" * 80)
    print("EVALUACIÓN REAL SOBRE DATASET - AEINNOVA")
    print("=" * 80)

    clf = AEInnovaFaultClassifier('models')

    # === CARGA DATASET ===
    json_path = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'

    frequencies_list = []
    values_list = []
    temp_list = []

    print("\n[CARGA] Leyendo dataset...")

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                item = obj.get('Item', None)
                if not item:
                    continue

                if 'readings' not in item:
                    continue

                readings = item['readings']
                if 'L' not in readings:
                    continue

                freqs = []
                vals = []

                for r in readings['L']:
                    if 'M' in r:
                        m = r['M']
                        try:
                            freqs.append(float(m['frequency']['N']))
                            vals.append(float(m['value']['N']))
                        except:
                            pass

                if len(vals) == 0:
                    continue

                frequencies_list.append(np.array(freqs))
                values_list.append(np.array(vals))

                temp = float(item.get('T1', {}).get('N', 0))
                temp_list.append(temp)

            except:
                continue

    print(f"[OK] Registros cargados: {len(values_list)}")

    # === CLASIFICACIÓN ===
    print("\n[ANÁLISIS] Procesando dataset completo...\n")

    results = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        "normal": 0
    }

    anomaly_count = 0

    for i in range(len(values_list)):
        freq = frequencies_list[i]
        vals = values_list[i]
        temp = temp_list[i]

        res = clf.predict(freq, vals, temp)

        if not res['is_anomaly']:
            results["normal"] += 1
            continue

        anomaly_count += 1
        ft = res['fault_type']
        results[ft] += 1

    # === RESULTADOS FINALES ===
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES - CLASIFICACIÓN DE FALLOS")
    print("=" * 80)

    print(f"Total muestras: {len(values_list)}")
    print(f"Anomalías detectadas: {anomaly_count}")
    print(f"Normales: {results['normal']}\n")

    fault_names = {
        0: "Type 0 - Alta variabilidad (holgura/impacto)",
        1: "Type 1 - Energía media (desalineación)",
        2: "Type 2 - Baja frecuencia (desbalance)",
        3: "Type 3 - Mixto (degradación general)"
    }

    print("[DISTRIBUCIÓN DE FALLOS]")
    for k in [0, 1, 2, 3]:
        print(f"{fault_names[k]} → {results[k]} muestras")


