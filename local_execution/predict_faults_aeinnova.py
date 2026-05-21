import json
import os
import pickle
import numpy as np

# Rutes locals dels models entrenats i del dataset per a l'avaluació
MODEL_DIR = r'C:/Users/adria/OneDrive/Escritorio/TFG/models'
DATA_PATH = r'C:/Users/adria/OneDrive/Escritorio/TFG/data/DatasetI.json'

# Característiques espectrals en el mateix ordre que durant l'entrenament
FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]

# Descripció de cada tipus de fallo identificat pel sistema
FAULT_TYPES = {
    0: 'Type 0 - High Std Amplitude - holgura / impacto mecánico',
    1: 'Type 1 - High Energy Mid-Frequency - desalineación',
    2: 'Type 2 - High Energy Low-Frequency - desbalance mecánico',
    3: 'Type 3 - Balanced Multi-Band Energy - degradación general',
}


def load_models(model_dir):
    # Càrrega dels quatre models serialitzats i la llista de features
    models = {}
    for key, fname in [
        ('anomaly_detector', 'aeinnova_anomaly_detector.pkl'),
        ('classifier',       'aeinnova_fault_type_classifier.pkl'),
        ('scaler',           'aeinnova_scaler_anomaly.pkl'),
        ('kmeans',           'aeinnova_kmeans.pkl'),
    ]:
        with open(os.path.join(model_dir, fname), 'rb') as f:
            models[key] = pickle.load(f)

    with open(os.path.join(model_dir, 'aeinnova_feature_names.txt'), 'r') as f:
        models['feature_names'] = [line.strip() for line in f]

    print(f"Models carregats des de: {model_dir}/")
    return models


def predict_batch(X, models):
    # Normalització i detecció d'anomalies sobre tot el batch
    X_scaled      = models['scaler'].transform(X)
    anomaly_preds = models['anomaly_detector'].predict(X_scaled)
    is_anomaly    = anomaly_preds == -1

    fault_types       = np.full(len(X), -1, dtype=int)
    confidences       = np.zeros(len(X), dtype=float)
    probabilities_all = [None] * len(X)

    # Classificació del tipus de fallo només per als registres anòmals
    if np.any(is_anomaly):
        X_anom = X_scaled[is_anomaly]
        ft     = models['classifier'].predict(X_anom)
        probas = models['classifier'].predict_proba(X_anom)

        fault_types[is_anomaly] = ft
        confidences[is_anomaly] = np.max(probas, axis=1)
        for i, idx in enumerate(np.where(is_anomaly)[0]):
            probabilities_all[idx] = {k: float(p) for k, p in enumerate(probas[i])}

    # Construcció del diccionari de resultats per a cada registre
    results = []
    for i in range(len(X)):
        ft = int(fault_types[i])
        results.append({
            'is_anomaly':        bool(is_anomaly[i]),
            'fault_type':        ft,
            'fault_description': FAULT_TYPES.get(ft, 'Comportamiento normal'),
            'confidence':        round(float(confidences[i]), 4),
            'probabilities':     probabilities_all[i] if probabilities_all[i] else {},
        })

    return results


if __name__ == '__main__':
    import pandas as pd
    from train_fault_types import extract_features, load_dataset

    # Càrrega dels models i del dataset complet per a l'avaluació
    models  = load_models(MODEL_DIR)
    records = load_dataset(DATA_PATH)

    print(f"Total mostres carregades: {len(records)}")

    # Extracció de les característiques en el mateix ordre que l'entrenament
    rows = []
    for r in records:
        try:
            feat = extract_features(r)
            rows.append([feat[c] for c in FEATURE_COLS])
        except Exception:
            pass

    X       = np.array(rows)
    results = predict_batch(X, models)

    # Resum de les prediccions i distribució per tipus de fallo
    n_anom = sum(r['is_anomaly'] for r in results)
    print(f"Anomalies detectades: {n_anom} ({100 * n_anom / len(results):.1f}%)")

    counts = {k: 0 for k in FAULT_TYPES}
    for r in results:
        if r['is_anomaly']:
            counts[r['fault_type']] += 1

    print("Distribució de fallos:")
    for ft, label in FAULT_TYPES.items():
        print(f"  {label}: {counts[ft]}")
