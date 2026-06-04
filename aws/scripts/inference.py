# SageMaker Batch Transform
# SageMaker crida: model_fn → input_fn → predict_fn → output_fn

import io
import json
import os
import pickle
import numpy as np
import pandas as pd

# Característiques en el mateix ordre que durant l'entrenament.
# Mode i eix s'inclouen perquè el rang de freqüències mesurat varia amb el mode.
FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
    'mode', 'axis',
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


# Interfície SageMaker: les quatre funcions que el framework crida automàticament

def model_fn(model_dir):
    return load_models(model_dir)


def input_fn(request_body, content_type='text/csv'):
    # Parseja el cos de la petició com un CSV i valida les columnes necessàries
    if content_type != 'text/csv':
        raise ValueError(f"Content-type no suportat: {content_type}")
    df      = pd.read_csv(io.StringIO(request_body))
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnes faltants: {missing}")
    return df[FEATURE_COLS].fillna(0).values


def predict_fn(data, models):
    return predict_batch(data, models)


def output_fn(predictions, accept='application/json'):
    # Serialitza les prediccions com a JSON per retornar-les al client
    if accept != 'application/json':
        raise ValueError(f"Accept type no suportat: {accept}")
    return json.dumps(predictions, ensure_ascii=False), 'application/json'
