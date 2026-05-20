# Script de inferencia para SageMaker Batch Transform
# SageMaker carga este fichero y llama a las funciones model_fn / predict_fn /
# input_fn / output_fn segun el protocolo de serving de sklearn.

import os
import pickle
import io
import json
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'rms', 'energy', 'max_amplitude', 'mean_amplitude', 'std_amplitude',
    'peak_to_peak', 'crest_factor', 'temp',
    'energy_low', 'energy_mid', 'energy_high',
    'ratio_low', 'ratio_mid', 'ratio_high', 'dominant_frequency',
]

FAULT_LABELS = {
    0: 'High Amplitude Variability - holgura/impacto mecanico',
    1: 'Mid-Frequency Energy - desalineacion',
    2: 'Low-Frequency Energy - desbalance mecanico',
    3: 'Distributed Energy - degradacion general',
}


def model_fn(model_dir):
    models = {}
    for name in ('isolation_forest', 'kmeans', 'gradient_boosting', 'scaler'):
        path = os.path.join(model_dir, f'{name}.pkl')
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    print(f"[OK] Modelos cargados desde {model_dir}")
    return models


def input_fn(request_body, content_type='text/csv'):
    if content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(request_body))
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas faltantes en el CSV: {missing}")
        return df[FEATURE_COLS].fillna(0).values
    raise ValueError(f"Content-type no soportado: {content_type}. Usa text/csv.")


def predict_fn(data, models):
    X_scaled = models['scaler'].transform(data)

    anomaly_preds = models['isolation_forest'].predict(X_scaled)
    is_anomaly    = anomaly_preds == -1

    fault_types = np.full(len(data), -1, dtype=int)
    severity    = np.zeros(len(data), dtype=float)

    if np.any(is_anomaly):
        X_anom = X_scaled[is_anomaly]
        scores = models['isolation_forest'].score_samples(X_anom)
        ft     = models['gradient_boosting'].predict(X_anom)
        fault_types[is_anomaly] = ft
        severity[is_anomaly]    = np.abs(scores)

    results = []
    for i in range(len(data)):
        row = {
            'is_anomaly': bool(is_anomaly[i]),
            'fault_type': int(fault_types[i]),
            'fault_label': FAULT_LABELS.get(int(fault_types[i]), 'Normal'),
            'severity':   round(float(severity[i]), 4),
        }
        results.append(row)

    return results


def output_fn(predictions, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(predictions, ensure_ascii=False), 'application/json'
    raise ValueError(f"Accept type no soportado: {accept}. Usa application/json.")
