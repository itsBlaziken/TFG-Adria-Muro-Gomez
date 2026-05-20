# Script que corre DENTRO del SageMaker Processing Job (contenedor sklearn)
# Input:  /opt/ml/processing/input/raw/DatasetI.json
# Output: /opt/ml/processing/output/features.csv

import json
import argparse
import os
import numpy as np
import pandas as pd


def extract_features(readings_data, temp=0.0):
    if 'L' not in readings_data or len(readings_data['L']) == 0:
        return None

    frequencies, values = [], []
    for entry in readings_data['L']:
        if 'M' in entry:
            m = entry['M']
            if 'frequency' in m and 'value' in m:
                try:
                    frequencies.append(float(m['frequency']['N']))
                    values.append(float(m['value']['N']))
                except (KeyError, ValueError):
                    pass

    if len(values) == 0:
        return None

    v = np.array(values)
    f = np.array(frequencies)
    rms = np.sqrt(np.mean(v ** 2))
    energy = np.sum(v ** 2) + 1e-8

    low_mask  = f < 30
    mid_mask  = (f >= 30) & (f < 70)
    high_mask = f >= 70

    e_low  = float(np.sum(v[low_mask]  ** 2)) if np.any(low_mask)  else 0.0
    e_mid  = float(np.sum(v[mid_mask]  ** 2)) if np.any(mid_mask)  else 0.0
    e_high = float(np.sum(v[high_mask] ** 2)) if np.any(high_mask) else 0.0

    return {
        'rms':                float(rms),
        'energy':             float(energy),
        'max_amplitude':      float(np.max(v)),
        'mean_amplitude':     float(np.mean(v)),
        'std_amplitude':      float(np.std(v)),
        'peak_to_peak':       float(np.max(v) - np.min(v)),
        'crest_factor':       float(np.max(v) / (rms + 1e-8)),
        'temp':               float(temp),
        'energy_low':         e_low,
        'energy_mid':         e_mid,
        'energy_high':        e_high,
        'ratio_low':          e_low  / energy,
        'ratio_mid':          e_mid  / energy,
        'ratio_high':         e_high / energy,
        'dominant_frequency': float(f[np.argmax(v)]),
    }


def main(input_dir, output_dir):
    input_file = os.path.join(input_dir, 'DatasetI.json')
    print(f"Leyendo: {input_file}")

    records = []
    skipped = 0

    with open(input_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                item = obj.get('Item', {})
                if 'readings' not in item:
                    skipped += 1
                    continue

                temp = float(item.get('T1', {}).get('N', 0))
                feats = extract_features(item['readings'], temp)
                if feats is None:
                    skipped += 1
                    continue

                feats['dev_eui']   = item.get('dev_eui',   {}).get('S', 'unknown')
                feats['mode']      = float(item.get('mode',      {}).get('N', -1))
                feats['axis']      = float(item.get('axis',      {}).get('N', -1))
                feats['timestamp'] = float(item.get('timestamp', {}).get('N',  0))
                records.append(feats)
            except Exception:
                skipped += 1

    print(f"Registros validos: {len(records)} | Descartados: {skipped}")

    df = pd.DataFrame(records).fillna(0).replace([np.inf, -np.inf], 0)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'features.csv')
    df.to_csv(out_file, index=False)
    print(f"CSV guardado: {out_file}  ({len(df)} filas, {len(df.columns)} columnas)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data',  type=str, default='/opt/ml/processing/input/raw')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()
    main(args.input_data, args.output_data)
