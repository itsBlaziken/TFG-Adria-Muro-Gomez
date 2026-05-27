# ─────────────────────────────────────────────────────────────────────────────
# TRAIN_LOCAL_UPLOAD.PY — Entrena els models i puja els artefactes a S3
#
# QUÈ FA:
#   1. Descarrega el features.csv generat pel Processing Job de S3
#   2. Entrena els tres models del pipeline en local:
#        - Isolation Forest  → detecta anomalies
#        - K-Means           → agrupa les anomalies en tipologies de fallo
#        - Gradient Boosting → classifica cada anomalia en la seva tipologia
#   3. Empaqueta tots els models en un fitxer model.tar.gz
#   4. Puja el paquet a S3 perquè l'inferència el pugui descarregar
#
# PER QUÈ S'ENTRENA EN LOCAL I NO A SAGEMAKER?
#   Els comptes nous d'AWS tenen quota 0 per a Training Jobs de SageMaker
#   (límit imposat per Amazon per prevenir usos fraudulents en comptes nous).
#   El codi d'entrenament és idèntic al que s'executaria al núvol: simplement
#   s'executa a la màquina local i es puja el resultat a S3.
#
# QUÈ ÉS UN model.tar.gz?
#   És un fitxer comprimit (com un .zip) que conté tots els models serialitzats
#   (.pkl) i el script d'inferència. SageMaker espera aquest format per als
#   artefactes dels models.
#
# ÚS:
#   python train_local_upload.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os

# Afegeix el directori scripts/ al path de Python perquè pugui importar train.py
# (els scripts de SageMaker viuen en una carpeta separada)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import boto3
import tarfile
import tempfile
import pandas as pd
from io import BytesIO
from config import REGION, S3_PROCESSED, S3_MODELS, get_bucket_name
from train import train  # Funció d'entrenament definida a scripts/train.py

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()


def download_features():
    """Descarrega el CSV de característiques generat pel Processing Job.

    El features.csv conté una fila per registre amb les 15 característiques
    espectrals calculades (RMS, energia per bandes, factor de cresta, etc.)
    que s'utilitzen com a entrada dels models.
    """
    print("Descarregant features.csv des de S3...")
    # tempfile.mkdtemp() crea un directori temporal al sistema operatiu.
    # S'utilitza per evitar guardar fitxers en rutes absolutes codificades.
    tmp_dir = tempfile.mkdtemp()
    local   = os.path.join(tmp_dir, 'features.csv')
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", local)
    df = pd.read_csv(local)
    print(f"Features carregades: {len(df)} registres")
    return df


def package_and_upload(model_dir):
    """Empaqueta els models en tar.gz i el puja a S3.

    SageMaker espera que els artefactes dels models estiguin empaquetats
    en un fitxer .tar.gz (format estàndard de compressió Unix).
    El paquet inclou:
      - Els fitxers .pkl dels models (Isolation Forest, K-Means, Gradient Boosting, Scaler)
      - El fitxer de noms de features (.txt)
      - El script d'inferència (inference.py) per fer-lo compatible amb Batch Transform
    """
    print("Empaquetant models...")

    # BytesIO és un buffer en memòria. L'usem per crear el tar.gz sense
    # haver de desar un fitxer temporal al disc.
    tar_buffer = BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Afegir tots els fitxers del directori de models al paquet
        for fname in os.listdir(model_dir):
            tar.add(os.path.join(model_dir, fname), arcname=fname)

        # inference.py s'inclou dins del paquet perquè SageMaker Batch Transform
        # el necessita al mateix lloc que els models per poder fer prediccions
        inf_path = os.path.join(os.path.dirname(__file__), 'scripts', 'inference.py')
        tar.add(inf_path, arcname='inference.py')

    # Tornem al principi del buffer per poder llegir-lo des del principi
    tar_buffer.seek(0)

    s3_key    = f"{S3_MODELS}/model.tar.gz"
    s3.upload_fileobj(tar_buffer, bucket, s3_key)
    model_uri = f"s3://{bucket}/{s3_key}"
    print(f"Model pujat: {model_uri}")
    return model_uri


if __name__ == "__main__":
    # Pas 1: Descarregar les dades de S3
    df = download_features()

    # Pas 2: Crear un directori temporal per desar els models entrenats
    model_dir = tempfile.mkdtemp()

    # Pas 3: Entrenar els models (Isolation Forest + K-Means + Gradient Boosting)
    # La funció train() està definida a scripts/train.py i desa els .pkl a model_dir
    train(df, model_dir)

    # Pas 4: Empaquetar i pujar els models a S3
    model_uri = package_and_upload(model_dir)

    # Desa la URI del model en un fitxer local per referència futura
    with open(".last_model_uri", "w") as f:
        f.write(model_uri)

    print(f"\nModels a: {model_uri}")
    print("Fet")
