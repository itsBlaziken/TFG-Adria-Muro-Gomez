# Entrena els models en local i puja els artefactes a S3.
# Les comptes noves d'AWS tenen quota 0 per a Training Jobs, per tant s'entrena
# localment amb el mateix codi que s'executaria al contenidor de SageMaker.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import boto3
import tarfile
import tempfile
import pandas as pd
from io import BytesIO
from config import REGION, S3_PROCESSED, S3_MODELS, get_bucket_name
from train import train

s3     = boto3.client("s3", region_name=REGION)
bucket = get_bucket_name()


def download_features():
    # Descarrega el CSV de característiques generat pel Processing Job
    print("Descarregant features.csv des de S3...")
    tmp_dir = tempfile.mkdtemp()
    local   = os.path.join(tmp_dir, 'features.csv')
    s3.download_file(bucket, f"{S3_PROCESSED}/features.csv", local)
    df = pd.read_csv(local)
    print(f"Features carregades: {len(df)} registres")
    return df


def package_and_upload(model_dir):
    # Empaqueta tots els artefactes del model en un tar.gz i el puja a S3
    print("Empaquetant models...")
    tar_buffer = BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for fname in os.listdir(model_dir):
            tar.add(os.path.join(model_dir, fname), arcname=fname)

        # Inclou inference.py perquè el paquet sigui compatible amb Batch Transform
        inf_path = os.path.join(os.path.dirname(__file__), 'scripts', 'inference.py')
        tar.add(inf_path, arcname='inference.py')

    tar_buffer.seek(0)
    s3_key    = f"{S3_MODELS}/model.tar.gz"
    s3.upload_fileobj(tar_buffer, bucket, s3_key)
    model_uri = f"s3://{bucket}/{s3_key}"
    print(f"Model pujat: {model_uri}")
    return model_uri


if __name__ == "__main__":
    # Descarrega les dades, entrena i puja el model a S3
    df        = download_features()
    model_dir = tempfile.mkdtemp()

    train(df, model_dir)

    model_uri = package_and_upload(model_dir)

    with open(".last_model_uri", "w") as f:
        f.write(model_uri)

    print(f"\nModels a: {model_uri}")
    print("Ara executa: python inference_local_upload.py")
