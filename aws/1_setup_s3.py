# PASO 1: Crear bucket S3 y subir el dataset
# Ejecutar una sola vez (o cuando quieras subir datos nuevos).

import boto3
import os
from config import REGION, S3_RAW, LOCAL_DATA, get_bucket_name

s3 = boto3.client("s3", region_name=REGION)


def create_bucket(bucket_name):
    print(f"Creando bucket '{bucket_name}' en {REGION}...")
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        print(f"[OK] Bucket creado: s3://{bucket_name}/")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"[INFO] Bucket ya existe: s3://{bucket_name}/")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise

    # Bloquear acceso publico (buena practica)
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True
        }
    )
    print("[OK] Acceso publico bloqueado")


def upload_dataset(bucket_name):
    local_path = os.path.join(os.path.dirname(__file__), LOCAL_DATA)
    local_path = os.path.abspath(local_path)

    if not os.path.exists(local_path):
        print(f"[ERROR] No se encuentra el dataset en: {local_path}")
        return

    s3_key = f"{S3_RAW}/DatasetI.json"
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Subiendo dataset ({file_size_mb:.1f} MB) a s3://{bucket_name}/{s3_key} ...")

    s3.upload_file(
        local_path,
        bucket_name,
        s3_key,
        Callback=UploadProgress(local_path)
    )
    print(f"\n[OK] Dataset subido: s3://{bucket_name}/{s3_key}")


class UploadProgress:
    def __init__(self, filepath):
        self._size = os.path.getsize(filepath)
        self._seen = 0

    def __call__(self, bytes_amount):
        self._seen += bytes_amount
        pct = (self._seen / self._size) * 100
        print(f"\r  Progreso: {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    bucket_name = get_bucket_name()
    create_bucket(bucket_name)
    upload_dataset(bucket_name)
    print("\nListo. Ahora ejecuta: python 2_run_processing.py")
