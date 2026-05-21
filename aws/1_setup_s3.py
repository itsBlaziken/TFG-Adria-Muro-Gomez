# Crea el bucket S3 i puja el dataset original.
# Executar una sola vegada (o quan es vulgui pujar dades noves).

import boto3
import os
from config import REGION, S3_RAW, LOCAL_DATA, get_bucket_name

s3 = boto3.client("s3", region_name=REGION)


def create_bucket(bucket_name):
    # Crea el bucket a la regió configurada i bloqueja l'accés públic
    print(f"Creant bucket '{bucket_name}' a {REGION}...")
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        print(f"Bucket creat: s3://{bucket_name}/")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket ja existia: s3://{bucket_name}/")
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Bloqueja l'accés públic com a mesura de seguretat
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls":      True,
            "IgnorePublicAcls":     True,
            "BlockPublicPolicy":    True,
            "RestrictPublicBuckets": True,
        }
    )
    print("Accés públic bloquejat")


def upload_dataset(bucket_name):
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DATA))

    if not os.path.exists(local_path):
        print(f"Error: no es troba el dataset a: {local_path}")
        return

    # Puja el dataset al prefix raw/ mostrant el progrés
    s3_key        = f"{S3_RAW}/DatasetI.json"
    file_size_mb  = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Pujant dataset ({file_size_mb:.1f} MB) → s3://{bucket_name}/{s3_key} ...")

    s3.upload_file(local_path, bucket_name, s3_key, Callback=UploadProgress(local_path))
    print(f"\nDataset pujat: s3://{bucket_name}/{s3_key}")


class UploadProgress:
    # Callback que mostra el percentatge de pujada en temps real
    def __init__(self, filepath):
        self._size = os.path.getsize(filepath)
        self._seen = 0

    def __call__(self, bytes_amount):
        self._seen += bytes_amount
        pct = (self._seen / self._size) * 100
        print(f"\r  Progrés: {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    bucket_name = get_bucket_name()
    create_bucket(bucket_name)
    upload_dataset(bucket_name)
    print("\nFet. Ara executa: python 2_run_processing.py")
