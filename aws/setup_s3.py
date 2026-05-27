# ─────────────────────────────────────────────────────────────────────────────
# SETUP_S3.PY — Crea el bucket S3 i puja el dataset original
#
# QUÈ FA:
#   1. Crea el bucket S3 del projecte a la regió configurada
#   2. Bloqueja l'accés públic al bucket per seguretat
#   3. Puja el dataset original (DatasetI.json) al prefix "raw/"
#
# QUÈ ÉS UN BUCKET S3?
#   Amazon S3 (Simple Storage Service) és el servei d'emmagatzematge d'AWS.
#   Un bucket és com una carpeta principal al núvol. Dins s'organitzen els
#   fitxers en "prefixos" (carpetes virtuals). El nom del bucket ha de ser
#   únic a tot AWS perquè és part d'una URL pública.
#
# QUAN EXECUTAR:
#   Una sola vegada per inicialitzar la infraestructura. Si el bucket ja
#   existeix, el script passa directament a pujar el dataset.
#
# ÚS:
#   python setup_s3.py
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import os
from config import REGION, S3_RAW, LOCAL_DATA, get_bucket_name

# Client S3 — és la interfície Python per interactuar amb Amazon S3
s3 = boto3.client("s3", region_name=REGION)


def create_bucket(bucket_name):
    """Crea el bucket S3 i bloqueja l'accés públic."""

    print(f"Creant bucket '{bucket_name}' a {REGION}...")
    try:
        # LocationConstraint indica la regió. Us i est d'EUA no necessiten aquest paràmetre,
        # però totes les altres regions sí. Sense ell, el bucket es crearia a us-east-1.
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        print(f"Bucket creat: s3://{bucket_name}/")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        # Si el bucket ja existeix al nostre compte, simplement continuem
        print(f"Bucket ja existia: s3://{bucket_name}/")
    except Exception as e:
        print(f"Error: {e}")
        raise

    # ─────────────────────────────────────────────────────────────────────────
    # Bloquejar l'accés públic
    #
    # Per defecte, els fitxers d'un bucket S3 no són públics, però és
    # bona pràctica bloquejar explícitament l'accés públic per evitar
    # exposicions accidentals de dades sensibles.
    # ─────────────────────────────────────────────────────────────────────────
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls":       True,   # Ignora les ACL públiques noves
            "IgnorePublicAcls":      True,   # Ignora les ACL públiques existents
            "BlockPublicPolicy":     True,   # Bloqueja polítiques de bucket públiques
            "RestrictPublicBuckets": True,   # Restringeix l'accés públic al bucket
        }
    )
    print("Accés públic bloquejat")


def upload_dataset(bucket_name):
    """Puja el dataset local al prefix raw/ de S3."""

    # Construeix la ruta absoluta al dataset (el fitxer és un nivell amunt de /aws)
    local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DATA))

    if not os.path.exists(local_path):
        print(f"Error: no es troba el dataset a: {local_path}")
        return

    # La "clau" S3 és el nom complet del fitxer dins del bucket, incloent el prefix.
    # Ex: "raw/DatasetI.json" → s3://aeinnova-tfg-xxxx/raw/DatasetI.json
    s3_key = f"{S3_RAW}/DatasetI.json"
    print(f"Pujant dataset a s3://{bucket_name}/{s3_key} ...")

    # El paràmetre Callback permet mostrar el progrés de la pujada en temps real
    s3.upload_file(local_path, bucket_name, s3_key, Callback=UploadProgress(local_path))


class UploadProgress:
    """Callback que s'invoca periòdicament durant la pujada per mostrar el progrés."""

    def __init__(self, filepath):
        # Llegim la mida total del fitxer per calcular el percentatge
        self._size = os.path.getsize(filepath)
        self._seen = 0  # Bytes ja pujats

    def __call__(self, bytes_amount):
        # boto3 crida aquest mètode cada vegada que puja un fragment del fitxer
        self._seen += bytes_amount
        pct = (self._seen / self._size) * 100
        # \r torna el cursor al principi de la línia per sobreescriure el progrés anterior
        print(f"\r  Progrés: {pct:.1f}%", end="", flush=True)


if __name__ == "__main__":
    bucket_name = get_bucket_name()
    create_bucket(bucket_name)
    upload_dataset(bucket_name)
    print("\nFet")
