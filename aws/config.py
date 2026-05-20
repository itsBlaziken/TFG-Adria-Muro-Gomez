import boto3

REGION = "eu-west-1"

# El bucket debe ser único globalmente — se genera con tu account ID automáticamente
def get_bucket_name():
    account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return f"aeinnova-tfg-{account_id}"

ROLE_NAME = "AEInnovaSageMakerRole"

# Prefijos S3
S3_RAW       = "raw"
S3_PROCESSED = "processed"
S3_MODELS    = "models"
S3_OUTPUTS   = "outputs"

# Ruta local al dataset
LOCAL_DATA = "../data/DatasetI.json"
