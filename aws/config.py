import boto3

# Regió AWS on es despleguen tots els recursos del projecte
REGION = "eu-west-1"

# El nom del bucket incorpora l'account ID per garantir la unicitat global
def get_bucket_name():
    account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return f"aeinnova-tfg-{account_id}"

# Nom del rol IAM que SageMaker utilitza per accedir a S3 i executar jobs
ROLE_NAME = "AEInnovaSageMakerRole"

# Prefixos S3 que separen les diferents fases del pipeline
S3_RAW       = "raw"
S3_PROCESSED = "processed"
S3_MODELS    = "models"
S3_OUTPUTS   = "outputs"

# Ruta local al dataset (relativa al directori aws/)
LOCAL_DATA = "../data/DatasetI.json"
