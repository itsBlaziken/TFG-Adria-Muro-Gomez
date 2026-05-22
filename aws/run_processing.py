# Llança el SageMaker Processing Job que transforma el dataset JSON en features.csv.
# El job s'executa en un contenidor sklearn gestionat per AWS.

import boto3
import time
from config import REGION, ROLE_NAME, S3_RAW, S3_PROCESSED, get_bucket_name

sm  = boto3.client("sagemaker", region_name=REGION)
s3  = boto3.client("s3",        region_name=REGION)
iam = boto3.client("iam")

# Paràmetres del job
role_arn  = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
bucket    = get_bucket_name()
job_name  = f"aeinnova-preprocess-{int(time.time())}"
IMAGE_URI = f"141502667606.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def upload_script(local_path, s3_key):
    # Puja el script de preprocessament a S3 perquè el contenidor el pugui llegir
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Script pujat: s3://{bucket}/{s3_key}")


def wait_for_job(job_name):
    # Espera que el job acabi comprovant l'estat cada 30 segons
    print(f"\nEsperant el job '{job_name}'...")
    while True:
        resp   = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        print(f"  [{time.strftime('%H:%M:%S')}] Estat: {status}")
        if status == "Completed":
            return resp
        if status in ("Failed", "Stopped"):
            raise RuntimeError(f"Job fallat ({status}): {resp.get('FailureReason', '')}")
        time.sleep(30)


# Puja el script i llança el Processing Job a SageMaker
upload_script("scripts/preprocess.py", "code/preprocess.py")

print(f"\nLlançant Processing Job: {job_name}")
print(f"  Input:  s3://{bucket}/{S3_RAW}/")
print(f"  Output: s3://{bucket}/{S3_PROCESSED}/")

sm.create_processing_job(
    ProcessingJobName=job_name,
    RoleArn=role_arn,
    AppSpecification={
        "ImageUri": IMAGE_URI,
        "ContainerEntrypoint": [
            "python3", "/opt/ml/processing/input/code/preprocess.py",
            "--input-data",  "/opt/ml/processing/input/raw",
            "--output-data", "/opt/ml/processing/output",
        ],
    },
    # Canals d'entrada: el dataset raw i el script de preprocessament
    ProcessingInputs=[
        {
            "InputName": "raw",
            "S3Input": {
                "S3Uri":       f"s3://{bucket}/{S3_RAW}/",
                "LocalPath":   "/opt/ml/processing/input/raw",
                "S3DataType":  "S3Prefix",
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri":       f"s3://{bucket}/code/preprocess.py",
                "LocalPath":   "/opt/ml/processing/input/code",
                "S3DataType":  "S3Prefix",
                "S3InputMode": "File",
            },
        },
    ],
    # Canal de sortida: el CSV de característiques
    ProcessingOutputConfig={
        "Outputs": [
            {
                "OutputName": "processed",
                "S3Output": {
                    "S3Uri":        f"s3://{bucket}/{S3_PROCESSED}/",
                    "LocalPath":    "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    },
    # Instància ml.t3.medium: compatible amb les quotes d'un compte nou
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount":  1,
            "InstanceType":   "ml.t3.medium",
            "VolumeSizeInGB": 30,
        }
    },
)

wait_for_job(job_name)

print(f"\nProcessing Job completat.")
print(f"Features a: s3://{bucket}/{S3_PROCESSED}/features.csv")

# Desa el nom del job per referència futura
with open(".last_processing_job", "w") as f:
    f.write(job_name)

print("Ara executa: python train_local_upload.py")
