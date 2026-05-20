# PASO 2: SageMaker Processing Job (preprocesamiento)
# Sube el script a S3, lanza el job y espera resultado.

import boto3
import time
import json
from config import REGION, ROLE_NAME, S3_RAW, S3_PROCESSED, get_bucket_name

sm  = boto3.client("sagemaker", region_name=REGION)
s3  = boto3.client("s3",        region_name=REGION)
iam = boto3.client("iam")

role_arn   = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
bucket     = get_bucket_name()
job_name   = f"aeinnova-preprocess-{int(time.time())}"
IMAGE_URI  = f"141502667606.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def upload_script(local_path, s3_key):
    s3.upload_file(local_path, bucket, s3_key)
    print(f"[OK] Script subido: s3://{bucket}/{s3_key}")


def wait_for_job(job_name):
    print(f"\nEsperando job '{job_name}'...")
    while True:
        resp   = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")
        if status == "Completed":
            return resp
        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "Sin detalle")
            raise RuntimeError(f"Job fallido ({status}): {reason}")
        time.sleep(30)


# 1. Subir script de preprocesamiento a S3
upload_script("scripts/preprocess.py", "code/preprocess.py")

# 2. Crear el Processing Job
print(f"\nLanzando Processing Job: {job_name}")
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
    ProcessingInputs=[
        {
            "InputName": "raw",
            "S3Input": {
                "S3Uri":        f"s3://{bucket}/{S3_RAW}/",
                "LocalPath":    "/opt/ml/processing/input/raw",
                "S3DataType":   "S3Prefix",
                "S3InputMode":  "File",
            },
        },
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri":        f"s3://{bucket}/code/preprocess.py",
                "LocalPath":    "/opt/ml/processing/input/code",
                "S3DataType":   "S3Prefix",
                "S3InputMode":  "File",
            },
        },
    ],
    ProcessingOutputConfig={
        "Outputs": [
            {
                "OutputName": "processed",
                "S3Output": {
                    "S3Uri":          f"s3://{bucket}/{S3_PROCESSED}/",
                    "LocalPath":      "/opt/ml/processing/output",
                    "S3UploadMode":   "EndOfJob",
                },
            }
        ]
    },
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount":    1,
            "InstanceType":     "ml.t3.medium",
            "VolumeSizeInGB":   30,
        }
    },
)

# 3. Esperar resultado
wait_for_job(job_name)

print(f"\n[OK] Processing Job completado.")
print(f"Features en: s3://{bucket}/{S3_PROCESSED}/features.csv")

with open(".last_processing_job", "w") as f:
    f.write(job_name)

print("Ahora ejecuta: python 3_run_training.py")
