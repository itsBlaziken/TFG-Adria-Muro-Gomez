# PASO 0: Crear el IAM Role para SageMaker (ejecutar solo una vez)
# Este script crea el rol que SageMaker necesita para acceder a S3 y ejecutar jobs.

import boto3
import json
import sys
from config import REGION, ROLE_NAME

iam = boto3.client("iam")

trust_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}

def create_role():
    print(f"Creando IAM Role '{ROLE_NAME}'...")

    try:
        response = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Rol de ejecucion para SageMaker - TFG AEInnova"
        )
        role_arn = response["Role"]["Arn"]
        print(f"[OK] Role creado: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"[INFO] Role ya existia: {role_arn}")

    # Adjuntar politicas necesarias
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ]

    for policy_arn in policies:
        try:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
            print(f"[OK] Politica adjuntada: {policy_arn.split('/')[-1]}")
        except iam.exceptions.PolicyNotAttachableException as e:
            print(f"[WARN] {e}")

    print(f"\nROLE ARN (guardalo, lo necesitaras):\n{role_arn}\n")
    return role_arn


if __name__ == "__main__":
    role_arn = create_role()
    print("Listo. Ahora ejecuta: python 1_setup_s3.py")
