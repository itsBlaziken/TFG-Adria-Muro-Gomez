# Crea el rol IAM que SageMaker necessita per accedir a S3 i executar jobs.
# Executar només una vegada per compte AWS.

import boto3
import json
from config import REGION, ROLE_NAME

iam = boto3.client("iam")

# Política de confiança que permet a SageMaker assumir aquest rol
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
    print(f"Creant IAM Role '{ROLE_NAME}'...")

    # Crea el rol o recupera l'ARN si ja existia
    try:
        response = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Rol d'execució per a SageMaker - TFG AEInnova"
        )
        role_arn = response["Role"]["Arn"]
        print(f"Rol creat: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"Rol ja existia: {role_arn}")

    # Adjunta les polítiques necessàries per a SageMaker, S3 i CloudWatch
    policies = [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
        "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
    ]

    for policy_arn in policies:
        try:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
            print(f"Política adjuntada: {policy_arn.split('/')[-1]}")
        except iam.exceptions.PolicyNotAttachableException as e:
            print(f"Avís: {e}")

    print(f"\nRole ARN: {role_arn}")
    return role_arn


if __name__ == "__main__":
    role_arn = create_role()
    print("Fet. Ara executa: python 1_setup_s3.py")
