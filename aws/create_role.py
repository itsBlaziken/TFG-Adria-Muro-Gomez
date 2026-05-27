# ─────────────────────────────────────────────────────────────────────────────
# CREATE_ROLE.PY — Crea el rol IAM per a SageMaker
#
# QUÈ FA:
#   Crea el rol IAM que SageMaker necessita per accedir a S3 i executar jobs.
#   Si el rol ja existeix (per exemple, si ja has executat aquest script abans),
#   simplement recupera el seu identificador i no fa res més.
#
# PER QUÈ CAL UN ROL IAM?
#   AWS no deixa que SageMaker accedeixi a S3 per defecte. Cal crear un "rol"
#   (un conjunt de permisos) i assignar-lo a SageMaker. És com donar un carnet
#   d'identitat a SageMaker que li diu "pots llegir i escriure a S3".
#   D'aquesta manera no cal posar cap contrasenya al codi.
#
# QUAN EXECUTAR:
#   Una sola vegada per compte AWS. El rol queda creat permanentment.
#
# ÚS:
#   python create_role.py
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import json
from config import REGION, ROLE_NAME

# Client IAM — IAM és el servei d'AWS que gestiona usuaris, rols i permisos.
# A diferència d'altres serveis, IAM és global (no té regió).
iam = boto3.client("iam")

# ─────────────────────────────────────────────────────────────────────────────
# POLÍTICA DE CONFIANÇA (Trust Policy)
#
# Defineix QUI pot assumir aquest rol. En aquest cas, únicament el servei
# de SageMaker d'AWS. Sense aquesta política, ningú podria utilitzar el rol.
#
# "sts:AssumeRole" és l'acció que permet a un servei o usuari "prendre"
# temporalment els permisos d'un rol.
# ─────────────────────────────────────────────────────────────────────────────
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
    # ─────────────────────────────────────────────────────────────────────────
    # Pas 1: Crear el rol (o recuperar-lo si ja existeix)
    #
    # Un rol IAM és un conjunt de permisos que s'assigna a un servei d'AWS.
    # El creem amb la política de confiança definida a dalt, que indica
    # que únicament SageMaker pot assumir-lo.
    # ─────────────────────────────────────────────────────────────────────────
    try:
        response = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Rol d'execució per a SageMaker - TFG AEInnova"
        )
        role_arn = response["Role"]["Arn"]
        print(f"Rol creat: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        # Si el rol ja existia d'una execució anterior, simplement llegim el seu ARN.
        # ARN (Amazon Resource Name) és l'identificador únic de qualsevol recurs AWS.
        role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        print(f"Rol ja existia: {role_arn}")

    # ─────────────────────────────────────────────────────────────────────────
    # Pas 2: Adjuntar polítiques de permisos al rol
    #
    # Les polítiques defineixen QUÈ pot fer el rol. Adjuntem tres polítiques
    # gestionades per Amazon (ja existeixen, no cal crear-les):
    #
    #   AmazonSageMakerFullAccess  → permisos per crear i gestionar jobs
    #   AmazonS3FullAccess         → permisos per llegir i escriure al bucket
    #   CloudWatchLogsFullAccess   → permisos per escriure logs dels jobs
    #                                (CloudWatch és el servei de logs d'AWS)
    # ─────────────────────────────────────────────────────────────────────────
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
    print("Fet")
