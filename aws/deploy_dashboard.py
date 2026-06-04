# ─────────────────────────────────────────────────────────────────────────────
# DEPLOY_DASHBOARD.PY — Desplega el dashboard Streamlit a Amazon EC2
#
# QUÈ FA:
#   Automatitza tot el procés de desplegament d'una aplicació web a AWS.
#   El resultat és un dashboard accessible des de qualsevol navegador a través
#   d'una URL pública, sense cap configuració manual al servidor.
#
# QUÈ ÉS AMAZON EC2?
#   EC2 (Elastic Compute Cloud) és el servei de màquines virtuals d'AWS.
#   Una instància EC2 és un ordinador virtual que funciona al núvol d'Amazon.
#   S'utilitza per allotjar serveis web que han d'estar sempre accessible.
#   La instència t3.micro és la més petita (i econòmica): 2 vCPU, 1 GB RAM.
#
# COM FUNCIONA EL DESPLEGAMENT?
#   1. Es puja el codi del dashboard a S3
#   2. Es crea un rol IAM perquè EC2 pugui llegir S3 sense credencials al codi
#   3. Es crea un Security Group (tallafocs) que obre el port 8501
#   4. Es llança la instència amb un "user data script" (script d'arrencada)
#      que instal·la les dependències i arrenca Streamlit com a servei
#   5. La instència s'executa contínuament i el dashboard queda accessible
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import json
import time
import sys
import os

# Directori on es troba aquest script (per construir rutes relatives)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGION   = "eu-west-1"
BUCKET   = "aeinnova-tfg-836321169819"  # Nom del bucket del projecte
APP_NAME = "aeinnova-dashboard"
PORT     = 8501  # Port per defecte de Streamlit


def step(msg):
    """Imprimeix una capçalera de pas per facilitar el seguiment del procés."""
    print(f"\n{'─'*60}\n{msg}\n{'─'*60}")


# ── 1. Verificar compte AWS ────────────────────────────────────────────────────
# STS (Security Token Service) retorna informació sobre el compte AWS actiu.
# Serveix per verificar que les credencials AWS estan configurades correctament.
step("1 · Obtenint compte AWS")
sts        = boto3.client("sts", region_name=REGION)
account_id = sts.get_caller_identity()["Account"]
print(f"  Account: {account_id}")


# ── 2. Pujar fitxers del dashboard a S3 ───────────────────────────────────────
# El codi del dashboard i les seves dependències es pugen a S3 perquè
# la instència EC2 els pugui descarregar durant l'arrencada.
# D'aquesta manera no cal copiar fitxers manualment al servidor.
step("2 · Pujant dashboard.py i requirements a S3")
s3 = boto3.client("s3", region_name=REGION)
s3.upload_file(os.path.join(SCRIPT_DIR, "dashboard.py"),               BUCKET, "dashboard/dashboard.py")
s3.upload_file(os.path.join(SCRIPT_DIR, "requirements_dashboard.txt"), BUCKET, "dashboard/requirements_dashboard.txt")
print(f"  Pujat a s3://{BUCKET}/dashboard/")


# ── 3. Crear rol IAM per a la instència EC2 ────────────────────────────────────
# La instència EC2 necessita un rol IAM per poder llegir fitxers de S3
# sense tenir credencials hardcodejades al codi. El rol actua com un
# carnet d'identitat que EC2 presenta a S3 per autenticar-se.
#
# Un "Instance Profile" és el contenidor que associa un rol IAM a una instència EC2.
# Tots els rols que s'assignen a instències EC2 necessiten un Instance Profile.
step("3 · Creant rol IAM per a la instència EC2")
iam          = boto3.client("iam")
role_name    = f"{APP_NAME}-ec2-role"
profile_name = f"{APP_NAME}-ec2-profile"

# Política de confiança: permet al servei EC2 assumir aquest rol
trust = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "ec2.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]
})

# Política de permisos: únicament permet llegir del bucket del projecte.
# S'utilitza el mínim de permisos necessaris (principi de least privilege).
s3_policy = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Action": ["s3:GetObject", "s3:ListBucket"],
                   "Resource": [f"arn:aws:s3:::{BUCKET}",
                                f"arn:aws:s3:::{BUCKET}/*"]}]
})

try:
    iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
    iam.put_role_policy(RoleName=role_name, PolicyName="S3Access", PolicyDocument=s3_policy)
    print(f"  Rol creat: {role_name}")
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  Rol ja existeix: {role_name}")

try:
    iam.create_instance_profile(InstanceProfileName=profile_name)
    iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
    print(f"  Instance profile creat: {profile_name}")
    # IAM necessita uns segons per propagar els canvis a tots els serveis d'AWS
    print("  Esperant propagació IAM (15 s)...")
    time.sleep(15)
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  Instance profile ja existeix: {profile_name}")


# ── 4. Crear Security Group (tallafocs virtual) ────────────────────────────────
# Un Security Group controla el tràfic de xarxa entrant i sortint d'una
# instència EC2. Hem d'obrir el port 8501 perquè els navegadors puguin
# accedir al dashboard Streamlit des d'internet.
#
# VPC (Virtual Private Cloud) és la xarxa virtual d'AWS. Cada compte té
# una VPC per defecte on es creen les instències.
step("4 · Creant Security Group (port 8501)")
ec2 = boto3.client("ec2", region_name=REGION)

# Obtenim l'identificador de la VPC per defecte del compte
vpc_id = ec2.describe_vpcs(
    Filters=[{"Name": "isDefault", "Values": ["true"]}]
)["Vpcs"][0]["VpcId"]

sg_name = f"{APP_NAME}-sg"
try:
    sg_id = ec2.create_security_group(
        GroupName=sg_name,
        Description="AEInnova Streamlit Dashboard",
        VpcId=vpc_id
    )["GroupId"]

    # Autoritzem tràfic TCP entrant al port 8501 des de qualsevol IP (0.0.0.0/0)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},  # Qualsevol IP pot accedir
        ]
    )
    print(f"  Security group creat: {sg_id}")
except ec2.exceptions.ClientError as e:
    if "InvalidGroup.Duplicate" in str(e):
        # Si el Security Group ja existia, recuperem el seu ID
        sg_id = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [sg_name]},
                     {"Name": "vpc-id",     "Values": [vpc_id]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  Security group ja existeix: {sg_id}")
    else:
        raise


# ── 5. Obtenir la AMI més recent d'Amazon Linux 2023 ──────────────────────────
# AMI (Amazon Machine Image) és la imatge del sistema operatiu que s'instal·la
# a la instència. Fem servir Amazon Linux 2023, que és la distribució Linux
# oficial d'Amazon, optimitzada per a EC2.
#
# SSM Parameter Store és un servei d'AWS que emmagatzema configuració.
# Amazon publica la última AMI disponible a un paràmetre SSM públic,
# així sempre obtenim la versió més actualitzada del sistema operatiu.
step("5 · Obtenint AMI Amazon Linux 2023 (última versió)")
ssm    = boto3.client("ssm", region_name=REGION)
ami_id = ssm.get_parameter(
    Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
)["Parameter"]["Value"]
print(f"  AMI: {ami_id}")


# ── 6. Script d'arrencada de la instència (User Data) ─────────────────────────
# Aquest script bash s'executa automàticament la primera vegada que
# la instència EC2 arrenca, com a root (administrador del sistema).
# S'encarrega de:
#   1. Actualitzar el sistema operatiu
#   2. Descarregar el codi del dashboard des de S3
#   3. Crear un entorn virtual Python per aïllar les dependències
#   4. Instal·lar les dependències (Streamlit, Plotly, boto3, etc.)
#   5. Configurar Streamlit com a servei systemd perquè s'arrenci
#      automàticament i es reiniciï si falla
user_data = f"""#!/bin/bash
set -e
# Actualitzar paquets del sistema operatiu
dnf update -y
dnf install -y python3 python3-pip python3-virtualenv

# Descarregar el codi del dashboard i les dependències des de S3.
# La instència pot accedir a S3 gràcies al rol IAM assignat (pas 3),
# sense necessitat de credencials explícites.
aws s3 cp s3://{BUCKET}/dashboard/requirements_dashboard.txt /home/ec2-user/requirements_dashboard.txt
aws s3 cp s3://{BUCKET}/dashboard/dashboard.py               /home/ec2-user/dashboard.py
chown ec2-user:ec2-user /home/ec2-user/requirements_dashboard.txt /home/ec2-user/dashboard.py

# Crear un entorn virtual Python per aïllar les dependències del dashboard
# de les llibreries del sistema operatiu (evita conflictes de versions)
python3 -m venv /home/ec2-user/venv
/home/ec2-user/venv/bin/pip install --upgrade pip
/home/ec2-user/venv/bin/pip install -r /home/ec2-user/requirements_dashboard.txt
chown -R ec2-user:ec2-user /home/ec2-user/venv

# Crear el fitxer de configuració del servei systemd per a Streamlit.
# systemd és el gestor de serveis de Linux: permet que Streamlit s'arrenci
# automàticament quan la instència arrenca i es reiniciï si falla.
cat > /etc/systemd/system/streamlit.service << 'SVCEOF'
[Unit]
Description=AEInnova Predictive Maintenance Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment=AWS_DEFAULT_REGION={REGION}
ExecStart=/home/ec2-user/venv/bin/streamlit run dashboard.py \
    --server.port={PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SVCEOF

# Activar i iniciar el servei
systemctl daemon-reload   # Recarregar la configuració de systemd
systemctl enable streamlit  # Activar el servei perquè s'arrenci en cada reboot
systemctl start streamlit   # Iniciar el servei ara mateix
"""


# ── 7. Llançar la instència EC2 ────────────────────────────────────────────────
# Comprovem si ja existeix una instència en marxa per evitar duplicats.
# Si ja existeix, informem de la seva IP i no en creem una de nova.
step("6 · Llançant instència EC2 t3.micro")

existing = ec2.describe_instances(Filters=[
    {"Name": "tag:Name",            "Values": [APP_NAME]},
    {"Name": "instance-state-name", "Values": ["running", "pending"]},
])["Reservations"]

if existing:
    # Ja existeix una instència: mostrem la seva IP i no fem res
    inst        = existing[0]["Instances"][0]
    instance_id = inst["InstanceId"]
    public_ip   = inst.get("PublicIpAddress", "pendent...")
    print(f"  Instència ja en marxa: {instance_id} ({public_ip})")
    print("  Per redesplegar, atura-la primer:")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id} --region {REGION}")
else:
    # Llançar una nova instència EC2
    resp = ec2.run_instances(
        ImageId      = ami_id,          # Sistema operatiu (Amazon Linux 2023)
        InstanceType = "t3.micro",      # Tipus de màquina (2 vCPU, 1 GB RAM)
        MinCount=1, MaxCount=1,
        SecurityGroupIds   = [sg_id],          # Tallafocs que obre el port 8501
        IamInstanceProfile = {"Name": profile_name},  # Rol per accedir a S3
        UserData           = user_data,        # Script d'instal·lació automàtica
        TagSpecifications  = [{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name",    "Value": APP_NAME},
                     {"Key": "Project", "Value": "AEInnova-TFG"}]
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"  Instència creada: {instance_id}")

    # Esperar que la instència obtingui una IP pública (pot trigar uns segons)
    print("  Esperant IP pública", end="", flush=True)
    for _ in range(30):
        inst = ec2.describe_instances(
            InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
        if inst.get("PublicIpAddress"):
            public_ip = inst["PublicIpAddress"]
            print(f" → {public_ip}")
            break
        print(".", end="", flush=True)
        time.sleep(5)
    else:
        public_ip = "<IP_PUBLICA>"
        print("\n  Comprova la IP a la consola AWS EC2.")


# ── 8. Resultat final ──────────────────────────────────────────────────────────
step("7 · Desplegament completat!")
print(f"""
  Instència  : {instance_id}
  IP pública : {public_ip}

  El dashboard s'instal·la en segon pla (~3 min).
  Un cop llest, accedeix a:

      http://{public_ip}:{PORT}

  ─────────────────────────────────────────────
  Per ATURAR la instència (evitar costos):
    aws ec2 stop-instances --instance-ids {instance_id} --region {REGION}

  Per REINICIAR-LA:
    aws ec2 start-instances --instance-ids {instance_id} --region {REGION}
  ─────────────────────────────────────────────
""")
