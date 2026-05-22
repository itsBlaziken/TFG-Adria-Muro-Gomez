# Desplega el dashboard Streamlit a AWS EC2 t3.micro (sense Docker).
# Puja dashboard.py a S3, crea la infraestructura IAM/SG i llança la instància.
# Executar amb: python deploy_dashboard.py

import boto3
import json
import time
import sys
import os

# Rutes relatives al directori on es troba aquest script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGION   = "eu-west-1"
BUCKET   = "aeinnova-tfg-836321169819"
APP_NAME = "aeinnova-dashboard"
PORT     = 8501

def step(msg):
    print(f"\n{'─'*60}\n{msg}\n{'─'*60}")

# ── 1. Compte AWS ─────────────────────────────────────────────────────────────
step("1 · Obtenint compte AWS")
sts        = boto3.client("sts", region_name=REGION)
account_id = sts.get_caller_identity()["Account"]
print(f"  Account: {account_id}")

# ── 2. Pujar fitxers a S3 ─────────────────────────────────────────────────────
step("2 · Pujant dashboard.py i requirements a S3")
s3 = boto3.client("s3", region_name=REGION)
s3.upload_file(os.path.join(SCRIPT_DIR, "dashboard.py"),               BUCKET, "dashboard/dashboard.py")
s3.upload_file(os.path.join(SCRIPT_DIR, "requirements_dashboard.txt"), BUCKET, "dashboard/requirements_dashboard.txt")
print(f"  Pujat a s3://{BUCKET}/dashboard/")

# ── 3. Rol IAM + Instance Profile ─────────────────────────────────────────────
step("3 · Creant rol IAM per a l'instància EC2")
iam          = boto3.client("iam")
role_name    = f"{APP_NAME}-ec2-role"
profile_name = f"{APP_NAME}-ec2-profile"

trust = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "ec2.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]
})

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
    print("  Esperant propagació IAM (15 s)...")
    time.sleep(15)
except iam.exceptions.EntityAlreadyExistsException:
    print(f"  Instance profile ja existeix: {profile_name}")

# ── 4. Security Group ─────────────────────────────────────────────────────────
step("4 · Creant Security Group (port 8501)")
ec2    = boto3.client("ec2", region_name=REGION)
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
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {"IpProtocol": "tcp", "FromPort": PORT, "ToPort": PORT,
             "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        ]
    )
    print(f"  Security group creat: {sg_id}")
except ec2.exceptions.ClientError as e:
    if "InvalidGroup.Duplicate" in str(e):
        sg_id = ec2.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": [sg_name]},
                     {"Name": "vpc-id",     "Values": [vpc_id]}]
        )["SecurityGroups"][0]["GroupId"]
        print(f"  Security group ja existeix: {sg_id}")
    else:
        raise

# ── 5. AMI Amazon Linux 2023 ──────────────────────────────────────────────────
step("5 · Obtenint AMI Amazon Linux 2023 (última versió)")
ssm    = boto3.client("ssm", region_name=REGION)
ami_id = ssm.get_parameter(
    Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
)["Parameter"]["Value"]
print(f"  AMI: {ami_id}")

# ── 6. User data (script d'arrencada de la instància) ─────────────────────────
user_data = f"""#!/bin/bash
set -e
dnf update -y
dnf install -y python3 python3-pip python3-virtualenv

# Descarregar fitxers des de S3
aws s3 cp s3://{BUCKET}/dashboard/requirements_dashboard.txt /home/ec2-user/requirements_dashboard.txt
aws s3 cp s3://{BUCKET}/dashboard/dashboard.py               /home/ec2-user/dashboard.py
chown ec2-user:ec2-user /home/ec2-user/requirements_dashboard.txt /home/ec2-user/dashboard.py

# Virtualenv per evitar conflictes amb paquets del sistema (RPM)
python3 -m venv /home/ec2-user/venv
/home/ec2-user/venv/bin/pip install --upgrade pip
/home/ec2-user/venv/bin/pip install -r /home/ec2-user/requirements_dashboard.txt
chown -R ec2-user:ec2-user /home/ec2-user/venv

# Servei systemd
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

systemctl daemon-reload
systemctl enable streamlit
systemctl start streamlit
"""

# ── 7. Llançar instància EC2 ──────────────────────────────────────────────────
step("6 · Llançant instància EC2 t3.micro")

# Evitar duplicats
existing = ec2.describe_instances(Filters=[
    {"Name": "tag:Name",            "Values": [APP_NAME]},
    {"Name": "instance-state-name", "Values": ["running", "pending"]},
])["Reservations"]

if existing:
    inst        = existing[0]["Instances"][0]
    instance_id = inst["InstanceId"]
    public_ip   = inst.get("PublicIpAddress", "pendent...")
    print(f"  Instància ja en marxa: {instance_id} ({public_ip})")
    print("  Per redesplegar, atura-la primer:")
    print(f"  aws ec2 terminate-instances --instance-ids {instance_id} --region {REGION}")
else:
    resp = ec2.run_instances(
        ImageId      = ami_id,
        InstanceType = "t3.micro",
        MinCount=1, MaxCount=1,
        SecurityGroupIds    = [sg_id],
        IamInstanceProfile  = {"Name": profile_name},
        UserData            = user_data,
        TagSpecifications   = [{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name",    "Value": APP_NAME},
                     {"Key": "Project", "Value": "AEInnova-TFG"}]
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"  Instància creada: {instance_id}")

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

# ── 8. Resultat final ─────────────────────────────────────────────────────────
step("7 · Desplegament completat!")
print(f"""
  Instància  : {instance_id}
  IP pública : {public_ip}

  El dashboard s'instal·la en segon pla (~3 min).
  Un cop llest, accedeix a:

      http://{public_ip}:{PORT}

  ─────────────────────────────────────────────
  Per ATURAR la instància (evitar costos):
    aws ec2 stop-instances --instance-ids {instance_id} --region {REGION}

  Per REINICIAR-LA:
    aws ec2 start-instances --instance-ids {instance_id} --region {REGION}
  ─────────────────────────────────────────────
""")
