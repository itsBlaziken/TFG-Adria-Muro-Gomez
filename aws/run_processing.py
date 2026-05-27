# ─────────────────────────────────────────────────────────────────────────────
# RUN_PROCESSING.PY — Llança el SageMaker Processing Job
#
# QUÈ FA:
#   Llança un SageMaker Processing Job que executa scripts/preprocess.py
#   dins d'un contenidor scikit-learn gestionat per AWS. El job llegeix el
#   dataset JSON de S3, calcula les característiques espectrals de cada
#   registre i desa el resultat com a features.csv a S3.
#
# QUÈ ÉS UN PROCESSING JOB DE SAGEMAKER?
#   És una tasca de processament de dades que s'executa en un ordinador
#   virtual d'AWS. Funciona així:
#     1. AWS arrenca un contenidor Docker amb scikit-learn preinstal·lat
#     2. Copia els fitxers d'entrada de S3 al contenidor
#     3. Executa el script de preprocessament
#     4. Copia els fitxers de sortida del contenidor a S3
#     5. Apaga el contenidor automàticament
#   No cal gestionar cap servidor: AWS ho fa tot.
#
# QUÈ ÉS UN CONTENIDOR DOCKER?
#   És un paquet que conté un sistema operatiu mínim, les llibreries Python
#   necessàries i el codi. Garanteix que el codi s'executa sempre igual
#   independentment de la màquina on corri.
#
# DURADA APROXIMADA: ~10 minuts (inclou el temps d'arrencada del contenidor)
#
# ÚS:
#   python run_processing.py
# ─────────────────────────────────────────────────────────────────────────────

import boto3
import time
from config import REGION, ROLE_NAME, S3_RAW, S3_PROCESSED, get_bucket_name

# Clients AWS necessaris
sm  = boto3.client("sagemaker", region_name=REGION)  # Per crear i monitoritzar jobs
s3  = boto3.client("s3",        region_name=REGION)  # Per pujar el script a S3
iam = boto3.client("iam")                             # Per obtenir l'ARN del rol

# Recupera l'ARN (identificador únic) del rol IAM creat amb create_role.py.
# SageMaker necessita aquest ARN per saber amb quins permisos executar el job.
role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
bucket   = get_bucket_name()

# Nom únic per al job basat en el timestamp actual. SageMaker requereix que
# cada job tingui un nom diferent, fins i tot si fan el mateix.
job_name = f"aeinnova-preprocess-{int(time.time())}"

# URI del contenidor scikit-learn d'Amazon. Aquest contenidor ja té Python,
# scikit-learn, pandas i numpy instal·lats. El número "141502667606" és
# l'account d'Amazon que publica imatges oficials de ML per a SageMaker.
IMAGE_URI = f"141502667606.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def upload_script(local_path, s3_key):
    """Puja el script de preprocessament a S3 perquè el contenidor el pugui llegir.

    El contenidor no té accés al sistema de fitxers local. Per tant, cal
    pujar el script a S3 primer, i SageMaker el copiarà al contenidor
    com a canal d'entrada.
    """
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Script pujat: s3://{bucket}/{s3_key}")


def wait_for_job(job_name):
    """Espera que el job acabi comprovant l'estat cada 30 segons.

    Els estats possibles d'un Processing Job són:
      - InProgress: el job s'està executant
      - Completed:  el job ha acabat correctament
      - Failed:     el job ha fallat (consultar logs a CloudWatch)
      - Stopped:    el job s'ha aturat manualment
    """
    print(f"\nEsperant el job '{job_name}'...")
    while True:
        resp   = sm.describe_processing_job(ProcessingJobName=job_name)
        status = resp["ProcessingJobStatus"]
        print(f"  [{time.strftime('%H:%M:%S')}] Estat: {status}")
        if status == "Completed":
            return resp
        if status in ("Failed", "Stopped"):
            raise RuntimeError(f"Job fallat ({status}): {resp.get('FailureReason', '')}")
        time.sleep(30)  # Esperem 30 segons abans de tornar a comprovar


# ─────────────────────────────────────────────────────────────────────────────
# EXECUCIÓ PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

# Pas 1: Pujar el script a S3 perquè el contenidor el pugui descarregar
upload_script("scripts/preprocess.py", "code/preprocess.py")

print(f"\nLlançant Processing Job: {job_name}")
print(f"  Input:  s3://{bucket}/{S3_RAW}/")
print(f"  Output: s3://{bucket}/{S3_PROCESSED}/")

# Pas 2: Crear el Processing Job a SageMaker
sm.create_processing_job(
    ProcessingJobName=job_name,
    RoleArn=role_arn,

    # AppSpecification: defineix quin contenidor usar i quin script executar.
    # ContainerEntrypoint és l'equivalent a "python3 preprocess.py --input ... --output ..."
    AppSpecification={
        "ImageUri": IMAGE_URI,
        "ContainerEntrypoint": [
            "python3", "/opt/ml/processing/input/code/preprocess.py",
            "--input-data",  "/opt/ml/processing/input/raw",    # on el contenidor trobarà el dataset
            "--output-data", "/opt/ml/processing/output",       # on el contenidor ha de desar el resultat
        ],
    },

    # ProcessingInputs: defineix els fitxers que SageMaker ha de copiar de S3 al contenidor
    # abans d'executar el script. Cada canal té un nom, una ruta a S3 i una ruta local
    # dins del contenidor on es copiaran els fitxers.
    ProcessingInputs=[
        {
            "InputName": "raw",      # Canal per al dataset original
            "S3Input": {
                "S3Uri":       f"s3://{bucket}/{S3_RAW}/",
                "LocalPath":   "/opt/ml/processing/input/raw",  # ruta dins del contenidor
                "S3DataType":  "S3Prefix",   # copiar tot el que hi ha sota el prefix
                "S3InputMode": "File",
            },
        },
        {
            "InputName": "code",     # Canal per al script de preprocessament
            "S3Input": {
                "S3Uri":       f"s3://{bucket}/code/preprocess.py",
                "LocalPath":   "/opt/ml/processing/input/code",
                "S3DataType":  "S3Prefix",
                "S3InputMode": "File",
            },
        },
    ],

    # ProcessingOutputConfig: defineix on ha de copiar SageMaker els fitxers de sortida
    # un cop el script hagi acabat. EndOfJob significa que la còpia es fa al final.
    ProcessingOutputConfig={
        "Outputs": [
            {
                "OutputName": "processed",
                "S3Output": {
                    "S3Uri":        f"s3://{bucket}/{S3_PROCESSED}/",
                    "LocalPath":    "/opt/ml/processing/output",  # on el script ha desat el resultat
                    "S3UploadMode": "EndOfJob",                   # puja quan el job acaba
                },
            }
        ]
    },

    # ProcessingResources: tipus de màquina virtual on s'executarà el job.
    # ml.t3.medium és la instència més petita compatible amb Processing Jobs,
    # adequada per a datasets de mida moderada i compatible amb les quotes
    # d'un compte nou d'AWS.
    ProcessingResources={
        "ClusterConfig": {
            "InstanceCount":  1,               # Nombre de màquines (1 n'hi ha prou)
            "InstanceType":   "ml.t3.medium",  # 2 vCPU, 4 GB RAM
            "VolumeSizeInGB": 30,              # Espai en disc temporal del contenidor
        }
    },
)

# Pas 3: Esperar que el job acabi
wait_for_job(job_name)

print(f"\nProcessing Job completat.")
print(f"Features a: s3://{bucket}/{S3_PROCESSED}/features.csv")

# Desa el nom del job en un fitxer local per referència futura (logs, debugging)
with open(".last_processing_job", "w") as f:
    f.write(job_name)

print("Fet")
