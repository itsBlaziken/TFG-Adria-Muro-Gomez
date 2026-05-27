import boto3

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓ CENTRAL DEL PROJECTE AWS
#
# Aquest fitxer centralitza tots els paràmetres de configuració que utilitzen
# la resta d'scripts. Modificar aquí qualsevol valor el canvia a tot el projecte.
# ─────────────────────────────────────────────────────────────────────────────

# Regió AWS on es creen tots els recursos (S3, SageMaker, EC2).
# "eu-west-1" correspon a Irlanda. Tots els recursos han d'estar a la
# mateixa regió per evitar costos de transferència de dades entre regions.
REGION = "eu-west-1"


def get_bucket_name():
    """
    Retorna el nom del bucket S3 del projecte.

    El nom del bucket S3 ha de ser únic a tot AWS (a nivell global, no només
    al teu compte). Per garantir-ho, s'afegeix el número de compte AWS al nom.
    Exemple: "aeinnova-tfg-123456789012"

    boto3.client("sts") és el servei d'AWS que retorna informació del compte
    actual (STS = Security Token Service).
    """
    account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return f"aeinnova-tfg-{account_id}"


# Nom del rol IAM que SageMaker utilitza per accedir a S3 i executar jobs.
# Un rol IAM és com un carnet d'identitat amb permisos: SageMaker l'assumeix
# quan llança un job i així pot llegir/escriure a S3 sense necessitar
# credencials explícites al codi.
ROLE_NAME = "AEInnovaSageMakerRole"

# ─────────────────────────────────────────────────────────────────────────────
# PREFIXOS S3
#
# Dins del bucket, les dades s'organitzen en "carpetes virtuals" (prefixos).
# S3 no té carpetes reals; un prefix és simplement una part del nom del fitxer
# que actua com a ruta. Ex: "raw/DatasetI.json" → prefix "raw/", fitxer "DatasetI.json"
# ─────────────────────────────────────────────────────────────────────────────
S3_RAW       = "raw"        # Dataset original tal com ve dels sensors
S3_PROCESSED = "processed"  # Característiques espectrals calculades (features.csv)
S3_MODELS    = "models"     # Models entrenats empaquetats (model.tar.gz)
S3_OUTPUTS   = "outputs"    # Prediccions i alertes generades pel pipeline

# Ruta local al dataset (relativa al directori aws/)
LOCAL_DATA = "../data/DatasetI.json"
