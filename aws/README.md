# AEInnova — Sistema de Detecció d'Anomalies en AWS

> TFG · Adrià Muro Gómez · UAB 2025/26

---

## Què és AWS i per què s'utilitza aquí?

**AWS (Amazon Web Services)** és una plataforma de serveis en el núvol d'Amazon. En lloc de tenir un servidor físic propi, llogues recursos de computació i emmagatzematge per internet, pagues només pel que uses i pots escalar fàcilment.

En aquest projecte s'utilitzen tres serveis principals:

| Servei | Analogia senzilla | Per a què s'utilitza |
|--------|------------------|----------------------|
| **Amazon S3** | Disc dur al núvol | Guardar el dataset, els models entrenats i els resultats |
| **AWS SageMaker** | Ordinador virtual per a ML | Executar el preprocessament de dades |
| **Amazon EC2** | Servidor virtual | Allotjar el dashboard web accessible des del navegador |
| **AWS IAM** | Sistema de permisos | Controlar qui pot accedir a cada recurs d'AWS |

---

## Conceptes d'AWS que cal entendre

**Bucket S3**: és com una carpeta principal al núvol. Dins hi pots crear subcarpetes (prefixos) per organitzar les dades. El nom ha de ser únic a tot AWS, per això s'hi afegeix el número de compte.

**Rol IAM**: és com un carnet d'identitat amb permisos. Quan SageMaker o EC2 necessiten llegir fitxers de S3, ho fan amb un rol que els autoritza. Així no cal posar contrasenyes al codi.

**Processing Job de SageMaker**: és com enviar un script Python a executar en un ordinador virtual gestionat per AWS. AWS arrenca el contenidor, executa el codi, guarda el resultat a S3 i apaga la màquina automàticament.

**Contenidor Docker**: és un paquet que inclou el sistema operatiu, les llibreries i el codi, de manera que sempre s'executa igual independentment de la màquina. SageMaker utilitza contenidors predefinits d'Amazon per a scikit-learn, PyTorch, etc.

**Instància EC2**: és un ordinador virtual que funciona de manera contínua. S'utilitza per allotjar el dashboard Streamlit que els usuaris poden visitar des del navegador.

**Security Group**: és com un tallafocs virtual. Controla quins ports de la instència EC2 són accessibles des d'internet.

---

## Arquitectura del sistema

```
[Dataset JSON local]
        │
        ▼
  setup_s3.py ──────────► S3: raw/DatasetI.json
                                    │
                                    ▼
  run_processing.py ───► SageMaker Processing Job
                         (executa scripts/preprocess.py)
                                    │
                                    ▼
                          S3: processed/features.csv
                                    │
                                    ▼
  train_local_upload.py ──► entrena models en local
                                    │
                                    ▼
                          S3: models/model.tar.gz
                                    │
                                    ▼
  inference_local_upload.py ──► genera prediccions
                                    │
                                    ▼
                          S3: outputs/alerts.csv
                                    │
                                    ▼
  deploy_dashboard.py ──► EC2 t3.micro (Streamlit)
                          llegeix alerts.csv de S3
                          accessible per navegador
```

---

## Ordre d'execució

Els dos primers scripts s'executen **una sola vegada** per inicialitzar la infraestructura. La resta formen el pipeline i es poden tornar a executar quan es vulgui actualitzar els models o les prediccions.

### Pas 1 — `create_role.py` (una sola vegada)

Crea el **rol IAM** que SageMaker necessita per accedir a S3 i executar jobs. Sense aquest rol, SageMaker no tindria permisos per llegir el dataset ni desar els resultats.

Adjunta tres polítiques gestionades per Amazon:
- `AmazonSageMakerFullAccess` — permisos per crear i gestionar jobs
- `AmazonS3FullAccess` — permisos per llegir i escriure al bucket
- `CloudWatchLogsFullAccess` — permisos per escriure logs dels jobs

```bash
python create_role.py
```

---

### Pas 2 — `setup_s3.py` (una sola vegada)

Crea el **bucket S3** del projecte i puja el dataset original (`DatasetI.json`). El nom del bucket incorpora el número de compte AWS per garantir que és únic a tot el món (requisit d'Amazon).

```bash
python setup_s3.py
```

Estructura S3 que es crearà:
```
s3://aeinnova-tfg-<account_id>/
    raw/          ← dataset original
    processed/    ← característiques calculades (el pas 3 ho omple)
    models/       ← models entrenats (el pas 4 ho omple)
    outputs/      ← prediccions i alertes (el pas 5 ho omple)
    dashboard/    ← codi del dashboard (el pas 6 ho omple)
```

---

### Pas 3 — `run_processing.py`

Llança un **SageMaker Processing Job** que executa `scripts/preprocess.py` dins d'un contenidor scikit-learn gestionat per AWS. El job llegeix el dataset JSON de S3, calcula les 15 característiques espectrals de cada registre (RMS, energia per bandes, factor de cresta, etc.) i desa el resultat com a `features.csv` a S3.

El job s'executa sobre una instància `ml.t3.medium` (~10 minuts). AWS gestiona tot l'entorn automàticament.

```bash
python run_processing.py
```

---

### Pas 4 — `train_local_upload.py`

Descarrega el `features.csv` de S3, entrena els tres models del pipeline en local (Isolation Forest + K-Means + Gradient Boosting) i puja els models resultants a S3 empaquetats en un fitxer `model.tar.gz`.

S'entrena en local perquè els comptes nous d'AWS tenen quota 0 per a Training Jobs de SageMaker. El codi és idèntic al que s'executaria al núvol.

```bash
python train_local_upload.py
```

---

### Pas 5 — `inference_local_upload.py`

Descarrega els models i les característiques de S3, aplica el pipeline de predicció sobre tots els registres i puja els resultats a S3. Genera dos fitxers:
- `all_predictions.csv` — tots els registres amb la seva predicció
- `alerts.csv` — únicament els registres classificats com a anòmals

```bash
python inference_local_upload.py
```

---

### Pas 6 — `generate_alerts.py`

Llegeix les prediccions de S3 i genera el fitxer d'alertes final en format estructurat, afegint metadades del dispositiu (identificador, eix, timestamp) i el nivell de severitat (HIGH / MEDIUM / LOW) basat en el score de confiança del classificador.

```bash
python generate_alerts.py
```

---

### Pas 7 — `deploy_dashboard.py`

Desplega el dashboard Streamlit sobre una **instència EC2 t3.micro** (la més econòmica, ~0,01 $/hora). El script automatitza tot el procés:

1. Puja el codi del dashboard a S3
2. Crea un rol IAM perquè EC2 pugui llegir S3 sense credencials explícites
3. Crea un Security Group que obre el port 8501 (port de Streamlit)
4. Llança la instància amb un script d'arrencada que instal·la les dependències i arrenca Streamlit com a servei del sistema

Un cop desplegat, el dashboard és accessible des del navegador:
```
http://<IP_PUBLICA>:8501
```

```bash
python deploy_dashboard.py
```

**Important**: per evitar costos innecessaris, atura la instència quan no s'usi:
```bash
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region eu-west-1
```

---

## Fitxers de suport

| Fitxer | Descripció |
|--------|------------|
| `config.py` | Configuració central: regió, bucket, rol IAM i prefixos S3 |
| `dashboard.py` | Dashboard Streamlit interactiu connectat a S3 |
| `requirements_dashboard.txt` | Dependències Python del dashboard (s'instal·len a EC2) |
| `scripts/preprocess.py` | Codi que s'executa dins del Processing Job de SageMaker |
| `scripts/train.py` | Codi d'entrenament dels models (també reutilitzat en local) |
| `scripts/inference.py` | Codi d'inferència (també reutilitzat en local) |

---

## Costos estimats

| Recurs | Cost aproximat |
|--------|---------------|
| S3 (emmagatzematge ~1 GB) | ~0,02 $/mes |
| SageMaker Processing Job (ml.t3.medium, ~10 min) | ~0,05 $ per execució |
| EC2 t3.micro (en marxa contínua) | ~7 $/mes |
| EC2 t3.micro (parada quan no s'usa) | ~0 $ |

La major part del cost ve de la instència EC2 si es deixa en marxa. Per a un ús puntual, es recomana aturar-la entre sessions.
