# AEInnova — Sistema de Detecció d'Anomalies en AWS
## Guia Completa d'Ús i Posada en Marcha

> **TFG** · Adrià Muro Gómez · UAB 2025/26  
> Pipeline: Isolation Forest → K-Means → Gradient Boosting · Desplegat a AWS SageMaker + S3

---

## Taula de Continguts

1. [Arquitectura General](#1-arquitectura-general)
2. [Estructura de Fitxers](#2-estructura-de-fitxers)
3. [Recursos AWS Creats](#3-recursos-aws-creats)
4. [Prerequisits (primera vegada)](#4-prerequisits-primera-vegada)
5. [Com Executar el Pipeline Complet](#5-com-executar-el-pipeline-complet)
6. [Com Iniciar el Dashboard](#6-com-iniciar-el-dashboard)
7. [Resum de Resultats Obtinguts](#7-resum-de-resultats-obtinguts)
8. [Costos AWS Estimats](#8-costos-aws-estimats)
9. [Solució de Problemes](#9-solució-de-problemes)

---

## 1. Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Cloud (eu-west-1)                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Amazon S3   │    │  SageMaker       │    │  SageMaker       │  │
│  │              │───▶│  Processing Job  │───▶│  Training Job    │  │
│  │  raw/        │    │  (preprocess.py) │    │  (train.py)      │  │
│  │  processed/  │◀───│                  │    │                  │  │
│  │  models/     │    └──────────────────┘    └──────────────────┘  │
│  │  outputs/    │                │                    │             │
│  └──────────────┘                ▼                    ▼             │
│         │              features.csv en S3      model.tar.gz en S3  │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Inference   │    │  outputs/        │    │  Streamlit       │  │
│  │  (local)     │───▶│  alerts.csv      │───▶│  Dashboard       │  │
│  │              │    │  all_predict.csv │    │  localhost:8501  │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Flux de dades

| Etapa | Servei | Entrada | Sortida |
|-------|--------|---------|---------|
| Preprocessament | SageMaker Processing Job | `DatasetI.json` (248 MB) | `features.csv` (21.789 files) |
| Entrenament | Local + S3 | `features.csv` | `model.tar.gz` |
| Inferència | Local + S3 | `features.csv` + models | `alerts.csv`, `all_predictions.csv` |
| Visualització | Streamlit | `alerts.csv` + `features.csv` | Dashboard interactiu |

---

## 2. Estructura de Fitxers

```
aws/
│
├── config.py                    # Configuració central: bucket, regió, role
│
├── scripts/                     # Scripts que corren DINS dels contenidors AWS
│   ├── preprocess.py            # Processing Job: JSON → features.csv
│   ├── train.py                 # Training Job: features → models
│   └── inference.py             # Inference: model_fn, predict_fn, output_fn
│
├── 0_create_role.py             # Crea el IAM Role de SageMaker (1 sola vegada)
├── 1_setup_s3.py                # Crea el bucket S3 i puja el dataset (1 sola vegada)
├── 2_run_processing.py          # Llança Processing Job a SageMaker
├── 3_train_local_upload.py      # Entrena en local i puja model.tar.gz a S3
├── 4_inference_local_upload.py  # Inferència en local i puja resultats a S3
├── 5_generate_alerts.py         # Descarrega resultats i genera CSV d'alertes local
├── 6_setup_quicksight.py        # (Opcional) Configura Amazon QuickSight
│
├── dashboard.py                 # Dashboard Streamlit interactiu
├── requirements.txt             # Dependències Python
└── README.md                    # Aquest fitxer
```

### Fitxers generats (no pujar a git)

```
aws/
├── .last_processing_job         # Nom de l'últim Processing Job
├── .last_training_job           # Nom de l'últim Training Job
└── .last_model_uri              # URI S3 de l'últim model.tar.gz

outputs/                         # Creat automàticament
├── alerts.csv                   # 4.358 alertes amb tipus de fallo
├── all_predictions.csv          # 21.789 prediccions (normals + anomalies)
└── predictions/
    └── features.csv.out         # Sortida raw del Batch Transform
```

---

## 3. Recursos AWS Creats

### Account ID: `836321169819` · Regió: `eu-west-1` (Irlanda)

| Recurs | Identificador | Descripció |
|--------|---------------|------------|
| **S3 Bucket** | `aeinnova-tfg-836321169819` | Emmagatzematge central del pipeline |
| **IAM Role** | `AEInnovaSageMakerRole` | Permisos per a SageMaker |

### Estructura S3

```
s3://aeinnova-tfg-836321169819/
│
├── raw/
│   └── DatasetI.json            # Dataset original AEInnova (248 MB)
│
├── code/
│   ├── preprocess.py            # Script Processing Job
│   └── source.tar.gz            # Scripts empaquetats (train + inference)
│
├── processed/
│   └── features.csv             # 21.789 registres amb 19 característiques
│
├── models/
│   └── model.tar.gz             # Isolation Forest + K-Means + Gradient Boosting
│
├── outputs/
│   ├── alerts.csv               # Alertes finals (4.358 anomalies)
│   ├── all_predictions.csv      # Totes les prediccions
│   └── predictions/
│       └── features.csv.out     # Sortida de la inferència
│
└── quicksight/
    └── manifest.json            # Manifest per a QuickSight (si s'utilitza)
```

---

## 4. Prerequisits (primera vegada)

Aquests passos **JA ESTAN FETS**. Només cal repetir-los si es crea un compte nou o es vol reiniciar tot des de zero.

### 4.1 AWS CLI configurat
```powershell
aws configure
# Introduir:
#   AWS Access Key ID:      [la teva clau]
#   AWS Secret Access Key:  [el teu secret]
#   Default region name:    eu-west-1
#   Default output format:  json
```

### 4.2 Verificar credencials
```powershell
aws sts get-caller-identity
# Ha de mostrar Account: 836321169819
```

### 4.3 Instal·lar dependències Python
```powershell
cd c:\Users\adria\OneDrive\Escritorio\TFG\aws
pip install -r requirements.txt
```

### 4.4 Crear IAM Role (1 sola vegada)
```powershell
python 0_create_role.py
```

### 4.5 Crear bucket S3 i pujar dataset (1 sola vegada)
```powershell
python 1_setup_s3.py
```

---

## 5. Com Executar el Pipeline Complet

### Escenari A: Primera execució / Reinici complet

Executa els scripts en ordre:

```powershell
cd c:\Users\adria\OneDrive\Escritorio\TFG\aws

# Pas 2: Preprocessament (SageMaker, ~10 min)
python 2_run_processing.py

# Pas 3: Entrenament (local + puja a S3, ~2 min)
python 3_train_local_upload.py

# Pas 4: Inferència (local + puja resultats a S3, ~1 min)
python 4_inference_local_upload.py

# Pas 5: Generar alertes locals (descarrega de S3, ~30 seg)
python 5_generate_alerts.py
```

### Escenari B: Reentrenar amb dades noves

Si el dataset ha canviat, repetir des del pas 2.  
Si només canvien els paràmetres del model, repetir des del pas 3.

### Escenari C: Només veure els resultats actuals

Els resultats ja estan a S3 i a `outputs/`. Obre directament el dashboard (veure Secció 6).

---

## 6. Com Iniciar el Dashboard

> **Cada vegada que obris VS Code** i vulguis veure el dashboard, executa:

### Opció A: Des del terminal de VS Code

```powershell
cd c:\Users\adria\OneDrive\Escritorio\TFG\aws
streamlit run dashboard.py
```

Obre el navegador a: **http://localhost:8501**

### Opció B: Des de l'Explorador de Windows

Fes doble clic al fitxer `iniciar_dashboard.bat` (veure instruccions de creació a baix).

### Crear accés directe (recomanat)

Crea un fitxer `iniciar_dashboard.bat` a l'escriptori amb aquest contingut:

```bat
@echo off
cd /d "c:\Users\adria\OneDrive\Escritorio\TFG\aws"
streamlit run dashboard.py
pause
```

### Aturar el dashboard

Prem `Ctrl+C` al terminal on s'està executant.

### Funcionalitats del dashboard

| Secció | Visualitzacions |
|--------|-----------------|
| **KPIs** | Total registres, anomalies detectades, % taxa, severitat alta, accuracy CV |
| **Distribucions** | Donut per tipus de fallo, barres per severitat |
| **Per dispositiu** | Barres apilades per dev_eui, heatmap dispositiu × fallo |
| **Per eix** | Barres agrupades per eix X/Y/Z |
| **Espectral** | Histograma anomaly score, scatter RMS vs Energia |
| **Box plots** | Distribució de qualsevol característica per tipus de fallo |
| **Taula** | Top 20 alertes per severitat (coloritzada) |
| **Filtres** | Sidebar interactiu per severitat i tipus de fallo |

---

## 7. Resum de Resultats Obtinguts

### Pipeline ML

| Model | Paràmetres | Resultat |
|-------|-----------|---------|
| Isolation Forest | contamination=20%, n_estimators=100 | 4.358 anomalies (20%) |
| K-Means | k=4 clusters | 4 tipologies diferenciades |
| Gradient Boosting | n_estimators=100, lr=0.1, max_depth=5 | CV 99.45% ± 0.20% |

### Distribució de fallos

| Tipus | Descripció | Alertes |
|-------|-----------|---------|
| Type 1 | Mid-Frequency Energy (desalineació) | 1.667 (38.3%) |
| Type 0 | High Amplitude Variability (holgura/impacte) | 1.439 (33.0%) |
| Type 3 | Distributed Energy (degradació general) | 656 (15.1%) |
| Type 2 | Low-Frequency Energy (desbalanç) | 596 (13.7%) |

### Distribució per severitat

| Nivell | Alertes |
|--------|---------|
| HIGH (score ≥ 0.5) | 3.645 (83.7%) |
| MEDIUM (score ≥ 0.3) | 713 (16.4%) |

---

## 8. Costos AWS Estimats

| Servei | Ús realitzat | Cost aprox. |
|--------|-------------|-------------|
| S3 Storage | ~300 MB | ~$0.007/mes |
| S3 Requests | Upload/Download | ~$0.01 |
| SageMaker Processing Job | 1× ml.t3.medium, ~5 min | ~$0.01 |
| **Total acumulat** | | **~$0.03** |

> **Nota**: El Processing Job és l'únic cost de SageMaker. L'entrenament i la inferència es fan en local, per la qual cosa no generen cost de SageMaker.

### Per eliminar tots els recursos AWS (si cal)
```powershell
# Buidar i eliminar el bucket S3
aws s3 rm s3://aeinnova-tfg-836321169819 --recursive
aws s3api delete-bucket --bucket aeinnova-tfg-836321169819 --region eu-west-1

# Eliminar el IAM Role
aws iam detach-role-policy --role-name AEInnovaSageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name AEInnovaSageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name AEInnovaSageMakerRole --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
aws iam delete-role --role-name AEInnovaSageMakerRole
```

---

## 9. Solució de Problemes

### "No module named 'boto3'"
```powershell
pip install boto3 sagemaker pandas numpy scikit-learn
```

### "ResourceLimitExceeded" al Processing Job
El compte nou no té quota per a `ml.m5.large`. Ja s'ha configurat per usar `ml.t3.medium`.  
Si segueix fallant, comprova les quotes a:  
`AWS Console → Service Quotas → SageMaker → ml.t3.medium for processing job usage`

### "ResourceLimitExceeded" al Training o Transform Job
Els comptes nous tenen quota 0 per a Training i Transform Jobs.  
**Solució**: S'usa `3_train_local_upload.py` i `4_inference_local_upload.py` que corren en local i eviten el problema.

### El dashboard no obre
```powershell
# Comprova que Streamlit està instal·lat
streamlit --version

# Si no, instal·la:
pip install streamlit plotly

# Comprova que el port 8501 no està ocupat
netstat -ano | findstr :8501
```

### Error de credencials AWS al dashboard
```powershell
aws configure
# Re-introduir les credencials
```

### Credencials expirades
Les Access Keys no expiren per defecte en comptes root.  
Si has creat un usuari IAM, comprova la data d'expiració a la consola AWS.

---

## Flux ràpid de referència

```
Cada sessió nova de treball:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Obre VS Code
2. Obre un terminal (Ctrl+`)
3. cd c:\Users\adria\OneDrive\Escritorio\TFG\aws
4. streamlit run dashboard.py
5. Obre http://localhost:8501

Per re-executar el pipeline complet:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python 2_run_processing.py          # ~10 min (SageMaker)
python 3_train_local_upload.py      # ~2 min (local)
python 4_inference_local_upload.py  # ~1 min (local)
python 5_generate_alerts.py         # ~30 seg
streamlit run dashboard.py          # Dashboard
```

---

*Generat automàticament · TFG AEInnova · UAB 2025/26*
