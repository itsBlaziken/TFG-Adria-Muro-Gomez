# AEInnova — Sistema de Detecció d'Anomalies en AWS

> TFG · Adrià Muro Gómez · UAB 2025/26  
> Pipeline: Isolation Forest → K-Means → Gradient Boosting · AWS SageMaker + S3 + EC2

---

## Scripts del pipeline

Els scripts s'executen en ordre per construir el pipeline complet. Els dos primers (`create_role.py` i `setup_s3.py`) només cal executar-los una vegada per inicialitzar la infraestructura.

---

### `create_role.py`

Crea el rol IAM que SageMaker necessita per accedir a S3 i llançar jobs. Adjunta les polítiques `AmazonSageMakerFullAccess`, `AmazonS3FullAccess` i `CloudWatchLogsFullAccess`. Si el rol ja existeix, no fa res.

```bash
python create_role.py
```

---

### `setup_s3.py`

Crea el bucket S3 del projecte i puja el dataset original (`DatasetI.json`) al prefix `raw/`. El nom del bucket incorpora l'account ID per garantir la unicitat global. Si el bucket ja existeix, passa directament a la pujada del fitxer.

```bash
python setup_s3.py
```

---

### `run_processing.py`

Llança un SageMaker Processing Job que llegeix el dataset raw des de S3, extreu les característiques espectrals de cada registre (RMS, energia per bandes, factor de cresta, freqüència dominant, entre d'altres) i desa el resultat com a `features.csv` al prefix `processed/`. El job s'executa sobre una instància `ml.t3.medium` a la regió `eu-west-1`. Dura aproximadament 10 minuts.

```bash
python run_processing.py
```

---

### `train_local_upload.py`

Entrena el pipeline de models en local sobre el `features.csv` descarregat de S3: Isolation Forest per detectar anomalies, K-Means per agrupar-les en 4 tipologies i Gradient Boosting per classificar-les. Els models entrenats es serialitzen, s'empaqueten i es pugen al prefix `models/` de S3.

```bash
python train_local_upload.py
```

---

### `inference_local_upload.py`

Descarrega els models i el dataset de característiques des de S3, genera una predicció per a cada registre i puja els resultats consolidats al prefix `outputs/`. El fitxer de sortida inclou el tipus de fallada detectat, el score de confiança i el nivell de severitat per a cada registre.

```bash
python inference_local_upload.py
```

---

### `generate_alerts.py`

Descarrega les prediccions de S3 i genera el fitxer d'alertes final filtrant únicament els registres classificats com a anòmals. Aplica un postprocessament per consolidar deteccions per dispositiu i eix i desa el resultat localment.

```bash
python generate_alerts.py
```

---

### `deploy_dashboard.py`

Desplega el dashboard Streamlit sobre una instància EC2 `t3.micro` amb Amazon Linux 2023. El script puja el codi del dashboard i les seves dependències a S3, crea un rol IAM amb permisos de lectura sobre el bucket, configura un Security Group que obre el port 8501 i llança la instància amb un servei systemd que arrenca Streamlit automàticament. Si ja existeix una instància en marxa, informa de la seva IP i no en crea una de nova.

```bash
python deploy_dashboard.py
```

Un cop completat, el dashboard és accessible des del navegador a través de la IP pública de la instància al port 8501.

---

## Fitxers de suport

| Fitxer | Descripció |
|--------|------------|
| `config.py` | Configuració central: regió, bucket, rol IAM i prefixos S3 |
| `dashboard.py` | Dashboard Streamlit interactiu connectat a S3 |
| `requirements_dashboard.txt` | Dependències del dashboard (instal·lades a la instància EC2) |
| `scripts/preprocess.py` | Script que s'executa dins del Processing Job de SageMaker |
| `scripts/train.py` | Script d'entrenament dels models |
| `scripts/inference.py` | Script d'inferència |
