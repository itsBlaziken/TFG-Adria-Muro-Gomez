# Sistema escalable de detecció i classificació d'anomalies industrials mitjançant AWS SageMaker

**Adrià Muro Gómez** — Grau d'Enginyeria de Dades, UAB  
Tutor: Raúl Aragonés Ortíz (Departament de Microelectrònica i Sistemes Electrònics)  
Curs 2025/26

---

## Resum

Es presenta un sistema de detecció i classificació d'anomalies per al manteniment predictiu de maquinària rotativa industrial a partir de dades de sensors d'alta precisió. Les senyals de vibració es representen mitjançant vectors de característiques espectrals i estadístiques derivades de les mesures en posició i freqüència. El dataset proporcionat per l'empresa AEInnova comprèn registres reals de dispositius de la família NOD-0007.

En una primera etapa, s'entrena un model **Isolation Forest** amb normalització robusta i ajust del paràmetre de contaminació per detectar comportaments anòmals. En una segona etapa, les anomalies detectades es processen mitjançant tècniques de clustering (**K-Means**) per identificar patrons naturals en les dades, i s'entrena un model supervisat de **Gradient Boosting** capaç de classificar-les en diferents tipologies de fallada amb coherència física (desalineació, desbalanceig i degradació).

El pipeline complet —preprocessament, modelatge i generació d'alertes— es desplega a **AWS SageMaker** amb emmagatzematge centralitzat a **Amazon S3** i visualització mitjançant un dashboard interactiu desplegat a **Amazon EC2**.

**Paraules clau:** manteniment predictiu, classificació de fallades, detecció d'anomalies, Isolation Forest, Gradient Boosting, anàlisi de vibració, AWS SageMaker, sensors industrials.

---

## Descripció del projecte

El projecte es desenvolupa en col·laboració amb **AEInnova**, empresa especialitzada en la monitorització industrial, que proporciona dades reals de sensors instal·lats en màquines rotatives. Les dades comprenen espectres de vibració i velocitat de rotació recollits per dispositius de la família NOD-0007.

El sistema combina tècniques d'aprenentatge no supervisat i supervisat en un pipeline estructurat:

1. Identificació d'anomalies mitjançant models basats en aïllament (Isolation Forest)
2. Descoberta de patrons estructurats en les dades (K-Means)
3. Classificació en tipologies de fallada amb significat físic (Gradient Boosting)
4. Desplegament complet en entorn cloud AWS SageMaker

---

## Conjunt de dades

Les dades provenen del **DatasetI** d'AEInnova, format per registres reals en format JSON per línies (estructura compatible amb exportacions DynamoDB). Cada registre conté:

- Identificador del dispositiu (`dev_eui`)
- Mode de captació (`mode`)
- Eix de mesura (`axis`)
- Marca temporal (`timestamp`)
- Temperatura (`T1`)
- Espectre de vibració (`readings`)

El camp `readings` representa el senyal en el domini de la freqüència com una col·lecció de parelles freqüència–amplitud. Cada espectre es transforma en un vector de 15 característiques estadístiques i espectrals que constitueixen l'entrada dels models.

El pipeline s'ha executat sobre un total de **21.789 registres** de vibració.

---

## Metodologia

### Extracció de característiques espectrals

Cada espectre es transforma en un vector de 15 característiques:

| Característica | Descripció |
|---|---|
| Amplitud màxima / mitjana | Intensitat general del senyal |
| Desviació estàndard | Variabilitat del senyal |
| RMS | Energia efectiva del senyal |
| Freqüència dominant | Freqüència de màxima amplitud |
| Energia total | Suma de potències espectrals |
| Factor de cresta | Relació pic/RMS |
| Temperatura | Condició operativa |
| Energia per bandes (baixa/mitjana/alta) | Distribució freqüencial |
| Proporcions relatives d'energia | Ratis normalitzats per banda |

### Pipeline de detecció i classificació

```
JSON raw data
     │
     ▼
Preprocessament + extracció de característiques
     │
     ▼
Isolation Forest  →  Normal / Anòmal
     │ (anomalies)
     ▼
K-Means (k=4)  →  Patrons espectrals diferenciats
     │
     ▼
Gradient Boosting  →  Tipologia de fallada
     │
     ▼
Alertes estructurades + Dashboard
```

### Tipologies de fallada detectades

| Clúster | Tipologia | Característica principal |
|---|---|---|
| 0 | Desalineació | Energia en banda mitjana (30–70 Hz) |
| 1 | Inestabilitat estructural | Alta variabilitat d'amplitud |
| 2 | Degradació general | Distribució homogènia |
| 3 | Desbalanceig mecànic | Energia en baixa freqüència (<30 Hz) |

---

## Resultats

El model Isolation Forest classifica **17.431 (80%)** registres com a normals i **4.358 (20%)** com a anòmals, coherent amb el paràmetre de contaminació configurat.

El clustering K-Means obté un **Silhouette Score de 0.4553**, indicant una separació moderada i consistent entre clústers en un context industrial no supervisat.

Distribució de tipologies detectades:
- Desalineació: 38,3%
- Inestabilitat estructural: 33,0%
- Degradació general: 15,1%
- Desbalanceig mecànic: 13,7%

---

## Arquitectura AWS

```
Amazon S3 (emmagatzematge centralitzat)
    ├── data/raw/          ← Dataset original
    ├── data/processed/    ← Característiques calculades
    ├── models/            ← Models entrenats (.pkl)
    ├── predictions/       ← Alertes generades
    └── dashboard/         ← Codi del dashboard

SageMaker Processing Job  ← Preprocessament i extracció
SageMaker Training Job    ← Entrenament dels models
Amazon EC2 (t3.micro)     ← Dashboard Streamlit (port 8501)
```

El dashboard llegeix dades directament des de S3 en cada càrrega, de manera que qualsevol execució nova del pipeline es reflecteix automàticament sense redesplegar res.

---

## Estructura del repositori

```
TFG/
├── aws/
│   ├── scripts/
│   │   ├── preprocess.py          # SageMaker Processing Job
│   │   ├── train.py               # SageMaker Training Job
│   │   └── inference.py           # Batch inference
│   ├── config.py                  # Configuració AWS (bucket, regió, rols)
│   ├── setup_s3.py                # Creació i organització del bucket
│   ├── run_processing.py          # Llança el Processing Job
│   ├── train_local_upload.py      # Entrenament local + pujada a S3
│   ├── inference_local_upload.py  # Inferència local + pujada a S3
│   ├── generate_alerts.py         # Genera alertes estructurades
│   ├── dashboard.py               # Dashboard Streamlit
│   ├── deploy_dashboard.py        # Desplegament automàtic a EC2
│   ├── create_role.py             # Creació del rol IAM
│   ├── Dockerfile                 # Imatge per a SageMaker
│   └── requirements*.txt
├── local_execution/
│   ├── train_fault_types.py       # Entrenament en local
│   ├── predict_faults_aeinnova.py # Inferència en local
│   ├── analyze_visualizations.py  # Anàlisi exploratòria
│   └── evaluate_metrics.py        # Mètriques de validació
├── models/                        # Models entrenats (.pkl)
├── data/                          # Dataset 
├── doc/                           # Memòria i figures del TFG
└── outputs/                       # Alertes i prediccions generades
```

---

## Tecnologies

- **Python** — NumPy, Pandas, Scikit-learn
- **Machine Learning** — Isolation Forest, K-Means, Gradient Boosting
- **Anàlisi de senyals** — FFT, domini freqüencial, extracció de característiques espectrals
- **AWS** — SageMaker, S3, EC2, IAM
- **Visualització** — Streamlit, Plotly
- **Altres** — Docker, Git LFS

---

## Execució en local

```bash
# Entrenament
python local_execution/train_fault_types.py

# Inferència i generació de prediccions
python local_execution/predict_faults_aeinnova.py

# Anàlisi exploratòria i visualitzacions
python local_execution/analyze_visualizations.py

# Mètriques de validació
python local_execution/evaluate_metrics.py
```

## Desplegament a AWS

```bash
# 1. Configurar credencials AWS i paràmetres a aws/config.py

# 2. Crear estructura S3 i pujar dades
python aws/setup_s3.py

# 3. Executar preprocessament (SageMaker Processing Job)
python aws/run_processing.py

# 4. Entrenar models i pujar artefactes a S3
python aws/train_local_upload.py

# 5. Generar prediccions i alertes
python aws/inference_local_upload.py
python aws/generate_alerts.py

# 6. Desplegar dashboard a EC2
python aws/deploy_dashboard.py
```

---

## Treball futur

- **Etiquetes reals**: incorporar registres validats per tècnics d'AEInnova per substituir l'aproximació no supervisada per models supervisats entrenats sobre veritat de camp.
- **Ingesta en temps real**: integrar Amazon Kinesis per processar cada lectura en el moment d'arribada i reduir la latència de detecció.

---

## Autor

**Adrià Muro Gómez**  
Grau d'Enginyeria de Dades — Universitat Autònoma de Barcelona  
Contacte: Adria.Muro@autonoma.cat

Tutor: Raúl Aragonés Ortíz  
Departament de Microelectrònica i Sistemes Electrònics, UAB
