# TFG – Sistema de detecció d’anomalies en maquinària rotativa

Aquest repositori conté el dossier tècnic del Treball Final de Grau d’Adrià Muro Gómez, corresponent al Grau d’Enginyeria de Dades (UAB).

## Descripció del projecte

El projecte se centra en el desenvolupament d’un sistema de detecció d’anomalies en processos industrials, aplicat a maquinària rotativa, a partir de dades de vibració i velocitat de rotació.

L’objectiu és identificar comportaments anòmals que puguin indicar possibles fallades mecàniques de manera primerenca, com a suport al manteniment predictiu.

El sistema està basat en tècniques de *machine learning* no supervisat, principalment mitjançant l’algorisme **Isolation Forest**, i incorpora també mètodes complementaris com clustering i regles espectrals basades en coneixement de domini per millorar la interpretació de les anomalies detectades.

## Estructura del repositori

TFG/
│
├── data/
│ ├── DatasetI.json
│ ├── mafaulda_improved.csv
│ ├── mafaulda_processed.csv
│
├── doc/
│ ├── informes i memòria del TFG
│ ├── presentació
│ ├── figures i resultats
│
├── models/
│ ├── models entrenats (.pkl)
│ ├── scalers i features
│ ├── clustering i classificadors
│
├── src/
│ ├── analyze_visualizations.py
│ ├── predict_faults_aeinova_v2.py
│ ├── train_fault_types_v2.py
│
├── .gitattributes
├── README.md

## Metodologia

El projecte segueix un enfocament modular:

1. Ingesta i preprocessament de dades de sensors industrials  
2. Extracció de característiques de senyals de vibració  
3. Segmentació per context operatiu (dispositiu, mode i eix)  
4. Detecció d’anomalies amb **Isolation Forest**  
5. Clustering i anàlisi exploratòria de patrons  
6. Interpretació mitjançant regles espectrals  
7. Validació amb dataset etiquetat de referència (MAFAULDA)

## Tecnologies utilitzades

- Python
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Machine Learning (Isolation Forest, K-Means)
- Anàlisi de senyals (FFT i domini freqüencial)

## Contingut destacat

- 📄 Informes tècnics del TFG
- 📊 Figures i visualitzacions de resultats
- 🤖 Models entrenats (`.pkl`)
- 📈 Pipeline complet de detecció d’anomalies
- 📉 Validació amb dataset industrial

## Observacions

- Aquest repositori actua com a suport documental del TFG.
- Les dades poden estar parcialment anonimitzades per motius de confidencialitat.
- L’informe final és el document principal d’avaluació acadèmica.

## Autor

**Adrià Muro Gómez**  
Grau d’Enginyeria de Dades – UAB  

Tutor: Raúl Aragonés Ortíz


## Objectiu final

Desenvolupar un sistema robust de detecció d’anomalies per a manteniment predictiu en maquinària industrial, reduint fallades i millorant la interpretació de comportaments anòmals en entorns reals.
