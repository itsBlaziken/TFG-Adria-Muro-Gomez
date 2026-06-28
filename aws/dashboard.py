# Dashboard professional de manteniment predictiu per AEInnova.
# Carrega les dades directament des de S3 i les visualitza amb Streamlit i Plotly.
# Executar amb: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import boto3
import tempfile
import os

st.set_page_config(
    page_title="AEInnova — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Page background ── */
.stApp { background-color: #f0f4f8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #1a3a5c 60%, #0d2644 100%);
    border-right: 1px solid #00a8e833;
}
[data-testid="stSidebar"] * { color: #e8f0fe !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background: #1e3d5a; border: 1px solid #00a8e855;
}
[data-testid="stSidebar"] hr { border-color: #00a8e833 !important; }

/* ── Top header band ── */
.aeinnova-header {
    background: linear-gradient(135deg, #0d1f35 0%, #1a3a5c 50%, #0d3b60 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0,168,232,0.15);
    border: 1px solid #00a8e820;
}
.aeinnova-header .brand { display: flex; align-items: center; gap: 18px; }
.aeinnova-header .brand-text h1 {
    margin: 0; color: #ffffff; font-size: 26px; font-weight: 700; letter-spacing: 0.5px;
}
.aeinnova-header .brand-text p {
    margin: 4px 0 0; color: #00a8e8; font-size: 13px; font-weight: 400; letter-spacing: 1px;
}
.aeinnova-header .badges { display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
.badge {
    background: #00a8e815; border: 1px solid #00a8e840; border-radius: 20px;
    padding: 4px 12px; font-size: 11px; color: #00c8ff; font-weight: 500;
    white-space: nowrap;
}

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 22px;
    border-left: 4px solid #00a8e8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 4px;
    height: 100%;
}
.kpi-card.alert { border-left-color: #e63946; }
.kpi-card.warning { border-left-color: #f4a261; }
.kpi-card.success { border-left-color: #2a9d8f; }
.kpi-card.info { border-left-color: #457b9d; }
.kpi-label {
    font-size: 11px; font-weight: 600; color: #6b7a8d;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px; font-weight: 700; color: #0d1f35; line-height: 1;
}
.kpi-delta {
    font-size: 12px; font-weight: 500; margin-top: 6px;
}
.kpi-delta.up { color: #e63946; }
.kpi-delta.down { color: #2a9d8f; }
.kpi-delta.neutral { color: #6b7a8d; }

/* ── Section headers ── */
.section-title {
    font-size: 15px; font-weight: 600; color: #0d1f35;
    border-bottom: 2px solid #00a8e830; padding-bottom: 8px;
    margin-bottom: 14px; letter-spacing: 0.3px;
}

/* ── Panel cards ── */
.panel {
    background: #ffffff; border-radius: 12px; padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 20px;
}

/* ── Alert table row colors ── */
.high-row td { background-color: #fff0f0 !important; }
.med-row  td { background-color: #fffbf0 !important; }
.low-row  td { background-color: #f0fff8 !important; }

/* ── Model info pill ── */
.model-pill {
    display: inline-block; background: #e8f4fd;
    border: 1px solid #00a8e840; border-radius: 20px;
    padding: 3px 10px; font-size: 11px; color: #1a3a5c;
    font-weight: 500; margin: 2px;
}

/* ── Footer ── */
.aeinnova-footer {
    background: linear-gradient(90deg, #0d1f35, #1a3a5c);
    border-radius: 10px; padding: 14px 28px;
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 30px;
}
.aeinnova-footer span { color: #6b96c2; font-size: 11px; }
.aeinnova-footer .dot { color: #00a8e8; margin: 0 6px; }

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 8px; }

/* ── Divider ── */
hr { border-color: #dde3ea !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
BUCKET = "aeinnova-tfg-836321169819"
REGION = "eu-west-1"

# Colors de severitat: vermell / taronja / verd
SEV_COLORS = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f", "NONE": "#adb5bd"}

# Colors dels tipus de fallo: completament diferents dels de severitat
_FAULT_COLOR_SEQ = ["#3a86ff", "#8338ec", "#ff006e", "#ffbe0b", "#0096c7", "#06d6a0"]

# Traducció de les etiquetes de fallo al català
FAULT_LABEL_CA = {
    "Type 0 - High Std Amplitude - holgura / impacto mecánico":
        "Tipus 0 - Alta Desv. Amplitud - folgança / impacte mecànic",
    "Type 1 - High Energy Mid-Frequency - desalineación":
        "Tipus 1 - Alta Energia Freqüència Mitja - desalineació",
    "Type 2 - High Energy Low-Frequency - desbalance mecánico":
        "Tipus 3 - Energia Multi-Banda - degradació general",
    "Type 3 - Balanced Multi-Band Energy - degradación general":
        "Tipus 2 - Alta Energia Alta Freqüència - modes operatius alt rang freqüencial",
    "Comportamiento normal": "Comportament normal",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#4a5568"),
    margin=dict(t=10, b=10, l=10, r=10),
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    s3 = boto3.client("s3", region_name=REGION)

    def from_s3(key):
        tmp = tempfile.mktemp(suffix=".csv")
        s3.download_file(BUCKET, key, tmp)
        return pd.read_csv(tmp)

    predictions = from_s3("outputs/all_predictions.csv")
    features    = from_s3("processed/features.csv")

    df = pd.concat([features.reset_index(drop=True),
                    predictions.reset_index(drop=True)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    # Convertir timestamp Unix-ms a datetime llegible
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)

    # Traduir etiquetes de fallo al català
    if "fault_label" in df.columns:
        df["fault_label"] = df["fault_label"].map(FAULT_LABEL_CA).fillna(df["fault_label"])

    df_anom = df[df["is_anomaly"] == True].copy()
    df["axis_label"]      = df["axis"].map({1.0: "X", 2.0: "Y", 3.0: "Z"}).fillna(df["axis"].astype(str))
    df_anom["axis_label"] = df.loc[df_anom.index, "axis_label"]

    # Severitat basada en crest_factor (o rms) per percentils dins les anomalies
    # Severitat basada en crest_factor: ~20% HIGH, ~30% MEDIUM, ~50% LOW
    _metric = "crest_factor" if "crest_factor" in df_anom.columns else "rms"
    if _metric in df_anom.columns:
        q50 = df_anom[_metric].quantile(0.50)
        q80 = df_anom[_metric].quantile(0.80)
        def _sev(v):
            if v >= q80:   return "HIGH"
            elif v >= q50: return "MEDIUM"
            else:          return "LOW"
        df_anom["severity_level"] = df_anom[_metric].apply(_sev)

    return df, df_anom

# ── Carregar dades ABANS del sidebar per usar valors reals als filtres ─────────
with st.spinner("Carregant dades des de S3 · eu-west-1…"):
    df, df_anom = load_data()

# PALETTE dinàmica basada en els fault_labels reals del dataset
_fault_labels_sorted = sorted(df_anom["fault_label"].dropna().unique())
PALETTE = {label: _FAULT_COLOR_SEQ[i % len(_FAULT_COLOR_SEQ)]
           for i, label in enumerate(_fault_labels_sorted)}
PALETTE["Normal"] = "#adb5bd"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="26" cy="26" r="25" stroke="#00a8e8" stroke-width="1.5" fill="#0d2644"/>
            <path d="M26 10 L26 18 M26 34 L26 42 M10 26 L18 26 M34 26 L42 26"
                  stroke="#00a8e8" stroke-width="2" stroke-linecap="round"/>
            <circle cx="26" cy="26" r="8" fill="none" stroke="#00a8e8" stroke-width="1.5"/>
            <circle cx="26" cy="26" r="3" fill="#00a8e8"/>
        </svg>
        <div style="font-size:20px; font-weight:700; color:#ffffff; margin-top:8px; letter-spacing:1px;">
            AEInnova
        </div>
        <div style="font-size:10px; color:#00a8e8; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">
            Predictive Maintenance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; font-weight:600; letter-spacing:1px; color:#8ab0d0; text-transform:uppercase; margin-bottom:8px;">Filtres</div>', unsafe_allow_html=True)

    filter_severity = st.multiselect(
        "Nivell de Severitat", ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"]
    )
    filter_fault = st.multiselect(
        "Tipus de Fallo",
        _fault_labels_sorted,
        default=_fault_labels_sorted,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px; color:#4d7a9e;">AEInnova · NOD-0007 · UAB 2025/26</div>', unsafe_allow_html=True)

# ── Aplicar filtres ────────────────────────────────────────────────────────────
df_filtered = df_anom[
    df_anom["severity_level"].isin(filter_severity) &
    df_anom["fault_label"].isin(filter_fault)
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aeinnova-header">
    <div class="brand">
        <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="28" cy="28" r="27" stroke="#00a8e8" stroke-width="1.5" fill="#0d2644"/>
            <path d="M28 10 L28 20 M28 36 L28 46 M10 28 L20 28 M36 28 L46 28"
                  stroke="#00a8e8" stroke-width="2.5" stroke-linecap="round"/>
            <path d="M17 17 L23 23 M33 33 L39 39 M17 39 L23 33 M33 23 L39 17"
                  stroke="#00a8e860" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="28" cy="28" r="9" fill="none" stroke="#00a8e8" stroke-width="1.5"/>
            <circle cx="28" cy="28" r="4" fill="#00a8e8"/>
        </svg>
        <div class="brand-text">
            <h1>AEInnova &mdash; Predictive Maintenance</h1>
            <p>SISTEMA DE DETECCIÓ D'ANOMALIES &nbsp;·&nbsp; MONITORITZACIÓ VIBRATÒRIA EN TEMPS REAL</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
total        = len(df)
n_anom       = len(df_anom)
n_filtered   = len(df_filtered)
pct          = 100 * n_anom / total
n_high       = (df_anom["severity_level"] == "HIGH").sum()
n_medium     = (df_anom["severity_level"] == "MEDIUM").sum()
n_low        = (df_anom["severity_level"] == "LOW").sum()
n_devices     = df_anom["dev_eui"].nunique() if "dev_eui" in df_anom.columns else 0
n_fault_types = df_anom["fault_label"].nunique()
pct_normal    = 100 - pct

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card info">
        <div class="kpi-label">Total Registres</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-delta neutral">&#9632; Mostres vibratòries</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card warning">
        <div class="kpi-label">Anomalies Detectades</div>
        <div class="kpi-value">{n_anom:,}</div>
        <div class="kpi-delta up">&#9650; {pct:.1f}% del total</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card alert">
        <div class="kpi-label">Severitat ALTA / MITJA / BAIXA</div>
        <div class="kpi-value" style="font-size:22px;">
            <span style="color:#e63946;">{n_high:,}</span>
            <span style="font-size:16px; color:#adb5bd;"> / </span>
            <span style="color:#f4a261;">{n_medium:,}</span>
            <span style="font-size:16px; color:#adb5bd;"> / </span>
            <span style="color:#2a9d8f;">{n_low:,}</span>
        </div>
        <div class="kpi-delta up">&#9888; HIGH &nbsp;·&nbsp; MEDIUM &nbsp;·&nbsp; LOW</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card info">
        <div class="kpi-label">Dispositius Afectats</div>
        <div class="kpi-value">{n_devices}</div>
        <div class="kpi-delta neutral">&#9632; {n_fault_types} tipus de fallo</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card success">
        <div class="kpi-label">Registres Normals</div>
        <div class="kpi-value">{total - n_anom:,}</div>
        <div class="kpi-delta down">&#9660; {pct_normal:.1f}% sense anomalia</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

# ── Row 1: Donut + Severity bar + Gauge de salut ──────────────────────────────
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribució per Tipus de Fallo</div>', unsafe_allow_html=True)
    counts = df_filtered["fault_label"].value_counts().reset_index()
    counts.columns = ["fault_label", "count"]
    fig = px.pie(
        counts, values="count", names="fault_label",
        color="fault_label", color_discrete_map=PALETTE, hole=0.48
    )
    fig.update_traces(
        textposition="inside", textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value} anomalies<extra></extra>"
    )
    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
                    font=dict(size=10)),
        height=360,
        annotations=[dict(text=f"<b>{n_filtered}</b><br><span style='font-size:10px'>alertes</span>",
                          showarrow=False, font=dict(size=16, color="#0d1f35"), x=0.5, y=0.5)]
    )
    st.plotly_chart(fig, use_container_width=True, key="donut_fault")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribució per Nivell de Severitat</div>', unsafe_allow_html=True)
    sev_counts = df_filtered["severity_level"].value_counts().reset_index()
    sev_counts.columns = ["level", "count"]
    order = ["HIGH", "MEDIUM", "LOW"]
    sev_counts["level"] = pd.Categorical(sev_counts["level"], categories=order, ordered=True)
    sev_counts = sev_counts.sort_values("level")
    fig2 = px.bar(
        sev_counts, x="level", y="count", color="level",
        color_discrete_map=SEV_COLORS, text="count"
    )
    fig2.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} anomalies<extra></extra>"
    )
    fig2.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        xaxis=dict(title="Severitat", showgrid=False),
        yaxis=dict(title="Nombre d'anomalies", gridcolor="#edf0f5",
                   rangemode="nonnegative"),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True, key="bar_severity")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Severitat per Tipus de Fallo</div>', unsafe_allow_html=True)
    sev_fault = (
        df_filtered.groupby(["fault_label", "severity_level"])
        .size().reset_index(name="count")
    )
    order_sev = ["HIGH", "MEDIUM", "LOW"]
    sev_fault["severity_level"] = pd.Categorical(
        sev_fault["severity_level"], categories=order_sev, ordered=True
    )
    # Etiquetes curtes per l'eix Y (evitar text llarg solapat)
    short_labels = {
        lbl: lbl.split(" - ")[-1].capitalize() if " - " in lbl else lbl
        for lbl in sev_fault["fault_label"].unique()
    }
    sev_fault["fault_short"] = sev_fault["fault_label"].map(short_labels)

    fig_sf = px.bar(
        sev_fault.sort_values("severity_level"),
        x="fault_short", y="count", color="severity_level",
        color_discrete_map=SEV_COLORS, barmode="stack",
        text="count",
    )
    fig_sf.update_traces(textposition="inside", textfont_size=11)
    _layout_sf = {**CHART_LAYOUT, "margin": dict(t=10, b=130, l=10, r=10)}
    fig_sf.update_layout(
        **_layout_sf,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.62,
                    xanchor="center", x=0.5, font=dict(size=11), title=""),
        xaxis=dict(title="", showgrid=False, tickangle=-30,
                   tickfont=dict(size=11)),
        yaxis=dict(title="Nombre d'anomalies", gridcolor="#edf0f5",
                   rangemode="nonnegative"),
        height=320,
    )
    st.plotly_chart(fig_sf, use_container_width=True, key="bar_sev_fault")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2: Boxplot interactiu ─────────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Distribució de Característiques Espectrals per Tipus de Fallo</div>', unsafe_allow_html=True)

feat_cols  = ["rms", "energy_low", "energy_mid", "energy_high", "dominant_frequency", "crest_factor"]
available  = [c for c in feat_cols if c in df_filtered.columns]
feat_names = {
    "rms": "RMS (Arrel Quadràtica Mitjana)",
    "energy_low": "Energia Banda Baixa (<100 Hz)",
    "energy_mid": "Energia Banda Mitja (100–500 Hz)",
    "energy_high": "Energia Banda Alta (>500 Hz)",
    "dominant_frequency": "Freqüència Dominant (Hz)",
    "crest_factor": "Crest Factor",
}

if available:
    bcol1, bcol2 = st.columns([1, 4])
    with bcol1:
        col_sel = st.selectbox(
            "Característica:", available,
            format_func=lambda x: feat_names.get(x, x)
        )
    with bcol2:
        fig7 = px.box(
            df_filtered, x="fault_label", y=col_sel,
            color="fault_label", color_discrete_map=PALETTE, points="outliers"
        )
        fig7.update_layout(
            **CHART_LAYOUT,
            showlegend=False,
            xaxis=dict(title="", showgrid=False, tickangle=-12, tickfont=dict(size=11)),
            yaxis=dict(title=feat_names.get(col_sel, col_sel), gridcolor="#edf0f5"),
            height=340,
        )
        st.plotly_chart(fig7, use_container_width=True, key="boxplot_feat")

st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3: Heatmap ample ────────────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Heatmap Anomalies: Dispositiu × Tipus de Fallo</div>', unsafe_allow_html=True)
if "dev_eui" in df_filtered.columns:
    pivot = (
        df_filtered.groupby(["dev_eui", "fault_label"])
        .size().unstack(fill_value=0)
    )
    # Escurçar noms de columnes per llegibilitat
    pivot.columns = [c.split(" - ")[-1].capitalize() if " - " in c else c
                     for c in pivot.columns]
    fig9 = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#f0f4f8"], [0.3, "#7ec8e3"], [0.7, "#00a8e8"], [1, "#e63946"]],
        labels=dict(x="Tipus de Fallo", y="Dispositiu (dev_eui)", color="Anomalies"),
        text_auto=True,
    )
    fig9.update_traces(textfont=dict(size=14))
    _layout_hm = {**CHART_LAYOUT, "margin": dict(t=10, b=110, l=10, r=80)}
    fig9.update_layout(
        **_layout_hm,
        height=400,
        xaxis=dict(tickangle=-35, tickfont=dict(size=12), side="bottom"),
        yaxis=dict(tickfont=dict(size=12)),
        coloraxis_colorbar=dict(thickness=15, len=0.85, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig9, use_container_width=True, key="heatmap_device")
else:
    st.info("Columna dev_eui no disponible")
st.markdown('</div>', unsafe_allow_html=True)

# ── Row 4: Taula alertes recents ──────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Últimes 20 Alertes Detectades</div>', unsafe_allow_html=True)

display_cols = [c for c in ["timestamp", "dev_eui", "axis_label", "mode",
                             "fault_label", "severity_level"] if c in df_filtered.columns]

# Ordenar per timestamp si existeix, sinó per ordre d'entrada
if "timestamp" in df_filtered.columns:
    top_alerts = df_filtered.sort_values("timestamp", ascending=False).head(20)
else:
    top_alerts = df_filtered.head(20)

col_rename = {
    "timestamp":    "Timestamp",
    "dev_eui":      "Dispositiu",
    "axis_label":   "Eix",
    "mode":         "Mode",
    "fault_label":  "Tipus de Fallo",
    "severity_level": "Severitat",
}

def color_severity(val):
    palette = {
        "HIGH":   "background-color:#ffe0e0; color:#c0392b; font-weight:600;",
        "MEDIUM": "background-color:#fff8e0; color:#d68910; font-weight:600;",
        "LOW":    "background-color:#e8f8f0; color:#1e8449; font-weight:600;",
    }
    return palette.get(val, "")

fmt = {}
if "mode" in top_alerts.columns:
    top_alerts = top_alerts.copy()
    top_alerts["mode"] = top_alerts["mode"].apply(
        lambda v: str(int(v)) if pd.notna(v) else ""
    )

styled = (
    top_alerts[display_cols]
    .rename(columns=col_rename)
    .style
    .applymap(color_severity, subset=["Severitat"])
)
st.dataframe(styled, use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aeinnova-footer">
    <span>&#169; 2025 AEInnova &mdash; Sistema de Manteniment Predictiu</span>
    <span>
        Adrià Muro Gómez
        <span class="dot">·</span>
        Treball de Fi de Grau &mdash; Universitat Autònoma de Barcelona
        <span class="dot">·</span>
        2025/26
    </span>
</div>
""", unsafe_allow_html=True)
