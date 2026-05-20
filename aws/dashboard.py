"""
Dashboard interactivo - AEInnova Predictive Maintenance
Ejecutar con:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
import tempfile
import os

# ─── Configuració de la pàgina ───────────────────────────────────────────────
st.set_page_config(
    page_title="AEInnova — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BUCKET   = "aeinnova-tfg-836321169819"
REGION   = "eu-west-1"
PALETTE  = {
    "High Amplitude Variability - holgura/impacto": "#e63946",
    "Mid-Frequency Energy - desalineacion":          "#f4a261",
    "Low-Frequency Energy - desbalance":             "#2a9d8f",
    "Distributed Energy - degradacion general":      "#457b9d",
    "Normal":                                        "#adb5bd",
}
SEV_COLORS = {"HIGH": "#e63946", "MEDIUM": "#f4a261", "LOW": "#2a9d8f", "NONE": "#adb5bd"}

# ─── Càrrega de dades ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    s3  = boto3.client("s3", region_name=REGION)

    def from_s3(key):
        tmp = tempfile.mktemp(suffix=".csv")
        s3.download_file(BUCKET, key, tmp)
        return pd.read_csv(tmp)

    predictions = from_s3("outputs/all_predictions.csv")
    features    = from_s3("processed/features.csv")

    df = pd.concat([features.reset_index(drop=True),
                    predictions.reset_index(drop=True)], axis=1)

    # Anomalies only
    df_anom = df[df["is_anomaly"] == True].copy()

    # Axis / mode com a categories llegibles
    df["axis_label"]  = df["axis"].map({1.0: "X", 2.0: "Y", 3.0: "Z"}).fillna(df["axis"].astype(str))
    df_anom["axis_label"] = df["axis_label"]

    return df, df_anom

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg", width=80)
    st.title("AEInnova")
    st.caption("Sistema de Detecció d'Anomalies")
    st.divider()
    st.markdown("**Filtra per:**")
    filter_severity = st.multiselect(
        "Nivell de Severitat", ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"]
    )
    filter_fault = st.multiselect(
        "Tipus de Fallo",
        ["High Amplitude Variability - holgura/impacto",
         "Mid-Frequency Energy - desalineacion",
         "Low-Frequency Energy - desbalance",
         "Distributed Energy - degradacion general"],
        default=["High Amplitude Variability - holgura/impacto",
                 "Mid-Frequency Energy - desalineacion",
                 "Low-Frequency Energy - desbalance",
                 "Distributed Energy - degradacion general"]
    )
    st.divider()
    st.caption("Font: AEInnova · Dataset NOD-0007")

# ─── Càrrega ─────────────────────────────────────────────────────────────────
with st.spinner("Carregant dades des de S3..."):
    df, df_anom = load_data()

df_filtered = df_anom[
    df_anom["severity_level"].isin(filter_severity) &
    df_anom["fault_label"].isin(filter_fault)
].copy()

# ─── Títol ───────────────────────────────────────────────────────────────────
st.title("⚙️ AEInnova — Predictive Maintenance Dashboard")
st.caption(f"Pipeline: Isolation Forest → K-Means → Gradient Boosting · Desplegat a AWS SageMaker")
st.divider()

# ─── KPI Cards ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total        = len(df)
n_anom       = len(df_anom)
n_filtered   = len(df_filtered)
pct          = 100 * n_anom / total
n_high       = (df_anom["severity_level"] == "HIGH").sum()

k1.metric("Total Registres",       f"{total:,}")
k2.metric("Anomalies Detectades",  f"{n_anom:,}", f"{pct:.1f}%")
k3.metric("Alertes (filtrades)",   f"{n_filtered:,}")
k4.metric("Severitat ALTA",        f"{n_high:,}", delta_color="inverse")
k5.metric("Accuracy CV",           "99.45%")

st.divider()

# ─── Fila 1: Distribucions principals ────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("Distribució per Tipus de Fallo")
    counts = df_filtered["fault_label"].value_counts().reset_index()
    counts.columns = ["fault_label", "count"]
    fig = px.pie(
        counts, values="count", names="fault_label",
        color="fault_label", color_discrete_map=PALETTE,
        hole=0.45
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, key="donut_fault")

with col2:
    st.subheader("Distribució per Severitat")
    sev_counts = df_filtered["severity_level"].value_counts().reset_index()
    sev_counts.columns = ["level", "count"]
    order = ["HIGH", "MEDIUM", "LOW"]
    sev_counts["level"] = pd.Categorical(sev_counts["level"], categories=order, ordered=True)
    sev_counts = sev_counts.sort_values("level")
    fig2 = px.bar(
        sev_counts, x="level", y="count", color="level",
        color_discrete_map=SEV_COLORS, text="count"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(showlegend=False, xaxis_title="Severitat", yaxis_title="Nombre",
                       margin=dict(t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True, key="bar_severity")

with col3:
    st.subheader("Resum")
    for label, color in PALETTE.items():
        if label == "Normal":
            continue
        n = (df_filtered["fault_label"] == label).sum()
        short = label.split(" - ")[-1].capitalize()
        st.markdown(f"**{short}**")
        st.progress(int(n / max(n_filtered, 1) * 100))
        st.caption(f"{n:,} alertes")

st.divider()

# ─── Fila 2: Per dispositiu i per eix ────────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.subheader("Anomalies per Dispositiu (dev_eui)")
    if "dev_eui" in df_filtered.columns:
        dev_counts = (
            df_filtered.groupby(["dev_eui", "fault_label"])
            .size().reset_index(name="count")
        )
        fig3 = px.bar(
            dev_counts, x="count", y="dev_eui", color="fault_label",
            color_discrete_map=PALETTE, orientation="h",
            barmode="stack"
        )
        fig3.update_layout(showlegend=False, yaxis_title="", xaxis_title="Nombre d'anomalies",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("dev_eui no disponible")

with col5:
    st.subheader("Anomalies per Eix de Mesura")
    if "axis_label" in df_filtered.columns:
        ax_counts = (
            df_filtered.groupby(["axis_label", "fault_label"])
            .size().reset_index(name="count")
        )
        fig4 = px.bar(
            ax_counts, x="axis_label", y="count", color="fault_label",
            color_discrete_map=PALETTE, barmode="group", text="count"
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(showlegend=True, xaxis_title="Eix", yaxis_title="Nombre",
                           legend_title="Tipus de Fallo", margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("axis no disponible")

st.divider()

# ─── Fila 3: Anàlisi espectral ────────────────────────────────────────────────
col6, col7 = st.columns(2)

with col6:
    st.subheader("Distribució del Score d'Anomalia")
    fig5 = px.histogram(
        df_anom, x="severity", color="fault_label",
        color_discrete_map=PALETTE, nbins=50,
        barmode="overlay", opacity=0.75
    )
    fig5.update_layout(xaxis_title="Anomaly Score", yaxis_title="Freqüència",
                       legend_title="Tipus", margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)

with col7:
    st.subheader("RMS vs Energia per Tipus de Fallo")
    if "rms" in df_anom.columns and "energy" in df_anom.columns:
        sample = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)
        fig6 = px.scatter(
            sample, x="rms", y="energy", color="fault_label",
            color_discrete_map=PALETTE, opacity=0.6,
            hover_data=["severity", "severity_level"]
        )
        fig6.update_layout(xaxis_title="RMS", yaxis_title="Energia Total",
                           legend_title="Tipus", margin=dict(t=10, b=10))
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Columnes RMS/energia no disponibles")

st.divider()

# ─── Fila 4: Característiques espectrals ─────────────────────────────────────
st.subheader("Distribució de Característiques Espectrals per Tipus de Fallo")

feat_cols = ["rms", "energy_low", "energy_mid", "energy_high",
             "dominant_frequency", "crest_factor"]
available = [c for c in feat_cols if c in df_filtered.columns]

if available:
    col_sel = st.selectbox("Selecciona característica:", available)
    fig7 = px.box(
        df_filtered, x="fault_label", y=col_sel,
        color="fault_label", color_discrete_map=PALETTE,
        points="outliers"
    )
    fig7.update_layout(showlegend=False, xaxis_title="", yaxis_title=col_sel,
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig7, use_container_width=True)

st.divider()

# ─── Fila 5: Anomalies per mode ───────────────────────────────────────────────
col8, col9 = st.columns(2)

with col8:
    st.subheader("Anomalies per Mode de Captació")
    if "mode" in df_filtered.columns:
        mode_counts = df_filtered.groupby("mode")["fault_label"].count().reset_index()
        mode_counts.columns = ["mode", "count"]
        mode_counts["mode"] = mode_counts["mode"].astype(str)
        fig8 = px.bar(mode_counts, x="mode", y="count", text="count",
                      color_discrete_sequence=["#457b9d"])
        fig8.update_traces(textposition="outside")
        fig8.update_layout(xaxis_title="Mode", yaxis_title="Nombre",
                           margin=dict(t=10, b=10))
        st.plotly_chart(fig8, use_container_width=True)

with col9:
    st.subheader("Heatmap: Dispositiu × Tipus de Fallo")
    if "dev_eui" in df_filtered.columns:
        pivot = (
            df_filtered.groupby(["dev_eui", "fault_label"])
            .size().unstack(fill_value=0)
        )
        fig9 = px.imshow(
            pivot, color_continuous_scale="Reds",
            labels=dict(x="Tipus de Fallo", y="Dispositiu", color="Nombre")
        )
        fig9.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig9, use_container_width=True)

st.divider()

# ─── Taula de les pitjors alertes ────────────────────────────────────────────
st.subheader("Top 20 Alertes per Severitat")
display_cols = [c for c in ["dev_eui", "axis_label", "mode", "fault_label",
                             "severity", "severity_level"] if c in df_filtered.columns]
top_alerts = df_filtered.sort_values("severity", ascending=False).head(20)

def color_severity(val):
    colors = {"HIGH": "background-color:#ffe0e0", "MEDIUM": "background-color:#fff3cd",
              "LOW": "background-color:#d4edda"}
    return colors.get(val, "")

st.dataframe(
    top_alerts[display_cols].style.applymap(color_severity, subset=["severity_level"]),
    use_container_width=True, hide_index=True
)

st.caption("AEInnova · UAB TFG 2025/26 · Adrià Muro Gómez")
