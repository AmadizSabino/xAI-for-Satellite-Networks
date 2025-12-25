
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import datetime as dt
from functools import lru_cache
import requests
import json
import textwrap

# ------------------------------
# Paths and global constants
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

FEEDBACK_CSV = DATA_DIR / "dashboard_feedback.csv"
ALERT_HISTORY_CSV = DATA_DIR / "dashboard_alert_history.csv"

if not DATA_DIR.exists():
    st.error(f"Data directory not found: {DATA_DIR}")
    st.stop()


# ==========================================================
# Github repo
# ==========================================================

CODE_REPO_URL = "https://github.com/AmadizSabino/xAI-for-Satellite-Networks"
THESIS_URL = "https://your-thesis-link"

# ==========================================================
# File existence guards 
# ==========================================================
if not FEEDBACK_CSV.exists():
    st.warning(f"Feedback file not found: {FEEDBACK_CSV}")

if not ALERT_HISTORY_CSV.exists():
    st.warning(f"Alert history file not found: {ALERT_HISTORY_CSV}")

# ==========================================================
# Data loading (guaranteed initialization)
# ==========================================================
if FEEDBACK_CSV.exists():
    feedback_df = pd.read_csv(FEEDBACK_CSV)
else:
    feedback_df = pd.DataFrame(columns=["feedback"])

# ------------------------------
# Translation engine (local dictionaries)
# ------------------------------

LANG_CODES = {
    "English": "en",
    "Português": "pt",
    "Français": "fr",
    "Español": "es",
}

LOCAL_TRANSLATIONS = {
    "pt": {
        "Anomaly Dashboard": "Painel de Anomalias",
        "Language": "Idioma",
        "Overview": "Visão geral",
        "Signal Loss": "Perda de Sinal",
        "Jamming / Interference": "Jamming / Interferência",
        "SLA Breach": "Violação de SLA",
        "Beam Handover": "Mudança de Feixe",
        "Space Weather": "Clima Espacial",
        "Risk-aware Capacity Advisor": "Conselheiro de Capacidade Sensível ao Risco",
        "Stress Index & Joint Risk Radar": "Índice de Stress e Radar de Risco Conjunto",
        "Alert Analytics (Thesis Mode)": "Analítica de Alertas (Modo Tese)",
        "Feedback Analytics (Thesis Mode)": "Analítica de Feedback (Modo Tese)",
        "Academic mode (show literature links)": "Modo académico (mostrar referências)",
        "Time window": "Janela temporal",
        "Last 24h": "Últimas 24h",
        "Last 7 days": "Últimos 7 dias",
        "Full dataset": "Conjunto de dados completo",
        "Acknowledge": "Reconhecer",
        "Alerts and suggested actions – recent high-risk windows":
            "Alertas e ações sugeridas – janelas recentes de alto risco",
        "Current space weather (live NOAA Kp index)":
            "Clima espacial atual (índice Kp da NOAA em tempo real)",
        "Earth in real time (NOAA) – external view":
            "Terra em tempo real (NOAA) – vista externa",
        "Open NOAA Earth in Real Time": "Abrir NOAA Earth in Real Time",
    },
    "fr": {
        "Anomaly Dashboard": "Tableau de bord des anomalies",
        "Overview": "Vue d’ensemble",
        "Signal Loss": "Perte de signal",
        "Jamming / Interference": "Brouillage / Interférences",
        "SLA Breach": "Violation de SLA",
        "Beam Handover": "Changement de faisceau",
        "Space Weather": "Météo spatiale",
        "Risk-aware Capacity Advisor": "Conseiller capacité sensible au risque",
        "Stress Index & Joint Risk Radar":
            "Indice de stress et radar de risque conjoint",
        "Alert Analytics (Thesis Mode)":
            "Analyse des alertes (mode thèse)",
        "Feedback Analytics (Thesis Mode)":
            "Analyse des retours (mode thèse)",
        "Acknowledge": "Accuser réception",
    },
    "es": {
        "Anomaly Dashboard": "Panel de anomalías",
        "Overview": "Resumen",
        "Signal Loss": "Pérdida de señal",
        "Jamming / Interference": "Interferencia / Jamming",
        "SLA Breach": "Incumplimiento de SLA",
        "Beam Handover": "Transferencia de haz",
        "Space Weather": "Clima espacial",
        "Risk-aware Capacity Advisor": "Asesor de capacidad consciente del riesgo",
        "Stress Index & Joint Risk Radar":
            "Índice de estrés y radar de riesgo conjunto",
        "Alert Analytics (Thesis Mode)":
            "Analítica de alertas (modo tesis)",
        "Feedback Analytics (Thesis Mode)":
            "Analítica de feedback (modo tesis)",
        "Acknowledge": "Reconocer",
    },
}

@lru_cache(maxsize=4096)
def tr(text: str) -> str:
    """Translate using simple local dictionaries; fall back to English text."""
    lang_code = st.session_state.get("lang_code", "en")
    if lang_code == "en":
        return text
    mapping = LOCAL_TRANSLATIONS.get(lang_code, {})
    return mapping.get(text, text)

# ------------------------------
# Literature notes (Academic mode)
# ------------------------------

LIT_NOTES = {
    "overview": (
        "Human-centred AI and operator-in-the-loop tooling inspired by "
        "Amershi et al. (2019) and Tjoa & Guan (2020)."
    ),
    "signal_loss": (
        "SHAP-based telemetry explanations related to Cuéllar et al. (2024) "
        "on satellite telemetry anomaly explanation."
    ),
    "jamming": (
        "Interference detection and feature relevance inspired by "
        "Li (2023) and Tritscher (2023) on RF anomaly detection."
    ),
    "sla": (
        "Early-warning SLA risk modelling aligned with service-quality "
        "monitoring literature in satellite networks."
    ),
    "handover": (
        "Beam handover quality monitoring connects to mobility QoS work "
        "in satellite and 5G networks."
    ),
    "spaceweather": (
        "Space-weather risk combined with station-keeping manoeuvres uses "
        "NOAA indices as in Franco de la Peña et al. (2025)."
    ),
    "capacity": (
        "Risk-aware capacity advisor and risk_index introduced in "
        "Sabino (2025) as a way to merge utilisation, demand forecasts "
        "and explainability signals into a single operational score."
    ),
    "stress": (
        "Stress index and joint risk radar proposed by Sabino et al. (2025), "
        "inspired by system-level explanations in Iino et al. (2024)."
    ),
    "alerts": (
        "Alert analytics used to study alert fatigue and human factors "
        "following Tjoa & Guan (2020)."
    ),
    "feedback": (
        "Feedback analytics supports formative evaluation and usability "
        "assessment in line with human-centred XAI guidelines "
        "by Amershi et al. (2019)."
    ),
}

def lit_expander(key: str):
    if not st.session_state.get("academic_mode", False):
        return
    note = LIT_NOTES.get(key)
    if note:
        with st.expander("Literature context", expanded=False):
            st.caption(note)

# ------------------------------
# IO helpers
# ------------------------------

def load_csv(relative_paths, parse_dates=None):
    paths = relative_paths if isinstance(relative_paths, (list, tuple)) else [relative_paths]
    for rel in paths:
        p = BASE_DIR / rel
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=parse_dates)
            except Exception:
                return None
    return None

def load_image_path(relative_paths):
    paths = relative_paths if isinstance(relative_paths, (list, tuple)) else [relative_paths]
    for rel in paths:
        p = BASE_DIR / rel
        if p.exists():
            return str(p)
    return None

def append_feedback(row: dict):
    try:
        if FEEDBACK_CSV.exists():
            existing = pd.read_csv(FEEDBACK_CSV)
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.to_csv(FEEDBACK_CSV, index=False)
        else:
            pd.DataFrame([row]).to_csv(FEEDBACK_CSV, index=False)
    except Exception:
        st.warning("Could not persist feedback to CSV in this environment.")

def append_alert_history(rows: list):
    if not rows:
        return
    try:
        new_df = pd.DataFrame(rows)
        if ALERT_HISTORY_CSV.exists():
            old = pd.read_csv(ALERT_HISTORY_CSV)
            merged = pd.concat([old, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_csv(ALERT_HISTORY_CSV, index=False)
    except Exception:
        st.warning("Could not persist alert history to CSV in this environment.")

def load_shap_matrix(relative_path):
    df = load_csv(relative_path)
    if df is None or df.empty:        return None, None, None
    first_col = df.columns[0].lower()
    if first_col.startswith("unnamed"):
        df = df.set_index(df.columns[0])
    feature_names = df.index.tolist()
    time_labels = df.columns.tolist()
    return df, feature_names, time_labels



# ------------------------------
# Alerts helpers
# ------------------------------
def bootstrap_median_ci(values, n_boot=2000, ci=0.95, seed=42):
    """
    Non-parametric bootstrap CI for the median.
    Suitable for skewed time-to-ack distributions.
    """
    vals = pd.Series(values).dropna().astype(float).values
    if len(vals) < 5:
        return None

    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boots[i] = np.median(sample)

    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return lo, hi




# ------------------------------
# Time window helpers
# ------------------------------

def get_time_window_hours():
    options_display = [tr("Last 24h"), tr("Last 7 days"), tr("Full dataset")]
    current = st.session_state.get("time_window_display", options_display[-1])
    try:
        idx = options_display.index(current)
    except ValueError:
        return None
    if idx == 0:
        return 24
    if idx == 1:
        return 7 * 24
    return None

def apply_time_filter(df, time_col):
    hours = get_time_window_hours()
    if df is None or hours is None or time_col not in df.columns:
        return df
    latest_time = df[time_col].max()
    if pd.isna(latest_time):
        return df
    window_start = latest_time - pd.Timedelta(hours=hours)
    return df[df[time_col].between(window_start, latest_time)].copy()

# ------------------------------
# Thesis branding
# ------------------------------

def render_thesis_header():
    st.markdown(
        """
        <div style="padding:0.4rem 0 0.6rem 0; font-size:0.85rem; opacity:0.90;">
          <b>University of Hull – MSc Artificial Intelligence</b><br>
          Explainable AI for Satellite Networks – XAI Prototype Dashboard<br>
          Author: <b>Amadiz Sabino</b> · Organisation: SES · Academic year: 2025
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_thesis_footer():
    st.markdown(
        """
        <hr style="margin-top:2rem; margin-bottom:0.4rem;" />
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Prototype developed as part of the MSc AI thesis to evaluate "
        "Explainable AI techniques for anomaly detection in satellite networks. "
        "For methodology and evaluation details, please refer to the written dissertation."
    )

def render_academic_banner():
    if not st.session_state.get("academic_mode", False):
        return
    st.markdown(
        """
        <div style="
            background-color:#1d4ed8;
            color:white;
            padding:0.6rem 1.0rem;
            border-radius:0 0 0.75rem 0.75rem;
            font-size:0.85rem;
            margin-bottom:1.0rem;">
          <b>ACADEMIC MODE ENABLED</b><br/>
          Explanations include references to the literature review (Cuéllar 2024, Iino 2024,
          Li 2023, Tritscher 2023, Franco de la Peña 2025, Sabino 2025).
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# SHAP helper
# ------------------------------

def add_shap_hover(fig, x_label="time", y_label="feature", context_note=None):
    hover = f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<br>SHAP=%{{z:.3f}}"
    if st.session_state.get("academic_mode", False) and context_note:
        hover += f"<br><br>{context_note}"
    hover += "<extra></extra>"
    fig.update_traces(hovertemplate=hover)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ------------------------------
# Alerts rendering + 3D button CSS
# ------------------------------

def render_alerts(df, id_col, time_col, severity_col, title, usecase_key, max_rows=3):
    st.markdown(f"### {title}")

    #now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    now_utc = pd.Timestamp.now(tz="UTC")
    acked_at_utc = now_utc.isoformat()

    # 3D-like style for Streamlit buttons (including Acknowledge)
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button {
          background: linear-gradient(180deg,#22c55e,#16a34a);
          border: none;
          border-radius: 999px;
          padding: 0.25rem 0.9rem;
          color: white;
          font-size: 0.8rem;
          box-shadow: 0 3px 0 #15803d;
        }
        div[data-testid="stButton"] > button:active {
          box-shadow: 0 1px 0 #15803d;
          transform: translateY(2px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "acked_alerts" not in st.session_state:
        st.session_state["acked_alerts"] = []

    if "logged_alert_keys" not in st.session_state:
        st.session_state["logged_alert_keys"] = {}

    acked = set(st.session_state["acked_alerts"])
    logged = st.session_state["logged_alert_keys"]

    if df is None or df.empty:
        # Demo fallback
        demo = pd.DataFrame(
            {
                id_col: [f"ALERT-{usecase_key.upper()}-{i+1}" for i in range(3)],
                time_col: pd.date_range("2021-11-01", periods=3, freq="H", tz="UTC"),
                severity_col: ["high", "medium", "medium"],
            }
        )
        df_to_show = demo
    else:
        df_to_show = df.sort_values(time_col, ascending=False).head(max_rows)

    def make_key(row):
        return f"{usecase_key}:{row[id_col]}:{row[time_col]}"

    mask = []
    for _, r in df_to_show.iterrows():
        if make_key(r) not in acked:
            mask.append(True)
        else:
            mask.append(False)
    df_to_show = df_to_show[mask]

    if df_to_show.empty:
        st.success("No anomalies in this window – system healthy.")
        return

    rows_to_persist = []

    for i, (_, row) in enumerate(df_to_show.iterrows()):
        key = make_key(row)
        sev = str(row[severity_col]).lower()

        # If this alert has never been logged, log a non-acked entry
        if key not in logged:
            rows_to_persist.append(
                {
                    "usecase": usecase_key,
                    "alert_id": row[id_col],
                    "time_center": row[time_col],
                    "severity": sev,
                    "acked_at_utc": None,
                }
            )
            logged[key] = False  # seen but not acknowledged yet

        badge_color = "#f97316"
        if sev == "high":
            badge_color = "#dc2626"
        elif sev == "medium":
            badge_color = "#fb923c"
        elif sev == "low":
            badge_color = "#16a34a"

        card_bg = "#ffffff"
        border_color = "#e5e7eb"

        st.markdown(
            f"""
            <div style="
                border: 1px solid {border_color};
                border-radius: 0.6rem;
                padding: 0.75rem 0.9rem;
                margin-bottom: 0.6rem;
                background-color: {card_bg};
                box-shadow: 0 1px 2px rgba(15,23,42,0.08);
            ">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-weight:600;">{row[id_col]}</div>
                <span style="
                    font-size:0.75rem;
                    padding: 0.15rem 0.5rem;
                    border-radius: 999px;
                    background-color:{badge_color};
                    color:white;
                    text-transform:uppercase;
                    letter-spacing:0.04em;
                ">{sev}</span>
              </div>
              <div style="font-size:0.8rem; margin-top:0.15rem; opacity:0.8;">
                📡 Time: {row[time_col]}
              </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button(tr("Acknowledge"), key=f"{usecase_key}_ack_{i}"):
                if key not in acked:
                    st.session_state["acked_alerts"].append(key)
                    # Log an acknowledged entry
                    rows_to_persist.append(
                        {
                            "usecase": usecase_key,
                            "alert_id": row[id_col],
                            "time_center": row[time_col],
                            "severity": sev,
                            "acked_at_utc": pd.Timestamp.utcnow().isoformat(),
                        }
                    )
                    logged[key] = True
        with col_b:
            st.caption(
                "Typical action: open NOC ticket, attach supporting evidence, "
                "and notify the on-call engineer."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if rows_to_persist:
        append_alert_history(rows_to_persist)

# ------------------------------
# Page config + sidebar
# ------------------------------

st.set_page_config(page_title="Satellite Anomaly Dashboard", layout="wide")

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title(tr("Anomaly Dashboard"))

academic_mode = st.sidebar.checkbox(tr("Academic mode (show literature links)"), value=True)
st.session_state["academic_mode"] = academic_mode

lang_label = st.sidebar.selectbox("Language", list(LANG_CODES.keys()), index=0)
st.session_state["lang_code"] = LANG_CODES[lang_label]

view = st.sidebar.radio(
    "Select a view",
    [
        tr("Overview"),
        tr("Signal Loss"),
        tr("Jamming / Interference"),
        tr("SLA Breach"),
        tr("Beam Handover"),
        tr("Space Weather"),
        tr("Risk-aware Capacity Advisor"),
        tr("Stress Index & Joint Risk Radar"),
        tr("Alert Analytics (Thesis Mode)"),
        tr("Feedback Analytics (Thesis Mode)"),
    ],
)

with st.sidebar.expander("Demo filters", expanded=True):
    st.caption("In a live system these would filter the underlying data.")

satellite = st.sidebar.selectbox(
    "Satellite (aggregated data)", ["M003", "M008", "M015", "M017", "ALL"]
)

time_window_options = [tr("Last 24h"), tr("Last 7 days"), tr("Full dataset")]
time_window_display = st.sidebar.selectbox("Time window", time_window_options, index=2)
st.session_state["time_window_display"] = time_window_display

# ------------------------------
# Overview page
# ------------------------------

def page_overview():
    render_academic_banner()
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.title("Explainable AI for Satellite Networks Anomaly Detection")
        render_thesis_header()

        st.markdown("---")
        st.subheader("Purpose of this prototype")
        st.markdown(
            """
            Monitoring based on ground telemetry, space weather and SLA metrics.
            The goal is to give the Network Operations Center early insight into
            issues that affect availability and customer experience.
            """
        )

        st.markdown("---")
        st.subheader("What this prototype does")
        st.markdown(
            """
            - Tracks a set of anomaly use cases: signal loss, jamming or interference,
              SLA breach, beam handover issues, space weather maneuver risk,
              capacity pressure and a combined stress index.
            - Uses historical SES data to learn what healthy behaviour looks like,
              then scores new windows for risk.
            - Uses explainable AI (mainly SHAP heatmaps) so operators can see **why**
              a window is flagged.
            """
        )

        st.markdown("---")
        st.subheader("High level anomaly detection performance (test windows)")

        with st.expander("How to read these metrics", expanded=False):
            st.markdown(
                """
                - **PR-AUC (Precision–Recall area)** – how well the model lifts true
                  anomalies above noise. This is the main metric for rare events such
                  as signal loss or SLA breach.
                - **ROC-AUC** – overall ability to separate normal vs anomalous windows
                  across thresholds. Can look optimistic when anomalies are very rare.
                - **Event precision / recall** – precision asks *of the alerts raised,
                  how many were real?* Recall asks *of all real events, how many did we
                  catch?*  In this prototype the models are tuned towards **high precision** so
                  that operators can trust an alert, even if that means some events are
                  missed (moderate recall) to avoid flooding the NOC with false positives.
                """
            )

        metrics = [
            ("Signal Loss model", 0.72, 0.83, 0.93, 0.13),
            ("SLA early warning", 0.25, 0.86, 1.00, 0.03),
            ("Beam Handover anomalies", 0.022, 0.71, 0.50, 0.012),
        ]

        interpretations = {
            "Signal Loss model": (
                "The model is strong at prioritising true signal-loss events over noise. "
                "Most alerts are real (high precision), but it currently only catches a "
                "subset of all events (moderate recall). It is designed as a conservative "
                "early-warning signal."
            ),
            "SLA early warning": (
                "The model can identify some windows that are at risk of SLA degradation. "
                "Alerts are very rare but almost always correct (precision close to 1.0), "
                "so it behaves as a highly conservative early-warning indicator rather "
                "than a complete SLA monitor."
            ),
            "Beam Handover anomalies": (
                "Performance is weaker here, which is expected given the rarity and "
                "complexity of handover issues. The model can surface some interesting "
                "cases but still misses most true problems and generates some false "
                "alerts. This use case is marked as prototype / for further tuning."
            ),
        }

        metric_cols = st.columns(len(metrics))
        for col, (name, pr, roc, p, r) in zip(metric_cols, metrics):
            with col:
                st.markdown(f"**{name}**")
                st.metric("PR-AUC", f"{pr:.3f}")
                st.metric("ROC-AUC", f"{roc:.3f}")
                st.caption(f"Event precision / recall: **{p:.2f} / {r:.2f}**")
                with st.expander("Plain-English summary", expanded=False):
                    st.write(interpretations[name])


        st.markdown("---")
        st.markdown("### Radar view of trade-offs between models")

        radar_df = pd.DataFrame(
            {
                "metric": ["precision", "recall", "explainability", "impact"],
                "Signal Loss": [0.93, 0.13, 0.9, 0.8],
                "SLA early warning": [1.0, 0.03, 0.7, 0.75],
                "Beam Handover": [0.5, 0.012, 0.6, 0.4],
            }
        )
        radar_melted = radar_df.melt(id_vars="metric", var_name="model", value_name="score")
        radar_melted = pd.concat(
            [radar_melted, radar_melted[radar_melted["metric"] == "precision"]],
            ignore_index=True,
        )

        fig_radar = px.line_polar(
            radar_melted,
            r="score",
            theta="metric",
            color="model",
            line_close=True,
            title="Trade-offs between precision, recall, explainability and impact (demo)",
        )
        fig_radar.update_traces(fill="toself")
        fig_radar.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown(
            """
            This radar view makes the trade-offs between performance, interpretability
            and impact explicit, supporting the research questions on realistic,
            human-centred deployment scenarios.
            """
        )

        lit_expander("overview")


        st.markdown("---")
        st.subheader("Research questions and dashboard mapping")

        rq_rows = [
            ("RQ1 – Can XAI make anomaly alerts more interpretable for SES operators?",
             "Signal Loss, Jamming, SLA, Handover, Space Weather SHAP views."),
            ("RQ2 – Can system-level views help avoid alert fatigue?",
             "Stress Index & Joint Risk Radar, Alert Analytics pages."),
            ("RQ3 – How do operators perceive usefulness and trust in the XAI outputs?",
             "Overview feedback form + Feedback Analytics page."),
            ("RQ4 – How do different models trade off precision, recall and interpretability?",
             "Model metrics and radar view on this page."),
        ]
        rq_df = pd.DataFrame(rq_rows, columns=["Research question", "Where to look in the dashboard"])
        st.table(rq_df)


        st.markdown("---")
        st.subheader("Prototype feedback (for thesis user study)")

        with st.expander("Leave quick feedback about this dashboard", expanded=False):
            role = st.selectbox(
                "Your role (for context)",
                ["NOC operator", "Engineer", "Manager", "Student / Researcher", "Other"],
                index=0,
            )
            feedback_text = st.text_area(
                "What is most useful? What is confusing or missing?",
                height=120,
                key="overview_feedback",
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                ux_shap = st.slider("How helpful are the SHAP explanations? (1–5)", 1, 5, 4)
            with col_s2:
                ux_layout = st.slider("How clear is the layout? (1–5)", 1, 5, 4)
            with col_s3:
                ux_trust = st.slider("How much would you trust these alerts? (1–5)", 1, 5, 4)

            impact_estimate = st.selectbox(
                "Rough impact estimate if models were deployed",
                [
                    "Unknown / hard to estimate",
                    "Minor – quality-of-life improvements",
                    "Moderate – minutes of downtime avoided per week",
                    "High – tens of thousands of EUR per year in avoided penalties",
                ],
            )

            if st.button("Record feedback for thesis analysis"):
                if feedback_text.strip():
                    row = {
                        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                        "role": role,
                        "feedback": feedback_text.strip(),
                        "impact_estimate": impact_estimate,
                        "ux_shap": ux_shap,
                        "ux_layout": ux_layout,
                        "ux_trust": ux_trust,
                        "satellite_filter": satellite,
                        "time_window_display": st.session_state.get("time_window_display"),
                    }
                    append_feedback(row)
                    st.success("Feedback stored in CSV for thesis analysis.")
                else:
                    st.warning("Please write some feedback before submitting.")

    with col_right:
        gif_url = "https://i.gifer.com/AHJv.gif"
        st.image(gif_url, caption="Orbital view (www.gifer.com)", use_container_width=True)

    render_thesis_footer()




# ------------------------------
# Signal Loss page
# ------------------------------

def page_signal_loss():
    render_academic_banner()
    st.title(tr("Signal Loss"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "The model watches a small set of modem power and quality indicators over time. "
        "When they drift away from their usual pattern, the window is flagged as a potential "
        "signal loss scenario."
    )

    st.markdown("### Why this matters for operators")
    st.markdown(
        "- Persistent signal loss directly reduces availability and can trigger SLA penalties.\n"
        "- On a busy beam, an hour of partial outage may affect hundreds of terminals.\n"
        "- Repeated short drops are hard to see in raw KPIs; automated scoring focuses attention "
        "on the most risky windows."
    )

    st.markdown("### How this prototype works")
    st.markdown(
        "- Features are built from modem IN and OUT power statistics over short windows.\n"
        "- An autoencoder-style model learns the typical pattern and assigns an anomaly score.\n"
        "- Thresholding and eventisation convert noisy scores into a small number of alerts."
    )

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("### Feature importance over time for Signal Loss model (SHAP values)")

        shap_df, feat_names, time_labels = load_shap_matrix(
            "reports/figures/signal_loss_event_shap_values.csv"
        )
        if shap_df is not None:
            fig_shap = px.imshow(
                shap_df,
                x=time_labels,
                y=feat_names,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                labels={"x": "time step within window", "y": "feature", "color": "SHAP value"},
                title="Signal Loss – SHAP heatmap around one event",
            )
            fig_shap = add_shap_hover(
                fig_shap,
                x_label="time step",
                y_label="feature",
                context_note=(
                    "Per Cuéllar et al. (2024), brighter cells indicate features that most "
                    "pushed the model towards the anomalous class."
                ),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = load_image_path("reports/figures/signal_loss_event_heatmap.png")
            cont_img = load_image_path("reports/figures/signal_loss_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Signal Loss – SHAP heatmap around one event", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Signal Loss – continuous SHAP importance over time", use_container_width=True)

        with st.expander("Learn more about how to read this SHAP heatmap"):
            st.markdown(
                "- Rows are modem features (IN/OUT power statistics).\n"
                "- Columns are time steps in the window.\n"
                "- Warm colours push the model towards 'signal loss'; cool colours push towards 'normal'."
            )

        lit_expander("signal_loss")

        st.markdown("#### Example anomaly-score trend")
        scores = load_csv("data/processed/test_scores_raw.csv", parse_dates=["timestamp"])
        if scores is not None:
            if "timestamp" in scores.columns:
                scores = scores.rename(columns={"timestamp": "time"})
            if "proba_raw" in scores.columns:
                scores = scores.rename(columns={"proba_raw": "anomaly_score"})
            if "time" in scores.columns and "anomaly_score" in scores.columns:
                scores = apply_time_filter(scores, "time")
                scores = scores.sort_values("time").tail(600)
                fig = px.line(
                    scores,
                    x="time",
                    y="anomaly_score",
                    title="Recent signal-loss anomaly scores",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Signal-loss scores CSV not found; trend chart skipped in this environment.")

    with col_side:
        gif_url = "https://i.gifer.com/K6mM.gif"
        st.image(gif_url, caption="Signal Loss illustration", use_container_width=True)

        events = load_csv(
            ["data/processed/sl_test_eventized_scores.csv", "data/processed/sl_test_scores.csv"],
            parse_dates=["t_start", "t_end"],
        )
        alerts_df = None
        if events is not None:
            events["time_center"] = events["t_start"]
            events["severity"] = np.where(events.get("label", 1) == 1, "high", "medium")
            events["id"] = events.get("modem", "Unknown modem")
            events = apply_time_filter(events, "time_center")
            if not events.empty:
                alerts_df = events[["id", "time_center", "severity"]]
        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Alerts and suggested actions – recent high-risk windows",
            "signal_loss",
        )
        st.markdown(
            "- Correlate with weather, maintenance and pointing information.\n"
            "- If multiple modems on the same beam are affected, escalate as RF impairment.\n"
            "- If a single modem is affected, open a customer ticket and check terminal side first."
        )

    render_thesis_footer()

# ------------------------------
# Jamming page
# ------------------------------

def page_jamming():
    render_academic_banner()
    st.title(tr("Jamming / Interference"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "The jamming detector looks for unusual energy patterns in a subset of modem outputs. "
        "Instead of analysing full spectra, it uses modem statistics as a proxy for interference "
        "on the uplink or downlink."
    )

    st.markdown("### Why this matters for operators")
    st.markdown(
        "- Intentional jamming can degrade whole beams and affect many customers at once.\n"
        "- Early detection allows the NOC to trigger geolocation or reconfigure beams.\n"
        "- Without automation, weak but persistent interferers can remain unnoticed for hours."
    )

    st.markdown("### How this prototype works")
    st.markdown(
        "- An unsupervised model learns the joint behaviour of communication channels in quiet periods.\n"
        "- Windows where many channels move together in an unusual way get a high anomaly score.\n"
        "- SHAP explanations highlight which channels and time slices drove the alarm."
    )

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("### Feature importance over time for Jamming model (SHAP values)")

        shap_df, feat_names, time_labels = load_shap_matrix(
            "reports/figures/jamming_event_shap_values.csv"
        )
        if shap_df is not None:
            fig_shap = px.imshow(
                shap_df,
                x=time_labels,
                y=feat_names,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                labels={"x": "time step within window", "y": "metric / modem", "color": "SHAP value"},
                title="Jamming – SHAP heatmap around a suspected event",
            )
            fig_shap = add_shap_hover(
                fig_shap,
                x_label="time step",
                y_label="metric",
                context_note="Related to Li (2023) and Tritscher (2023) on interference anomalies.",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = load_image_path("reports/figures/jamming_event_heatmap.png")
            cont_img = load_image_path("reports/figures/jamming_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Jamming – SHAP heatmap around a suspected event", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Jamming – continuous SHAP importance over time", use_container_width=True)

        with st.expander("Learn more about how to read this SHAP heatmap"):
            st.markdown(
                "- Look for blocks of warm cells across several modems at the same time: "
                "these often correspond to wide-band interference.\n"
                "- Narrow warm bands in a single row may indicate a localised carrier issue.\n"
                "- Cool regions show features that argued against a jamming interpretation."
            )

        lit_expander("jamming")

    with col_side:
        events = load_csv("data/processed/jam_test_eventized_scores.csv", parse_dates=["t_start", "t_end"])
        alerts_df = None
        if events is not None:
            events["time_center"] = events["t_start"]
            events["severity"] = np.where(events.get("label", 1) == 1, "high", "medium")
            events["id"] = events.get("beam", "Unknown beam")
            events = apply_time_filter(events, "time_center")
            if not events.empty:
                alerts_df = events[["id", "time_center", "severity"]]
        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Alerts and suggested actions – recent high-risk windows",
            "jamming",
        )
        st.markdown(
            "- Cross-check with spectrum monitoring tools and confirm on a waterfall view.\n"
            "- Start geolocation if multiple beams show correlated interference.\n"
            "- Coordinate with customers to move critical carriers if necessary."
        )

    render_thesis_footer()

# ------------------------------
# SLA page
# ------------------------------

def page_sla():
    render_academic_banner()
    st.title(tr("SLA Breach"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "This model watches a throughput proxy KPI and compares it against SLA thresholds "
        "learned from historical data, aiming to provide early warning."
    )

    st.markdown("### Why this matters for operators")
    st.markdown(
        "- SLA violations impact revenue and customer satisfaction.\n"
        "- For premium customers, 30–60 minutes of degraded throughput can correspond to "
        "tens of thousands of euros in penalties.\n"
        "- Even a small lead time is valuable to reroute traffic or add capacity."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("#### SLA thresholds and breaches")
        sla_df = load_csv("data/processed/sla_breach_events.csv", parse_dates=["start", "end"])
        if sla_df is not None:
            sla_df = apply_time_filter(sla_df, "start")
            if not sla_df.empty:
                fig = px.scatter(
                    sla_df.sort_values("start").head(200),
                    x="start",
                    y="duration_s",
                    color="severity" if "severity" in sla_df.columns else None,
                    title="Example windows leading into SLA breaches",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No SLA breaches in this filtered window – system healthy.")
        else:
            st.info("No SLA breach CSV found in data/processed/sla_. Showing explanations only.")

        st.markdown("#### Feature importance over time for SLA risk model (SHAP values)")

        shap_df, feat_names, time_labels = load_shap_matrix(
            "data/processed/sla_event_shap_values.csv"
        )
        if shap_df is not None:
            fig_shap = px.imshow(
                shap_df,
                x=time_labels,
                y=feat_names,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                labels={"x": "time step within window", "y": "feature", "color": "SHAP value"},
                title="SLA – SHAP heatmap around one breach",
            )
            fig_shap = add_shap_hover(fig_shap, x_label="time step", y_label="feature")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = load_image_path("reports/figures/sla_event_heatmap.png")
            cont_img = load_image_path("reports/figures/sla_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="SLA – SHAP heatmap around one breach", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="SLA – continuous SHAP importance over time", use_container_width=True)

        with st.expander("Learn more about how to read this SHAP heatmap"):
            st.markdown(
                "- Dominant rows typically correspond to throughput level, volatility and short-term slope.\n"
                "- Deep blue patches before a breach show that throughput itself is pushing the model towards the breach class.\n"
                "- Red spikes on slope indicate sharp drops that the model treats as especially risky."
            )

        lit_expander("sla")

    with col_side:
        st.markdown("#### Alerts and suggested actions")
        sim_key = "sla_sim_alerts"
        if sim_key not in st.session_state:
            st.session_state[sim_key] = []

        if st.button("Simulate new SLA risk window"):
            st.session_state[sim_key].append(
                {
                    "id": "SIM-SLA",
                    "time_center": pd.Timestamp.utcnow().round("S"),
                    "severity": "high",
                }
            )
            st.info("Simulated high-risk SLA window added to the alert list.")

        sla_df = load_csv("data/processed/sla_breach_events.csv", parse_dates=["start", "end"])
        alerts_df = None
        if sla_df is not None:
            sla_df = apply_time_filter(sla_df, "start")
            if not sla_df.empty:
                sla_df = sla_df.copy()
                sla_df["time_center"] = sla_df["start"]
                sla_df["severity"] = np.where(sla_df.get("breach_flag", 1) == 1, "high", "medium")
                sla_df["id"] = sla_df.get("kpi_id", "SLA throughput")
                alerts_df = sla_df[["id", "time_center", "severity"]]

        if st.session_state[sim_key]:
            sim_df = pd.DataFrame(st.session_state[sim_key])
            if alerts_df is None:
                alerts_df = sim_df
            else:
                alerts_df = pd.concat([sim_df, alerts_df], ignore_index=True)

        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Current SLA risk windows",
            "sla",
        )

        st.markdown(
            "- If predicted breach is local to one beam, check utilisation and consider "
            "temporary capacity boost.\n"
            "- If several beams show risk, escalate to network planning and investigate "
            "ground segment issues.\n"
            "- For severe risk, each hour of outage can cost roughly 10–20k EUR in penalties "
            "and lost business for premium customers."
        )

    render_thesis_footer()

# ------------------------------
# Beam Handover page
# ------------------------------

def page_handover():
    render_academic_banner()
    st.title(tr("Beam Handover"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "Whenever the satellite or network controller moves a terminal from one beam to another, "
        "there is a short period where throughput can dip. This model tracks handovers and "
        "highlights those where throughput drops more than expected or recovers slowly."
    )

    st.markdown("### Why this matters for operators")
    st.markdown(
        "- Poorly behaving handovers can create repeated short outages that are hard to diagnose.\n"
        "- They often only affect mobile or aeronautical customers.\n"
        "- Early visibility enables targeted tuning of handover parameters."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("### Feature importance over time for Handover model (SHAP values)")
        shap_df, feat_names, time_labels = load_shap_matrix(
            "reports/figures/handover_event_shap_values.csv"
        )
        if shap_df is not None:
            fig_shap = px.imshow(
                shap_df,
                x=time_labels,
                y=feat_names,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                labels={"x": "time step within window", "y": "feature", "color": "SHAP value"},
                title="Beam Handover – SHAP heatmap around one anomalous handover",
            )
            fig_shap = add_shap_hover(fig_shap, x_label="time step", y_label="feature")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = load_image_path("reports/figures/handover_event_heatmap.png")
            cont_img = load_image_path("reports/figures/handover_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Beam Handover – SHAP heatmap around one anomalous handover", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Beam Handover – continuous SHAP importance over time", use_container_width=True)

        with st.expander("Learn more about how to read this SHAP heatmap"):
            st.markdown(
                "- Features include throughput before and after the handover, drop percentage "
                "and recovery time.\n"
                "- Warm regions after the handover mark windows where the model is concerned "
                "about slow or incomplete recovery."
            )

        lit_expander("handover")

    with col_side:
        events = load_csv("data/processed/handover_table.csv", parse_dates=["t"])
        alerts_df = None
        if events is not None:
            events = events.tail(200)
            events["time_center"] = events["t"]
            events["severity"] = np.where(events["drop_pct"].abs() > 0.05, "high", "medium")
            events["id"] = events.get("beam_id", "Unknown beam")
            events = apply_time_filter(events, "time_center")
            if not events.empty:
                alerts_df = events[["id", "time_center", "severity"]]
        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Alerts and suggested actions – recent anomalous handovers",
            "handover",
        )
        st.markdown(
            "- Check whether the same customer or route is affected repeatedly.\n"
            "- Inspect handover timing relative to satellite motion and beam footprints.\n"
            "- Consider adjusting hysteresis or thresholds for problem beams."
        )

    render_thesis_footer()

# ------------------------------
# Space Weather helpers
# ------------------------------

def fetch_live_kp_index():
    if st.session_state.get("disable_live_kp", False):
        return None, None
    try:
        url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "ses-thesis-dashboard/1.0"})
        resp.raise_for_status()
        data = resp.json()
        if not data or len(data) < 2:
            return None, None
        last_row = data[-1]
        ts_str = str(last_row[0])
        kp_raw = last_row[1]
        kp = float(kp_raw)
        return kp, ts_str
    except Exception:
        return None, None

# ------------------------------
# Space Weather page
# ------------------------------

def page_space_weather():
    render_academic_banner()
    st.title(tr("Space Weather"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "Space weather indices such as Kp capture geomagnetic activity. "
        "This prototype links those indices with thruster temperature and attitude "
        "error during station keeping and unload maneuvers, to flag windows where "
        "maneuver risk may be elevated."
    )

    st.markdown("### Why this matters for operators")
    st.markdown(
        "- During strong geomagnetic storms the environment around the spacecraft changes.\n"
        "- Maneuvers executed in those periods may have higher fuel usage or tighter thermal constraints.\n"
        "- A simple risk indicator helps flight dynamics teams choose safer windows."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("### Feature importance over time for Space Weather maneuver model (SHAP values)")
        shap_df, feat_names, time_labels = load_shap_matrix(
            "reports/figures/spaceweather_risky_shap_values.csv"
        )
        if shap_df is not None:
            fig_shap = px.imshow(
                shap_df,
                x=time_labels,
                y=feat_names,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                labels={"x": "maneuver index / time", "y": "feature", "color": "SHAP value"},
                title="Space Weather – SHAP heatmap for top risky maneuvers",
            )
            fig_shap = add_shap_hover(
                fig_shap,
                x_label="maneuver index",
                y_label="feature",
                context_note="Per Franco de la Peña et al. (2025) on manoeuvre risk and space weather.",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = load_image_path("reports/figures/spaceweather_risky_heatmap.png")
            cont_img = load_image_path("reports/figures/spaceweather_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Space Weather – SHAP heatmap for top risky maneuvers", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Space Weather – continuous SHAP importance over time", use_container_width=True)

        with st.expander("Learn more about how to read this SHAP heatmap"):
            st.markdown(
                "- Top rows correspond to thruster temperature and attitude error statistics; "
                "lower rows show Kp history and storm flags.\n"
                "- Warm cells mean those values pushed the classifier towards the risky class.\n"
                "- Blocks of warm Kp features across multiple maneuvers highlight prolonged disturbed periods."
            )

        lit_expander("spaceweather")

    with col_side:
        maneuvers = load_csv("ses_spaceweather_dataset.csv", parse_dates=["time"])
        alerts_df = None
        if maneuvers is not None and "risk_score" in maneuvers.columns:
            maneuvers = maneuvers.sort_values("time", ascending=False).head(50)
            maneuvers["time_center"] = maneuvers["time"]
            maneuvers["severity"] = np.where(maneuvers["risk_score"] > 0.6, "high", "medium")
            maneuvers["id"] = maneuvers["maneuver_type"]
            maneuvers = apply_time_filter(maneuvers, "time_center")
            if not maneuvers.empty:
                alerts_df = maneuvers[["id", "time_center", "severity"]]
        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Upcoming or recent risky maneuvers",
            "spaceweather",
        )
        st.markdown(
            "- Avoid planning non-urgent maneuvers during long periods with high Kp.\n"
            "- Coordinate with ground segment teams when storm intensity is high, as link margins may also be affected."
        )

    st.markdown("---")
    st.markdown(tr("Current space weather (live NOAA Kp index)"))

    kp_val, kp_ts = fetch_live_kp_index()
    if kp_val is None:
        st.info(
            "In this offline thesis environment the live Kp index call may be blocked. "
            "In a production deployment this panel would query NOAA's public API."
        )
    else:
        st.metric("Latest planetary K-index", f"{kp_val:.1f}")
        if kp_ts:
            st.caption("As of: " + kp_ts)
        st.caption(
            "Values above ~5 indicate geomagnetic storm levels that may affect "
            "satellite operations and link margins."
        )

    st.markdown(tr("Earth in real time (NOAA) – external view"))
    st.info(
        "For a full interactive view of Earth's current cloud cover and weather, "
        "open the NOAA 'Earth in Real Time' map in a new browser tab."
    )
    st.link_button(tr("Open NOAA Earth in Real Time"), "https://www.nesdis.noaa.gov/imagery/interactive-maps/earth-real-time")

    render_thesis_footer()

# ------------------------------
# Capacity page (synthetic demo with alerts)
# ------------------------------

def synth_capacity_demo():
    idx = pd.date_range("2021-10-25", "2021-11-01", freq="H")
    beams = ["Beam-A", "Beam-B", "Beam-C"]
    rows = []
    rng = np.random.default_rng(42)
    for b in beams:
        base_cap = rng.uniform(200, 260)
        for t in idx:
            demand = base_cap * rng.uniform(0.4, 1.2)
            cap = base_cap * rng.uniform(0.9, 1.1)
            rows.append({"time": t, "beam": b, "capacity": cap, "demand": demand})
    df = pd.DataFrame(rows)
    df["headroom"] = df["capacity"] - df["demand"]
    df["risk_index"] = 1.0 - (df["headroom"] / df["capacity"]).clip(0, 1)
    return df

def page_capacity():
    render_academic_banner()
    st.title(tr("Risk-aware Capacity Advisor"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page illustrates a forward-looking risk score per beam or region. "
        "The score combines demand forecast (synthetic in this prototype), available "
        "capacity and historical utilisation."
    )
    st.caption("Synthetic data; real SES metrics are discussed in the thesis evaluation chapter.")
    lit_expander("capacity")

    df = load_csv("data/processed/capacity_risk_demo.csv", parse_dates=["time"])
    if df is None or df.empty:        df = synth_capacity_demo()

    df = apply_time_filter(df, "time")

    beam_options = ["ALL"] + sorted(df["beam"].unique().tolist())
    beam = st.selectbox("Beam or region", beam_options)

    horizon_label = st.selectbox("Forecast horizon", ["Next 6 hours", "Next 24 hours", "Next 72 hours"])
    if "6" in horizon_label:
        horizon_hours = 6
    elif "24" in horizon_label:
        horizon_hours = 24
    else:
        horizon_hours = 72

    latest_time = df["time"].max()
    window_start = latest_time - pd.Timedelta(hours=horizon_hours)
    df_win = df[df["time"].between(window_start, latest_time)].copy()
    if beam != "ALL":
        df_win = df_win[df_win["beam"] == beam]

    if df_win.empty:
        st.info("No capacity data in this filtered window.")
        return

    st.markdown("#### Capacity, demand and risk over selected horizon")
    fig = px.line(
        df_win,
        x="time",
        y=["capacity", "demand", "risk_index"],
        labels={"value": "Mbps / risk", "variable": "Series"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Risk index values close to 1.0 indicate little spare headroom between demand "
        "and capacity. In a production system this would be driven by a demand forecast "
        "and spectrum-plan optimisation model."
    )

    st.markdown("#### Simple feature importance for risk_index (synthetic)")
    if all(c in df_win.columns for c in ["headroom"]):
        demo_imp = pd.DataFrame(
            {
                "driver": ["Demand pressure", "Headroom", "Capacity variability"],
                "avg_risk": [0.85, 0.65, 0.55],
            }
        )
        fig_imp = px.bar(
            demo_imp,
            x="driver",
            y="avg_risk",
            title="Average contribution of drivers to risk_index (demo)",
        )
        fig_imp.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.caption(
            "In a full implementation this section would show SHAP-based feature "
            "importance for risk_index per beam."
        )

    st.markdown("#### Alerts and suggested actions")
    df_last = df_win.sort_values("time", ascending=False).head(20)
    df_last["severity"] = np.where(df_last["risk_index"] > 0.8, "high", "medium")
    df_last["id"] = df_last["beam"]
    df_last["time_center"] = df_last["time"]
    alerts_df = df_last[["id", "time_center", "severity"]]
    render_alerts(
        alerts_df,
        "id",
        "time_center",
        "severity",
        "Beams with highest short-term capacity risk",
        "capacity",
    )
    st.markdown(
        "- For beams with persistent high risk, review traffic mix and consider temporary capacity increase.\n"
        "- Coordinate with planning teams if several adjacent beams show rising risk.\n"
        "- High sustained risk on busy beams could translate to roughly 5–10k EUR per hour "
        "in potential SLA penalties if left unmitigated."
    )

    render_thesis_footer()

# ------------------------------
# Stress index page
# ------------------------------

def synth_stress_demo():
    idx = pd.date_range("2021-10-25", "2021-11-01", freq="H")
    rng = np.random.default_rng(123)
    df = pd.DataFrame({"time": idx})
    df["signal_loss_risk"] = rng.uniform(0, 0.6, len(idx))
    df["jamming_risk"] = rng.uniform(0, 0.5, len(idx))
    df["sla_risk"] = rng.uniform(0, 0.7, len(idx))
    df["capacity_risk"] = rng.uniform(0, 0.8, len(idx))
    df["stress_index"] = df[
        ["signal_loss_risk", "jamming_risk", "sla_risk", "capacity_risk"]
    ].max(axis=1)
    return df

def page_stress():
    render_academic_banner()
    st.title(tr("Stress Index & Joint Risk Radar"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "The stress index combines signals from several models into one compact view. "
        "Instead of four or five separate alert streams, the NOC gets an at-a-glance "
        "indicator of how stressed the network is over time on each satellite. "
        "This design is proposed by Sabino et al. (2025) and inspired by "
        "system-level explanation work from Iino et al. (2024)."
    )
    lit_expander("stress")

    df = load_csv("data/processed/stress_index_demo.csv", parse_dates=["time"])
    if df is None or df.empty:        df = synth_stress_demo()

    df = apply_time_filter(df, "time")

    st.markdown("#### Stress index over time")
    fig = px.line(
        df,
        x="time",
        y="stress_index",
        title="Combined stress index (demo)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Peaks in the stress index correspond to periods where at least one underlying "
        "model showed elevated risk. Use this page as a radar: when stress spikes, "
        "check the table below to see which driver dominates and then drill down into "
        "the relevant use-case page from the left-hand menu."
    )

    st.markdown("#### Radar-style breakdown of dominant drivers")
    latest_slice = df.tail(48).copy()
    rows = []
    for _, r in latest_slice.iterrows():
        drivers = {
            "Signal loss": r.get("signal_loss_risk", 0.0),
            "Jamming": r.get("jamming_risk", 0.0),
            "SLA": r.get("sla_risk", 0.0),
            "Capacity": r.get("capacity_risk", 0.0),
        }
        dominant = max(drivers, key=drivers.get)
        rows.append(
            {
                "time": r["time"],
                "stress_index": r["stress_index"],
                "dominant_driver": dominant,
            }
        )
    df_dom = pd.DataFrame(rows)
    st.dataframe(df_dom.tail(12))

    st.caption(
        "Use the dominant_driver column as a pointer: for example, if several rows show "
        "Jamming, open the 'Jamming / Interference' page and inspect SHAP heatmaps "
        "and alerts for that period."
    )

    st.markdown("#### Polar chart of average risk contributions (demo)")
    avg_vals = {
        "Signal loss": df["signal_loss_risk"].mean() if "signal_loss_risk" in df.columns else 0.0,
        "Jamming": df["jamming_risk"].mean() if "jamming_risk" in df.columns else 0.0,
        "SLA": df["sla_risk"].mean() if "sla_risk" in df.columns else 0.0,
        "Capacity": df["capacity_risk"].mean() if "capacity_risk" in df.columns else 0.0,
    }
    polar_df = pd.DataFrame(
        {
            "driver": list(avg_vals.keys()),
            "avg_risk": list(avg_vals.values()),
        }
    )
    polar_df = pd.concat([polar_df, polar_df.iloc[[0]]], ignore_index=True)
    fig_polar = px.line_polar(
        polar_df,
        r="avg_risk",
        theta="driver",
        line_close=True,
        title="Average contribution of each driver to stress (demo)",
    )
    fig_polar.update_traces(fill="toself")
    fig_polar.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_polar, use_container_width=True)

    st.markdown("#### Alerts and suggested actions")
    df_last = df.sort_values("time", ascending=False).head(10)
    df_last["severity"] = np.where(df_last["stress_index"] > 0.8, "high", "medium")
    df_last["id"] = "SAT net"
    df_last["time_center"] = df_last["time"]
    alerts_df = df_last[["id", "time_center", "severity"]]
    render_alerts(
        alerts_df,
        "id",
        "time_center",
        "severity",
        "Most stressed recent periods",
        "stress",
    )
    st.markdown(
        "- Use this page as a radar: when stress spikes, consult the dominant-driver table "
        "and open the corresponding use case page.\n"
        "- Periods with sustained stress above ~0.8 across key beams could correspond to "
        "significant operational risk, potentially translating into tens of thousands of "
        "EUR per day if left unmanaged."
    )

    render_thesis_footer()

# ------------------------------
# Alert Analytics (Thesis Mode) -- OLD
# ------------------------------

def page_alert_analytics_OLD():
    render_academic_banner()
    st.title("Alert Analytics (Thesis Mode)")
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page aggregates alerts from all use cases (including their severities and "
        "acknowledgements). In the thesis this supports Phase 4 evaluation: measuring alert "
        "volume, severity mix and acknowledgement behaviour as a proxy for alert fatigue "
        "and operational usefulness."
    )
    lit_expander("alerts")

    if not ALERT_HISTORY_CSV.exists():
        st.info("No alert history CSV found yet. Interact with alerts in other pages to generate data.")
        render_thesis_footer()
        return

    alerts = pd.read_csv(ALERT_HISTORY_CSV)
    # ---- Compute time-to-ack (seconds) ----
    #if "time_center" in alerts.columns and "acked_at_utc" in alerts.columns:
    #    alerts["time_center"] = pd.to_datetime(alerts["time_center"], errors="coerce")
    #    alerts["acked_at_utc"] = pd.to_datetime(alerts["acked_at_utc"], errors="coerce")
    #    alerts["time_to_ack_s"] = (
    #        alerts["acked_at_utc"] - alerts["time_center"]
    #    ).dt.total_seconds()
    #else:
    #    alerts["time_to_ack_s"] = np.nan

    if "time_center" in alerts.columns:
        alerts["time_center"] = pd.to_datetime(alerts["time_center"], errors="coerce", utc=True)

    if "acked_at_utc" in alerts.columns:
        alerts["acked_at_utc"] = pd.to_datetime(alerts["acked_at_utc"], errors="coerce", utc=True)

    # Compute time-to-ack in seconds (safe)
    if "time_center" in alerts.columns and "acked_at_utc" in alerts.columns:
        alerts["time_to_ack_s"] = (alerts["acked_at_utc"] - alerts["time_center"]).dt.total_seconds()
    else:
        alerts["time_to_ack_s"] = np.nan



    #-----
    #if "time_center" in alerts.columns:
    #    alerts["time_center"] = pd.to_datetime(alerts["time_center"], errors="coerce")
    #if "acked_at_utc" in alerts.columns:
    #    alerts["acked_at_utc"] = pd.to_datetime(alerts["acked_at_utc"], errors="coerce")

    st.markdown("### Latest alerts (combined)")
    st.dataframe(alerts.sort_values("acked_at_utc", ascending=False).head(25))

    st.markdown("### Alert severity mix")
    sev_counts = alerts["severity"].value_counts().reset_index()
    sev_counts.columns = ["severity", "count"]
    if not sev_counts.empty:
        fig = px.bar(sev_counts, x="severity", y="count", title="Alert counts by severity")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Alerts over time (hourly)")
    if "time_center" in alerts.columns:
        alerts_valid = alerts.dropna(subset=["time_center"]).copy()
        alerts_valid["time_hour"] = alerts_valid["time_center"].dt.floor("H")
        by_hour = alerts_valid.groupby("time_hour").size().reset_index(name="alert_count")
        if not by_hour.empty:
            fig = px.line(by_hour, x="time_hour", y="alert_count", title="Alert volume per hour")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)



    acked = alerts.dropna(subset=["acked_at_utc"])
    ack_rate = len(acked) / max(len(alerts), 1)

   #median_tta = (
   #     acked["time_to_ack_s"].median()
   #     if "time_to_ack_s" in acked.columns and not acked.empty
   #     else np.nan
   #)

    median_tta = (
        acked["time_to_ack_s"].median()
        if "time_to_ack_s" in acked.columns and not acked.empty
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts", len(alerts))
    c2.metric("Acknowledged alerts", len(acked))
    c3.metric("Acknowledgement rate", f"{ack_rate*100:.1f}%")
    #c4.metric(
    #    "Median time-to-ack (s)",
    #    f"{median_tta:.0f}" if not np.isnan(median_tta) else "N/A"
    #)
    c4.metric(
        "Median time-to-ack (s)",
        f"{int(round(median_tta))} s" if median_tta is not None and not np.isnan(median_tta) else "N/A"
    )

    st.caption(
        "Time-to-ack is computed as the elapsed time between alert creation "
        "and operator acknowledgement. Lower values indicate clearer, more "
        "actionable alerts and reduced operational friction."
    )


    st.markdown("### Before / after threshold comparison")

    if "threshold_version" not in alerts.columns:
        st.info(
            "No threshold version tags found. "
            "Enable threshold tagging in alert generation to compare regimes."
        )
    else:
        cmp = alerts.groupby("threshold_version").agg(
            total_alerts=("alert_key", "count"),
            acknowledged=("acked_at_utc", lambda x: x.notna().sum()),
            median_time_to_ack_s=("time_to_ack_s", "median"),
        ).reset_index()

        cmp["ack_rate"] = cmp["acknowledged"] / cmp["total_alerts"]

        st.dataframe(
            cmp.style.format(
                {
                    "ack_rate": "{:.1%}",
                    "median_time_to_ack_s": "{:.0f}",
                }
            ),
            use_container_width=True
        )

        fig_vol = px.bar(
            cmp,
            x="threshold_version",
            y="total_alerts",
            title="Alert volume by threshold regime"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        fig_ack = px.bar(
            cmp,
            x="threshold_version",
            y="ack_rate",
            range_y=[0, 1],
            title="Acknowledgement rate by threshold regime"
        )
        st.plotly_chart(fig_ack, use_container_width=True)

        fig_tta = px.bar(
            cmp,
            x="threshold_version",
            y="median_time_to_ack_s",
            title="Median time-to-ack by threshold regime (seconds)"
        )
        st.plotly_chart(fig_tta, use_container_width=True)

    st.caption(
        "This comparison demonstrates the operational trade-off between alert volume "
        "and actionability. Precision-first settings typically reduce alert load while "
        "improving acknowledgement rates and response times."
    )




    st.download_button("Download alert history CSV", data=alerts.to_csv(index=False), file_name="alert_history.csv")

    render_thesis_footer()

# ------------------------------
# Alert Analytics (Thesis Mode)
# ------------------------------

def page_alert_analytics():
    render_academic_banner()
    st.title("Alert Analytics (Thesis Mode)")
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page aggregates alerts from all use cases (including their severities and "
        "acknowledgements). In the thesis this supports Phase 4 evaluation: measuring alert "
        "volume, severity mix and acknowledgement behaviour as a proxy for alert fatigue "
        "and operational usefulness."
    )
    lit_expander("alerts")

    if not ALERT_HISTORY_CSV.exists():
        st.info("No alert history CSV found yet. Interact with alerts in other pages to generate data.")
        render_thesis_footer()
        return

    alerts = pd.read_csv(ALERT_HISTORY_CSV)

    # --- Parse timestamps consistently as tz-aware UTC ---
    if "time_center" in alerts.columns:
        alerts["time_center"] = pd.to_datetime(alerts["time_center"], errors="coerce", utc=True)
    else:
        alerts["time_center"] = pd.NaT

    if "acked_at_utc" in alerts.columns:
        alerts["acked_at_utc"] = pd.to_datetime(alerts["acked_at_utc"], errors="coerce", utc=True)
    else:
        alerts["acked_at_utc"] = pd.NaT

    # --- Compute time-to-ack (seconds) safely ---
    alerts["time_to_ack_s"] = (alerts["acked_at_utc"] - alerts["time_center"]).dt.total_seconds()

    # Remove pathological negatives (can happen if clocks/serialization are inconsistent)
    alerts.loc[alerts["time_to_ack_s"] < 0, "time_to_ack_s"] = np.nan

    # Acknowledged subset
    acked = alerts.dropna(subset=["acked_at_utc"]).copy()
    ack_rate = len(acked) / max(len(alerts), 1)

    # Median TTA
    median_tta = (
        float(acked["time_to_ack_s"].median())
        if "time_to_ack_s" in acked.columns and not acked.empty
        else None
    )

    # Median TTA CI (bootstrap)
    median_ci = None
    if not acked.empty and "time_to_ack_s" in acked.columns:
        median_ci = bootstrap_median_ci(acked["time_to_ack_s"], n_boot=2000, ci=0.95, seed=42)

    st.markdown("### Latest alerts (combined)")
    st.dataframe(alerts.sort_values("acked_at_utc", ascending=False).head(25))

    st.markdown("### Alert severity mix")
    if "severity" in alerts.columns:
        sev_counts = alerts["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        if not sev_counts.empty:
            fig = px.bar(sev_counts, x="severity", y="count", title="Alert counts by severity")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No severity column found in alert history CSV.")

    st.markdown("### Alerts over time (hourly)")
    alerts_valid = alerts.dropna(subset=["time_center"]).copy()
    if not alerts_valid.empty:
        alerts_valid["time_hour"] = alerts_valid["time_center"].dt.floor("H")
        by_hour = alerts_valid.groupby("time_hour").size().reset_index(name="alert_count")
        if not by_hour.empty:
            fig = px.line(by_hour, x="time_hour", y="alert_count", title="Alert volume per hour")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Acknowledgement behaviour")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts", len(alerts))
    c2.metric("Acknowledged alerts", len(acked))
    c3.metric("Acknowledgement rate", f"{ack_rate*100:.1f}%")
    c4.metric(
        "Median time-to-ack",
        f"{int(round(median_tta))} s" if median_tta is not None and not np.isnan(median_tta) else "N/A"
    )

    if median_ci is not None:
        lo, hi = median_ci
        st.caption(f"95% bootstrap CI for median time-to-ack: {int(round(lo))} s to {int(round(hi))} s")
    else:
        st.caption("95% bootstrap CI not shown (insufficient acknowledged alerts).")

    st.caption(
        "Time-to-ack is computed as the elapsed time between alert creation "
        "and operator acknowledgement. Lower values indicate clearer, more "
        "actionable alerts and reduced operational friction."
    )

    # -----------------------------
    # Box plot: time-to-ack by severity
    # -----------------------------
    st.markdown("### Time-to-ack by severity (box plot)")

    acked_valid = alerts.dropna(subset=["acked_at_utc", "time_center", "time_to_ack_s"]).copy()

    # Optional cap to keep plot readable; adjust as needed
    acked_valid = acked_valid[(acked_valid["time_to_ack_s"] >= 0) & (acked_valid["time_to_ack_s"] <= 7 * 24 * 3600)]

    if acked_valid.empty or "severity" not in acked_valid.columns:
        st.info("Not enough acknowledged alerts with severity to plot time-to-ack distribution.")
    else:
        acked_valid["severity"] = acked_valid["severity"].astype(str).str.lower().str.strip()
        fig_box = px.box(
            acked_valid,
            x="severity",
            y="time_to_ack_s",
            points="all",
            title="Time-to-ack distribution by severity (seconds)",
            labels={"severity": "Severity", "time_to_ack_s": "Time-to-ack (s)"},
        )
        fig_box.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(
            "Interpretation: Lower medians and tighter spreads indicate faster, more actionable alerts. "
            "In Phase 4 this supports the alert-fatigue / actionability analysis."
        )

    # -----------------------------
    # Before / after threshold comparison
    # -----------------------------
    st.markdown("### Before / after threshold comparison")

    if "threshold_version" not in alerts.columns:
        st.info(
            "No threshold version tags found. "
            "Enable threshold tagging in alert generation to compare regimes."
        )
    else:
        # Use a robust count column even if alert_key doesn't exist
        count_col = "alert_key" if "alert_key" in alerts.columns else "alert_id"
        if count_col not in alerts.columns:
            # fallback: count rows
            cmp = alerts.groupby("threshold_version").size().reset_index(name="total_alerts")
            cmp["acknowledged"] = alerts.groupby("threshold_version")["acked_at_utc"].apply(lambda x: x.notna().sum()).values
            cmp["median_time_to_ack_s"] = alerts.groupby("threshold_version")["time_to_ack_s"].median().values
        else:
            cmp = alerts.groupby("threshold_version").agg(
                total_alerts=(count_col, "count"),
                acknowledged=("acked_at_utc", lambda x: x.notna().sum()),
                median_time_to_ack_s=("time_to_ack_s", "median"),
            ).reset_index()

        cmp["ack_rate"] = cmp["acknowledged"] / cmp["total_alerts"].replace(0, np.nan)

        st.dataframe(
            cmp.style.format(
                {
                    "ack_rate": "{:.1%}",
                    "median_time_to_ack_s": "{:.0f}",
                }
            ),
            use_container_width=True
        )

        fig_vol = px.bar(cmp, x="threshold_version", y="total_alerts", title="Alert volume by threshold regime")
        fig_vol.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_vol, use_container_width=True)

        fig_ack = px.bar(cmp, x="threshold_version", y="ack_rate", range_y=[0, 1], title="Acknowledgement rate by threshold regime")
        fig_ack.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ack, use_container_width=True)

        fig_tta = px.bar(cmp, x="threshold_version", y="median_time_to_ack_s", title="Median time-to-ack by threshold regime (seconds)")
        fig_tta.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_tta, use_container_width=True)

    st.caption(
        "This comparison demonstrates the operational trade-off between alert volume "
        "and actionability. Precision-first settings typically reduce alert load while "
        "improving acknowledgement rates and response times."
    )

    st.download_button("Download alert history CSV", data=alerts.to_csv(index=False), file_name="alert_history.csv")
    render_thesis_footer()

# ------------------------------
# Feedback Analytics (Thesis Mode)
# ------------------------------

def page_feedback_analytics():
    render_academic_banner()
    st.title("Feedback Analytics (Thesis Mode)")
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page summarises the operator / stakeholder feedback collected via the dashboard. "
        "It supports Phase 4 evaluation by providing quick views of sentiment, usability scores "
        "and themes related to SHAP explanations and trust."
    )
    lit_expander("feedback")

    if not FEEDBACK_CSV.exists():
        st.info("No feedback CSV found yet. Provide feedback on the Overview page to generate data.")
        render_thesis_footer()
        return

    fb = pd.read_csv(FEEDBACK_CSV)

    st.markdown("### 1. Role mix")
    if "role" in fb.columns:
        role_counts = fb["role"].value_counts().reset_index()
        role_counts.columns = ["role", "count"]
        fig = px.bar(role_counts, x="role", y="count", title="Feedback count by role")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Impact estimates")
    if "impact_estimate" in fb.columns:
        impact_counts = fb["impact_estimate"].value_counts().reset_index()
        impact_counts.columns = ["impact_estimate", "count"]
        fig = px.bar(impact_counts, x="impact_estimate", y="count", title="Perceived impact of deployment")
        fig.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3. Usability scores (1–5)")
    ux_cols = [c for c in fb.columns if c.startswith("ux_")]
    if ux_cols:
        ux_means = fb[ux_cols].mean().reset_index()
        ux_means.columns = ["dimension", "mean_score"]
        fig = px.bar(ux_means, x="dimension", y="mean_score", range_y=[1, 5], title="Average UX scores")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quantitative UX scores found in the CSV (columns ux_*).")

    
    # ---- Always run section 4 (independent of UX columns) ----
    st.markdown("### 4. Keyword and sentiment analysis of free-text feedback")
    st.caption(
        textwrap.dedent(
            """
            Sentiment polarity is computed using TextBlob when available:
            values range from -1 (very negative) through 0 (neutral) to +1 (very positive).
            """
        )
    )
    
    fb["feedback"] = fb.get("feedback", "").fillna("").astype(str)
    
    keyword_counts = {
        "shap": int(fb["feedback"].str.contains("shap", case=False, na=False).sum()),
        "explain": int(fb["feedback"].str.contains("explain", case=False, na=False).sum()),
        "confusing": int(fb["feedback"].str.contains("confus", case=False, na=False).sum()),
        "useful": int(fb["feedback"].str.contains("useful", case=False, na=False).sum()),
    }
    
    # --- Sentiment polarity (optional; works only if TextBlob is available) ---
    try:
        from textblob import TextBlob
        textblob_available = True
    except Exception:
        TextBlob = None
        textblob_available = False
    
    def safe_polarity(text: str) -> float | None:
        if not textblob_available:
            return None
        if text is None:
            return None
        t = str(text).strip()
        if not t:
            return None
        try:
            return float(TextBlob(t).sentiment.polarity)
        except Exception:
            return None
    
    if textblob_available:
        fb["sentiment_polarity"] = fb["feedback"].apply(safe_polarity)
        m = fb["sentiment_polarity"].mean()
        avg_polarity = float(m) if m == m else None
    else:
        fb["sentiment_polarity"] = float('nan')
        avg_polarity = None
    
    with st.expander("Keyword and sentiment summary", expanded=True):
        st.json(
            {
                "keyword_counts": keyword_counts,
                "sentiment_available": bool(textblob_available),
                "average_polarity": avg_polarity,
            }
        )
    
    if textblob_available:
        st.markdown("#### Sentiment polarity distribution")
        fig_sent = px.histogram(
            fb.dropna(subset=["sentiment_polarity"]),
            x="sentiment_polarity",
            nbins=20,
            range_x=[-1, 1],
            title="Distribution of feedback sentiment (TextBlob polarity)",
        )
        fig_sent.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("TextBlob is not available in this environment; sentiment distribution plot is disabled.")
    
    st.markdown("### 5. Raw feedback (for thematic analysis)")
    st.dataframe(fb)

    st.download_button("Download feedback CSV", data=fb.to_csv(index=False), file_name="feedback.csv")

    st.caption(
        "These summaries are designed to be exported or transcribed into the thesis evaluation "
        "chapter to close the loop between the dashboard and the research questions."
    )

    render_thesis_footer()

# ------------------------------
# Router
# ------------------------------

if view == tr("Overview"):
    page_overview()
elif view == tr("Signal Loss"):
    page_signal_loss()
elif view == tr("Jamming / Interference"):
    page_jamming()
elif view == tr("SLA Breach"):
    page_sla()
elif view == tr("Beam Handover"):
    page_handover()
elif view == tr("Space Weather"):
    page_space_weather()
elif view == tr("Risk-aware Capacity Advisor"):
    page_capacity()
elif view == tr("Stress Index & Joint Risk Radar"):
    page_stress()
elif view == tr("Alert Analytics (Thesis Mode)"):
    page_alert_analytics()
elif view == tr("Feedback Analytics (Thesis Mode)"):
    page_feedback_analytics()
