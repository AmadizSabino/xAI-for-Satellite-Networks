##################################################################################################################################
#                                                                                                                                #
#                             An Explainable AI Framework for Satellite Network Anomaly Detection                                #
#                                                                                                                                #
#                                                                                                                                #
#                                           *******   Anomaly Dashboard   *******                                                #
#                                                                                                                                #
#                                                                                                                                #
#                                                    University of Hull                                                          #
#                                                MSc Artificial Intelligence                                                     #
#                                                                                                                                #
#                                                      Amadiz Sabino                                                             #
#                                                                                                                                #
##################################################################################################################################

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from functools import lru_cache
import requests
import json
import joblib
import textwrap

# ==========================================================
# Paths and global constants
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# Dashboard persistence (repo CSVs)
FEEDBACK_CSV = PROCESSED_DIR / "dashboard_feedback.csv"
ALERT_HISTORY_CSV = PROCESSED_DIR / "dashboard_alert_history.csv"

# Optional / legacy
ALERTS_CSV = PROCESSED_DIR / "alerts_history.csv"

CODE_REPO_URL = "https://github.com/AmadizSabino/xAI-for-Satellite-Networks"
THESIS_URL = "https://your-thesis-link"  # keep placeholder

# ==========================================================
# Robust IO helpers (MUST be defined before any reads)
# ==========================================================
def safe_read_csv_required(path: Path, name: str, **read_csv_kwargs) -> pd.DataFrame:
    """Hard-stop if a required file is missing."""
    if not path.exists():
        st.error(f"Required file not found: {name} (expected at: {path.as_posix()})")
        st.stop()
    try:
        return pd.read_csv(path, **read_csv_kwargs)
    except Exception as e:
        st.error(f"Failed to read required CSV: {name}. Error: {e}")
        st.stop()

def safe_read_csv_optional(path: Path, **read_csv_kwargs) -> pd.DataFrame | None:
    """Return None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **read_csv_kwargs)
    except Exception:
        return None

def load_csv(relative_paths, parse_dates=None):
    """
    Convenience loader that tries one or more relative paths under BASE_DIR.
    Returns None if nothing is found/readable.
    """
    paths = relative_paths if isinstance(relative_paths, (list, tuple)) else [relative_paths]
    for rel in paths:
        p = BASE_DIR / rel
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=parse_dates)
            except Exception:
                return None
    return None

def fig_path(filename: str) -> str | None:
    p = FIGURES_DIR / filename
    return str(p) if p.exists() else None

def load_shap_matrix(relative_path):
    df = load_csv(relative_path)
    if df is None:
        return None, None, None
    first_col = str(df.columns[0]).lower()
    if first_col.startswith("unnamed"):
        df = df.set_index(df.columns[0])
    feature_names = df.index.tolist()
    time_labels = df.columns.tolist()
    return df, feature_names, time_labels

# ==========================================================
# Optional config/model loads (do not crash app if missing)
# ==========================================================
def load_json_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_joblib_optional(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

model_handover = load_json_optional(MODELS_DIR / "model_handover.json")
scaler = load_joblib_optional(MODELS_DIR / "scaler_step8.joblib")

# Legacy names in your prior draft; keep optional to avoid runtime failures
config = load_json_optional(CONFIG_DIR / "config.json")
thresholds = load_json_optional(CONFIG_DIR / "thresholding.json")

# ==========================================================
# Translation engine (local dictionaries)
# ==========================================================
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
        "Academic mode (show literature links)":
            "Mode académique (afficher les références)",
        "Time window": "Fenêtre temporelle",
        "Last 24h": "Dernières 24 h",
        "Last 7 days": "7 derniers jours",
        "Full dataset": "Jeu de données complet",
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
        "Academic mode (show literature links)":
            "Modo académico (mostrar referencias)",
        "Time window": "Ventana de tiempo",
        "Last 24h": "Últimas 24 h",
        "Last 7 days": "Últimos 7 días",
        "Full dataset": "Conjunto completo",
        "Acknowledge": "Reconocer",
    },
}

@lru_cache(maxsize=4096)
def tr(text: str) -> str:
    lang_code = st.session_state.get("lang_code", "en")
    if lang_code == "en":
        return text
    mapping = LOCAL_TRANSLATIONS.get(lang_code, {})
    return mapping.get(text, text)

# ==========================================================
# Literature notes (Academic mode)
# ==========================================================
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

# ==========================================================
# Runtime persistence (works on Streamlit Cloud / restricted FS)
# ==========================================================
def append_feedback(row: dict):
    if "runtime_feedback" not in st.session_state:
        st.session_state["runtime_feedback"] = []
    st.session_state["runtime_feedback"].append(row)

def append_alert_history(rows: list[dict]):
    if not rows:
        return
    if "runtime_alert_history" not in st.session_state:
        st.session_state["runtime_alert_history"] = []
    st.session_state["runtime_alert_history"].extend(rows)

# ==========================================================
# Time window helpers
# ==========================================================
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

def apply_time_filter(df: pd.DataFrame | None, time_col: str):
    hours = get_time_window_hours()
    if df is None or hours is None or time_col not in df.columns:
        return df
    latest_time = df[time_col].max()
    if pd.isna(latest_time):
        return df
    window_start = latest_time - pd.Timedelta(hours=hours)
    return df[df[time_col].between(window_start, latest_time)].copy()

# ==========================================================
# Thesis branding
# ==========================================================
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
    st.markdown("""<hr style="margin-top:2rem; margin-bottom:0.4rem;" />""", unsafe_allow_html=True)
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

# ==========================================================
# SHAP hover helper
# ==========================================================
def add_shap_hover(fig, x_label="time", y_label="feature", context_note=None):
    hover = f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<br>SHAP=%{{z:.3f}}"
    if st.session_state.get("academic_mode", False) and context_note:
        hover += f"<br><br>{context_note}"
    hover += "<extra></extra>"
    fig.update_traces(hovertemplate=hover)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ==========================================================
# Alerts rendering + button CSS
# ==========================================================
def render_alerts(df, id_col, time_col, severity_col, title, usecase_key, max_rows=3):
    st.markdown(f"### {title}")

    # Button style
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
                time_col: pd.date_range("2021-11-01", periods=3, freq="H"),
                severity_col: ["high", "medium", "medium"],
            }
        )
        df_to_show = demo
    else:
        df_to_show = df.sort_values(time_col, ascending=False).head(max_rows)

    def make_key(row):
        return f"{usecase_key}:{row[id_col]}:{row[time_col]}"

    # Hide acked alerts
    df_to_show = df_to_show[[make_key(r) not in acked for _, r in df_to_show.iterrows()]]

    if df_to_show.empty:
        st.success("No anomalies in this window – system healthy.")
        return

    rows_to_persist = []

    for i, (_, row) in enumerate(df_to_show.iterrows()):
        key = make_key(row)
        sev = str(row[severity_col]).lower()

        # Log first-seen entry
        if key not in logged:
            rows_to_persist.append(
                {
                    "usecase": usecase_key,
                    "alert_id": row[id_col],
                    "time_center": row[time_col],
                    "severity": sev,
                    "acked_at_utc": None,
                    "threshold_version": None,  # placeholder if you later tag regimes
                    "alert_key": key,
                }
            )
            logged[key] = False

        badge_color = "#f97316"
        if sev == "high":
            badge_color = "#dc2626"
        elif sev == "medium":
            badge_color = "#fb923c"
        elif sev == "low":
            badge_color = "#16a34a"

        st.markdown(
            f"""
            <div style="
                border: 1px solid #e5e7eb;
                border-radius: 0.6rem;
                padding: 0.75rem 0.9rem;
                margin-bottom: 0.6rem;
                background-color: #ffffff;
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
                    rows_to_persist.append(
                        {
                            "usecase": usecase_key,
                            "alert_id": row[id_col],
                            "time_center": row[time_col],
                            "severity": sev,
                            "acked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                            "threshold_version": None,
                            "alert_key": key,
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

# ==========================================================
# Stats helpers
# ==========================================================
def bootstrap_median_ci(values, n_boot=2000, ci=0.95, seed=42):
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

# ==========================================================
# Space weather helpers
# ==========================================================
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
        kp = float(last_row[1])
        return kp, ts_str
    except Exception:
        return None, None

# ==========================================================
# Streamlit page config + sidebar
# ==========================================================
st.set_page_config(page_title="Satellite Anomaly Dashboard", layout="wide")

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title(tr("Anomaly Dashboard"))

academic_mode = st.sidebar.checkbox(tr("Academic mode (show literature links)"), value=True)
st.session_state["academic_mode"] = academic_mode

lang_label = st.sidebar.selectbox(tr("Language"), list(LANG_CODES.keys()), index=0)
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

satellite = st.sidebar.selectbox("Satellite (aggregated data)", ["M003", "M008", "M015", "M017", "ALL"])

time_window_options = [tr("Last 24h"), tr("Last 7 days"), tr("Full dataset")]
time_window_display = st.sidebar.selectbox(tr("Time window"), time_window_options, index=2)
st.session_state["time_window_display"] = time_window_display

# ==========================================================
# Pages
# ==========================================================
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
                - **PR-AUC (Precision–Recall area)** – how well the model lifts true anomalies above noise.
                - **ROC-AUC** – overall ability to separate normal vs anomalous windows across thresholds.
                - **Event precision / recall** – precision = of alerts raised, how many were real?
                  recall = of all real events, how many were caught?
                """
            )

        metrics = [
            ("Signal Loss model", 0.72, 0.83, 0.93, 0.13),
            ("SLA early warning", 0.25, 0.86, 1.00, 0.03),
            ("Beam Handover anomalies", 0.022, 0.71, 0.50, 0.012),
        ]

        interpretations = {
            "Signal Loss model": (
                "Strong at prioritising true signal-loss events (high precision), but catches a subset (moderate recall)."
            ),
            "SLA early warning": (
                "Very conservative early-warning indicator (rare alerts, high precision), not a complete SLA monitor."
            ),
            "Beam Handover anomalies": (
                "Weaker due to rarity/complexity; useful for surfacing candidates but needs further tuning."
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
        radar_melted = pd.concat([radar_melted, radar_melted[radar_melted["metric"] == "precision"]], ignore_index=True)

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

        lit_expander("overview")

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
                        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
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
                    st.success("Feedback recorded (runtime). You can export it on the Feedback Analytics page.")
                else:
                    st.warning("Please write some feedback before submitting.")

    with col_right:
        gif_url = "https://i.gifer.com/AHJv.gif"
        st.image(gif_url, caption="Orbital view (www.gifer.com)", use_container_width=True)

    render_thesis_footer()

# ----------------------------------------------------------
# Signal Loss page (use-case prefix: sl_)
# ----------------------------------------------------------
def page_signal_loss():
    render_academic_banner()
    st.title(tr("Signal Loss"))
    render_thesis_header()

    st.markdown("---")
    st.markdown("### What this use case monitors")
    st.markdown(
        "The model watches modem power and quality indicators over time. "
        "When they drift away from their usual pattern, the window is flagged as a potential signal-loss scenario."
    )

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("### Feature importance over time (SHAP values)")

        # Prefer CSV if you later export it; otherwise use PNGs already in reports/figures
        shap_df, feat_names, time_labels = load_shap_matrix("reports/figures/signal_loss_event_shap_values.csv")
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
                context_note="Brighter cells indicate features that most pushed the model towards the anomalous class.",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            event_img = fig_path("signal_loss_event_heatmap.png")
            cont_img = fig_path("signal_loss_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Signal Loss – SHAP heatmap around one event", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Signal Loss – continuous SHAP importance over time", use_container_width=True)

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
                fig = px.line(scores, x="time", y="anomaly_score", title="Recent signal-loss anomaly scores")
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Signal-loss scores CSV not found; trend chart skipped.")

    with col_side:
        gif_url = "https://i.gifer.com/K6mM.gif"
        st.image(gif_url, caption="Signal Loss illustration", use_container_width=True)

        # Use-case-prefixed files in processed/ (per your repo tree)
        events = load_csv(
            ["data/processed/sl_test_eventized_scores.csv", "data/processed/sl_test_scores.csv"],
            parse_dates=["t_start", "t_end"],
        )
        alerts_df = None
        if events is not None:
            # Make sure timestamps are real datetimes
            for c in ["t_start", "t_end"]:
                if c in events.columns:
                    events[c] = pd.to_datetime(events[c], errors="coerce", utc=True)

            events["time_center"] = events["t_start"] if "t_start" in events.columns else pd.NaT
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
            tr("Alerts and suggested actions – recent high-risk windows"),
            "sl",
        )

    render_thesis_footer()

# ----------------------------------------------------------
# Jamming page (use-case prefix: jam_ is not in tree; keep current names)
# ----------------------------------------------------------
def page_jamming():
    render_academic_banner()
    st.title(tr("Jamming / Interference"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "The jamming detector looks for unusual energy patterns in modem statistics as a proxy for interference."
    )

    col_main, col_side = st.columns([2, 1])
    with col_main:
        st.markdown("### Feature importance over time (SHAP values)")

        # Keep your current filename; you can later rename to jam_* if you produce it
        shap_df, feat_names, time_labels = load_shap_matrix("data/processed/jamming_event_shap_values.csv")
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
            event_img = fig_path("jamming_event_heatmap.png")
            cont_img = fig_path("jamming_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Jamming – SHAP heatmap around a suspected event", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Jamming – continuous SHAP importance over time", use_container_width=True)

        lit_expander("jamming")

    with col_side:
        # You do not currently have a jamming-prefixed events file in the tree.
        # Keep fallback: show demo alerts if None.
        events = None
        alerts_df = None
        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            tr("Alerts and suggested actions – recent high-risk windows"),
            "jam",
        )

    render_thesis_footer()

# ----------------------------------------------------------
# SLA page (use-case prefix: sla_)
# ----------------------------------------------------------
def page_sla():
    render_academic_banner()
    st.title(tr("SLA Breach"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This model watches a throughput proxy KPI and compares it against SLA thresholds learned from historical data."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("#### SLA thresholds and breaches")
        sla_df = load_csv("data/processed/sla_breach_events.csv", parse_dates=["start", "end"])
        if sla_df is not None:
            for c in ["start", "end"]:
                if c in sla_df.columns:
                    sla_df[c] = pd.to_datetime(sla_df[c], errors="coerce", utc=True)

            sla_df = apply_time_filter(sla_df, "start")
            if not sla_df.empty and "start" in sla_df.columns and "duration_s" in sla_df.columns:
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
            st.info("No sla_breach_events.csv found in data/processed/; showing explanations only.")

        st.markdown("#### Feature importance over time (SHAP values)")
        shap_df, feat_names, time_labels = load_shap_matrix("data/processed/sla_event_shap_values.csv")
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
            event_img = fig_path("sla_event_heatmap.png")
            cont_img = fig_path("sla_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="SLA – SHAP heatmap around one breach", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="SLA – continuous SHAP importance over time", use_container_width=True)

        lit_expander("sla")

    with col_side:
        st.markdown("#### Alerts and suggested actions")

        sim_key = "sla_sim_alerts"
        if sim_key not in st.session_state:
            st.session_state[sim_key] = []

        if st.button("Simulate new SLA risk window"):
            st.session_state[sim_key].append(
                {"id": "SIM-SLA", "time_center": pd.Timestamp.now(tz="UTC").round("S"), "severity": "high"}
            )
            st.info("Simulated high-risk SLA window added to the alert list.")

        alerts_df = None
        if sla_df is not None and not sla_df.empty:
            tmp = sla_df.copy()
            tmp["time_center"] = tmp["start"]
            tmp["severity"] = np.where(tmp.get("breach_flag", 1) == 1, "high", "medium")
            tmp["id"] = tmp.get("kpi_id", "SLA throughput")
            alerts_df = tmp[["id", "time_center", "severity"]]

        if st.session_state[sim_key]:
            sim_df = pd.DataFrame(st.session_state[sim_key])
            alerts_df = sim_df if alerts_df is None else pd.concat([sim_df, alerts_df], ignore_index=True)

        render_alerts(alerts_df, "id", "time_center", "severity", "Current SLA risk windows", "sla")

    render_thesis_footer()

# ----------------------------------------------------------
# Beam Handover page (use-case prefix: ho_)
# ----------------------------------------------------------
def page_handover():
    render_academic_banner()
    st.title(tr("Beam Handover"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This model tracks handovers and highlights those where throughput drops more than expected or recovers slowly."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("### Feature importance over time (SHAP values)")
        shap_df, feat_names, time_labels = load_shap_matrix("reports/figures/handover_event_shap_values.csv")
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
            event_img = fig_path("handover_event_heatmap.png")
            cont_img = fig_path("handover_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Beam Handover – SHAP heatmap around one anomalous handover", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Beam Handover – continuous SHAP importance over time", use_container_width=True)

        lit_expander("handover")

    with col_side:
        events = load_csv("data/processed/handover_table.csv", parse_dates=["t"])
        alerts_df = None
        if events is not None:
            if "t" in events.columns:
                events["t"] = pd.to_datetime(events["t"], errors="coerce", utc=True)
            events = events.tail(200)
            events["time_center"] = events["t"] if "t" in events.columns else pd.NaT
            events["severity"] = np.where(events.get("drop_pct", 0).abs() > 0.05, "high", "medium")
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
            "ho",
        )

    render_thesis_footer()

# ----------------------------------------------------------
# Space Weather page
# ----------------------------------------------------------
def page_space_weather():
    render_academic_banner()
    st.title(tr("Space Weather"))
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "Space weather indices such as Kp capture geomagnetic activity. This prototype links those indices with "
        "thruster temperature and attitude error during maneuvers to flag higher-risk windows."
    )

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("### Feature importance over time (SHAP values)")
        shap_df, feat_names, time_labels = load_shap_matrix("reports/figures/spaceweather_risky_shap_values.csv")
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
            event_img = fig_path("spaceweather_risky_heatmap.png")
            cont_img = fig_path("spaceweather_continuous_heatmap.png")
            if event_img:
                st.image(event_img, caption="Space Weather – SHAP heatmap for top risky maneuvers", use_container_width=True)
            if cont_img:
                st.image(cont_img, caption="Space Weather – continuous SHAP importance over time", use_container_width=True)

        lit_expander("spaceweather")

    with col_side:
        # FIXED: this was broken in your pasted file (you accidentally redefined safe_read_csv inside the page)
        maneuvers = load_csv("data/processed/ses_spaceweather_dataset.csv", parse_dates=["time"])
        alerts_df = None
        if maneuvers is not None:
            if "time" in maneuvers.columns:
                maneuvers["time"] = pd.to_datetime(maneuvers["time"], errors="coerce", utc=True)

            if "risk_score" in maneuvers.columns:
                maneuvers = maneuvers.sort_values("time", ascending=False).head(50)
                maneuvers["time_center"] = maneuvers["time"]
                maneuvers["severity"] = np.where(maneuvers["risk_score"] > 0.6, "high", "medium")
                maneuvers["id"] = maneuvers.get("maneuver_type", "maneuver")
                maneuvers = apply_time_filter(maneuvers, "time_center")
                if not maneuvers.empty:
                    alerts_df = maneuvers[["id", "time_center", "severity"]]

        render_alerts(
            alerts_df,
            "id",
            "time_center",
            "severity",
            "Upcoming or recent risky maneuvers",
            "sw",
        )

    st.markdown("---")
    st.markdown(tr("Current space weather (live NOAA Kp index)"))

    kp_val, kp_ts = fetch_live_kp_index()
    if kp_val is None:
        st.info(
            "In this offline thesis environment the live Kp index call may be blocked. "
            "In production this panel would query NOAA's public API."
        )
    else:
        st.metric("Latest planetary K-index", f"{kp_val:.1f}")
        if kp_ts:
            st.caption("As of: " + kp_ts)
        st.caption("Values above ~5 indicate geomagnetic storm levels that may affect satellite operations and link margins.")

    st.markdown(tr("Earth in real time (NOAA) – external view"))
    st.link_button(tr("Open NOAA Earth in Real Time"), "https://www.nesdis.noaa.gov/imagery/interactive-maps/earth-real-time")

    render_thesis_footer()

# ----------------------------------------------------------
# Capacity page (keep demo name even if file doesn't exist)
# ----------------------------------------------------------
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
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def page_capacity():
    render_academic_banner()
    st.title(tr("Risk-aware Capacity Advisor"))
    render_thesis_header()

    st.markdown("---")
    st.caption("Synthetic data if capacity_risk_demo.csv is missing; real SES metrics are discussed in the thesis.")

    lit_expander("capacity")

    # Keep the file name; if it doesn't exist, fall back to synth
    df = load_csv("data/processed/capacity_risk_demo.csv", parse_dates=["time"])
    if df is None:
        df = synth_capacity_demo()
    else:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    df = apply_time_filter(df, "time")

    beam_options = ["ALL"] + sorted(df["beam"].dropna().unique().tolist())
    beam = st.selectbox("Beam or region", beam_options)

    horizon_label = st.selectbox("Forecast horizon", ["Next 6 hours", "Next 24 hours", "Next 72 hours"])
    horizon_hours = 6 if "6" in horizon_label else 24 if "24" in horizon_label else 72

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
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Alerts and suggested actions")
    df_last = df_win.sort_values("time", ascending=False).head(20).copy()
    df_last["severity"] = np.where(df_last["risk_index"] > 0.8, "high", "medium")
    df_last["id"] = df_last["beam"]
    df_last["time_center"] = df_last["time"]
    alerts_df = df_last[["id", "time_center", "severity"]]
    render_alerts(alerts_df, "id", "time_center", "severity", "Beams with highest short-term capacity risk", "cap")

    render_thesis_footer()

# ----------------------------------------------------------
# Stress page
# ----------------------------------------------------------
def synth_stress_demo():
    idx = pd.date_range("2021-10-25", "2021-11-01", freq="H")
    rng = np.random.default_rng(123)
    df = pd.DataFrame({"time": pd.to_datetime(idx, utc=True)})
    df["signal_loss_risk"] = rng.uniform(0, 0.6, len(idx))
    df["jamming_risk"] = rng.uniform(0, 0.5, len(idx))
    df["sla_risk"] = rng.uniform(0, 0.7, len(idx))
    df["capacity_risk"] = rng.uniform(0, 0.8, len(idx))
    df["stress_index"] = df[["signal_loss_risk", "jamming_risk", "sla_risk", "capacity_risk"]].max(axis=1)
    return df

def page_stress():
    render_academic_banner()
    st.title(tr("Stress Index & Joint Risk Radar"))
    render_thesis_header()

    st.markdown("---")
    lit_expander("stress")

    df = load_csv("data/processed/stress_index_demo.csv", parse_dates=["time"])
    if df is None:
        df = synth_stress_demo()
    else:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    df = apply_time_filter(df, "time")

    st.markdown("#### Stress index over time")
    fig = px.line(df, x="time", y="stress_index", title="Combined stress index (demo)")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dominant drivers (last 48 points)")
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
        rows.append({"time": r["time"], "stress_index": r["stress_index"], "dominant_driver": dominant})
    st.dataframe(pd.DataFrame(rows).tail(12))

    st.markdown("#### Alerts and suggested actions")
    df_last = df.sort_values("time", ascending=False).head(10).copy()
    df_last["severity"] = np.where(df_last["stress_index"] > 0.8, "high", "medium")
    df_last["id"] = "SAT net"
    df_last["time_center"] = df_last["time"]
    alerts_df = df_last[["id", "time_center", "severity"]]
    render_alerts(alerts_df, "id", "time_center", "severity", "Most stressed recent periods", "stress")

    render_thesis_footer()

# ----------------------------------------------------------
# Alert Analytics (Thesis Mode)
# ----------------------------------------------------------
def page_alert_analytics():
    render_academic_banner()
    st.title("Alert Analytics (Thesis Mode)")
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page aggregates alerts from all use cases (including severities and acknowledgements). "
        "It supports Phase 4 evaluation: alert volume, severity mix, and acknowledgement behaviour."
    )
    lit_expander("alerts")

    repo_alerts = safe_read_csv_optional(ALERT_HISTORY_CSV)
    if repo_alerts is None:
        repo_alerts = pd.DataFrame()

    runtime_alerts = pd.DataFrame(st.session_state.get("runtime_alert_history", []))
    alerts = pd.concat([repo_alerts, runtime_alerts], ignore_index=True)

    if alerts.empty:
        st.info("No alert history found yet. Interact with alerts in other pages to generate data.")
        render_thesis_footer()
        return

    # Parse timestamps as tz-aware UTC
    alerts["time_center"] = pd.to_datetime(alerts.get("time_center", pd.NaT), errors="coerce", utc=True)
    alerts["acked_at_utc"] = pd.to_datetime(alerts.get("acked_at_utc", pd.NaT), errors="coerce", utc=True)

    alerts["time_to_ack_s"] = (alerts["acked_at_utc"] - alerts["time_center"]).dt.total_seconds()
    alerts.loc[alerts["time_to_ack_s"] < 0, "time_to_ack_s"] = np.nan

    acked = alerts.dropna(subset=["acked_at_utc"]).copy()
    ack_rate = len(acked) / max(len(alerts), 1)

    median_tta = float(acked["time_to_ack_s"].median()) if not acked.empty else None
    median_ci = bootstrap_median_ci(acked["time_to_ack_s"], n_boot=2000, ci=0.95, seed=42) if not acked.empty else None

    st.markdown("### Latest alerts (combined)")
    st.dataframe(alerts.sort_values("acked_at_utc", ascending=False).head(25))

    st.markdown("### Alert severity mix")
    if "severity" in alerts.columns:
        sev_counts = alerts["severity"].astype(str).str.lower().value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig = px.bar(sev_counts, x="severity", y="count", title="Alert counts by severity")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No severity column found in alert history.")

    st.markdown("### Alerts over time (hourly)")
    alerts_valid = alerts.dropna(subset=["time_center"]).copy()
    if not alerts_valid.empty:
        alerts_valid["time_hour"] = alerts_valid["time_center"].dt.floor("H")
        by_hour = alerts_valid.groupby("time_hour").size().reset_index(name="alert_count")
        fig = px.line(by_hour, x="time_hour", y="alert_count", title="Alert volume per hour")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Acknowledgement behaviour")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts", len(alerts))
    c2.metric("Acknowledged alerts", len(acked))
    c3.metric("Acknowledgement rate", f"{ack_rate*100:.1f}%")
    c4.metric("Median time-to-ack", f"{int(round(median_tta))} s" if median_tta is not None and not np.isnan(median_tta) else "N/A")

    if median_ci is not None:
        lo, hi = median_ci
        st.caption(f"95% bootstrap CI for median time-to-ack: {int(round(lo))} s to {int(round(hi))} s")
    else:
        st.caption("95% bootstrap CI not shown (insufficient acknowledged alerts).")

    st.markdown("### Time-to-ack by severity (box plot)")
    acked_valid = alerts.dropna(subset=["acked_at_utc", "time_center", "time_to_ack_s"]).copy()
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

    st.download_button("Download alert history CSV", data=alerts.to_csv(index=False), file_name="alert_history.csv")
    render_thesis_footer()

# ----------------------------------------------------------
# Feedback Analytics (Thesis Mode)
# ----------------------------------------------------------
def page_feedback_analytics():
    render_academic_banner()
    st.title("Feedback Analytics (Thesis Mode)")
    render_thesis_header()

    st.markdown("---")
    st.markdown(
        "This page summarises operator/stakeholder feedback collected via the dashboard, supporting Phase 4 evaluation."
    )
    lit_expander("feedback")

    repo_fb = safe_read_csv_optional(FEEDBACK_CSV)
    if repo_fb is None:
        repo_fb = pd.DataFrame()

    runtime_fb = pd.DataFrame(st.session_state.get("runtime_feedback", []))
    fb = pd.concat([repo_fb, runtime_fb], ignore_index=True)

    if fb.empty:
        st.info("No feedback found yet. Provide feedback on the Overview page to generate data.")
        render_thesis_footer()
        return

    st.markdown("### 1. Role mix")
    if "role" in fb.columns:
        role_counts = fb["role"].astype(str).value_counts().reset_index()
        role_counts.columns = ["role", "count"]
        fig = px.bar(role_counts, x="role", y="count", title="Feedback count by role")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Impact estimates")
    if "impact_estimate" in fb.columns:
        impact_counts = fb["impact_estimate"].astype(str).value_counts().reset_index()
        impact_counts.columns = ["impact_estimate", "count"]
        fig = px.bar(impact_counts, x="impact_estimate", y="count", title="Perceived impact of deployment")
        fig.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3. Usability scores (1–5)")
    ux_cols = [c for c in fb.columns if str(c).startswith("ux_")]
    if ux_cols:
        ux_means = fb[ux_cols].apply(pd.to_numeric, errors="coerce").mean().reset_index()
        ux_means.columns = ["dimension", "mean_score"]
        fig = px.bar(ux_means, x="dimension", y="mean_score", range_y=[1, 5], title="Average UX scores")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quantitative UX scores found (columns ux_*).")

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
        "shap": int(fb["feedback"].str.contains("shap", case=False).sum()),
        "explain": int(fb["feedback"].str.contains("explain", case=False).sum()),
        "confusing": int(fb["feedback"].str.contains("confus", case=False).sum()),
        "useful": int(fb["feedback"].str.contains("useful", case=False).sum()),
    }

    sentiment_available = False
    try:
        from textblob import TextBlob  # type: ignore
        fb["sentiment_polarity"] = fb["feedback"].apply(lambda x: TextBlob(x).sentiment.polarity)
        sentiment_available = True
    except Exception:
        fb["sentiment_polarity"] = np.nan
        sentiment_available = False

    avg_polarity = None
    if sentiment_available:
        m = fb["sentiment_polarity"].mean()
        if pd.notna(m):
            avg_polarity = float(m)

    with st.expander("Keyword and sentiment summary", expanded=True):
        st.json(
            {
                "keyword_counts": keyword_counts,
                "sentiment_available": bool(sentiment_available),
                "average_polarity": avg_polarity,
            }
        )

    if sentiment_available:
        st.markdown("#### Sentiment polarity distribution")
        fig_sent = px.histogram(
            fb,
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
    render_thesis_footer()

# ==========================================================
# Router
# ==========================================================
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
