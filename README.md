# Explainable AI for Satellite Networks — Anomaly Dashboard

This repository contains a Streamlit prototype dashboard built for the University of Hull **MSc Artificial Intelligence** dissertation:

**An Explainable AI Framework for Satellite Network Anomaly Detection**

The prototype is intended for demonstration and evaluation (Phase 4), not for production operations.

## What the dashboard provides

- **Multiple use-cases** presented as separate views:
  - Signal Loss
  - Jamming / Interference
  - SLA Breach (early warning)
  - Beam Handover
  - Space Weather
  - Risk-aware Capacity Advisor (demo / synthetic fallback)
  - Stress Index & Joint Risk Radar (demo / synthetic fallback)
- **Explainability panels**:
  - SHAP heatmaps (CSV-first, with optional PNG fallbacks under `reports/figures/`)
  - Hover tooltips and a small “Academic mode” literature context expander
- **Operator workflow simulation**:
  - Alerts list with per-alert acknowledgement
  - Cross-use-case **Alert Analytics** (alert volume, severity mix, time-to-ack metrics)
  - **Feedback capture** and **Feedback Analytics** (role mix, UX scores, simple keyword + sentiment analysis)
- **Optional live space-weather indicator**:
  - Fetches the NOAA planetary K-index via NOAA SWPC public endpoint (if outbound requests are allowed)

## Repository structure

Typical structure expected by `app.py`:

```
.
├── app.py
├── data
│   ├── processed
│   │   ├── dashboard_alert_history.csv        # optional (persisted)
│   │   ├── dashboard_feedback.csv             # optional (persisted)
│   │   ├── sl_test_eventized_scores.csv       # optional
│   │   ├── sl_test_scores.csv                 # optional
│   │   ├── test_scores_raw.csv                # optional
│   │   ├── sla_breach_events.csv              # optional
│   │   ├── handover_table.csv                 # optional
│   │   ├── ses_spaceweather_dataset.csv       # optional
│   │   ├── capacity_risk_demo.csv             # optional
│   │   └── stress_index_demo.csv              # optional
│   └── interim
├── models
│   ├── model_handover.json                    # optional
│   └── scaler_step8.joblib                    # optional
├── config
│   ├── ho_config.json                         # optional (legacy)
│   ├── sl_config.json                         # optional (legacy)
│   ├── ho_thresholding.json                   # optional (legacy)
│   └── sla_thresholding.json                  # optional (legacy)
└── reports
    └── figures
        ├── signal_loss_event_shap_values.csv  # optional
        ├── handover_event_shap_values.csv     # optional
        ├── spaceweather_risky_shap_values.csv # optional
        └── *.png                              # optional fallbacks
```

Notes:
- Most files are **optional**. If a dataset/figure file is missing, the dashboard either hides the chart or falls back to a demo visual (for specific views).
- Alert and feedback persistence is supported via **CSV files** in `data/processed/` if present. In restricted environments the app also uses Streamlit session state.

## Requirements

Core dependencies:
- Python 3.10+ (recommended)
- streamlit
- pandas
- numpy
- plotly
- requests
- joblib

Optional dependency:
- **textblob** (only used for sentiment polarity in the Feedback Analytics page). If not installed, sentiment plotting is disabled and the dashboard still runs.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install streamlit pandas numpy plotly requests joblib
# Optional
pip install textblob
```

## Run the dashboard

From the repository root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Operational notes

- **Time filtering:** The sidebar “Time window” selector filters charts and alerts where timestamps are available.
- **NOAA Kp index:** If outbound HTTP is blocked (common on some hosted environments), the Space Weather page will display an informational message instead of live values.
- **Reproducibility:** The dashboard is designed to be robust to missing files; however, to reproduce thesis figures exactly, place the exported CSVs/PNGs under `reports/figures/` using the filenames referenced in `app.py`.

## Links

- Code repository: https://github.com/AmadizSabino/xAI-for-Satellite-Networks
- Thesis report folder: https://github.com/AmadizSabino/xAI-for-Satellite-Networks/tree/main/ThesisReport
