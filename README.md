# Explainable AI for Satellite Networks — Anomaly Detection (MSc Dissertation)

This repository contains the implementation artefacts for the MSc Artificial Intelligence
dissertation:

**“A Framework for Explainable AI for Satellite Networks Anomaly Detection”**

The project investigates how Explainable AI (XAI) techniques can support anomaly
detection, operator trust, and decision-making in satellite network operations.

---

## Repository Contents

- `app.py`  
  Streamlit dashboard prototype used for demonstration and evaluation in the thesis.

- `data/processed/`  
  Pre-generated, deterministic datasets consumed by the dashboard.
  These are static artefacts to ensure reproducibility.

- `config/`  
  Configuration files for thresholds, eventization strategies, and model metadata.

- `models/`  
  Serialized model and preprocessing artefacts (e.g. scalers, model configs).

- `notebooks/`  
  Jupyter notebooks documenting data preparation, modelling, explainability analysis,
  and evaluation steps referenced in the dissertation.

- `reports/figures/`  
  Pre-rendered figures and heatmaps (e.g. SHAP visualisations) used by the dashboard
  and in the written report.

---

## Requirements

- Python **3.11** (recommended)
- See `requirements.txt` for dashboard runtime dependencies

The dashboard **does not train models at runtime**. All learning, SHAP computation,
and evaluation are performed offline in the notebooks.

---

## Quickstart — Run the Dashboard Locally

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

### 2. Install dependencies

```bash
pip install -r requirements.txt

streamlit run app.py


