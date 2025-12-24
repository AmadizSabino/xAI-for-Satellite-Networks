Md
# Explainable AI for Satellite Networks — Anomaly Detection (MSc Dissertation)

This repository contains the implementation artefacts for the MSc dissertation:
**“A Framework for Explainable AI for Satellite Networks Anomaly Detection”**.

It provides:
- A Streamlit dashboard prototype (`app.py`) for anomaly monitoring and explanation
- Pre-generated, deterministic datasets used by the dashboard (`data/processed/`)
- Configuration files for thresholds and eventization (`config/`)
- Model and preprocessing artefacts (`models/`)
- Notebooks documenting the analytical pipeline (`notebooks/`)

## Quickstart (Run the dashboard)

1. Create a virtual environment (recommended)
2. Install dependencies:
```bash
pip install -r requirements.txt

## Run dashboard
Bash
streamlit run app.py
