# 🛫 AI-Powered Airport Passenger Flow & Delay Prediction

A single, polished Streamlit application that combines **machine-learning delay
prediction** with a **SimPy passenger-flow simulation** of an airport terminal.
The trained model is wired directly into the live simulation, so the flight board
shows real model predictions — not hardcoded statuses.

## Features

| Tab | What it does |
| --- | --- |
| 📊 **Dataset & Preprocessing** | Auto-generates a realistic 3,000-flight dataset, shows raw preview, missing-value heatmap, delay-rate breakdowns, distributions, correlation heatmap, and live before/after cleaning stats. |
| 🤖 **ML Training & Comparison** | Trains Logistic Regression, Decision Tree & Random Forest with a live progress bar. Compares accuracy / precision / recall / F1, shows the best model's confusion matrix and Random Forest feature importance. Saves the best model to `models/best_model.joblib`. |
| 🎛 **Flow Simulation** | Configurable SimPy simulation (passengers, arrival rate, counters, lanes, gates) with summary metrics and four Plotly charts including an animated queue timeline. |
| 🎬 **Live Simulation** | Plotly-animated airport map of passengers flowing check-in → security → boarding, plus an ML-driven live flight board with Play/Pause and frame-speed controls. |
| 🔮 **Prediction Tool** | Form-based single-flight prediction with a colored verdict badge, probability gauge, and a SHAP-style feature-contribution chart. |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch the dataset is generated automatically. Train the models in the
**ML Training** tab to unlock the live simulation and prediction tabs.

## Project structure

```
airport_system_v2/
├── app.py                 # Streamlit entry point (UI orchestration only)
├── requirements.txt
├── data/raw/              # auto-generated airline_delay.csv
├── models/                # best_model.joblib (saved after training)
└── src/
    ├── data_pipeline.py   # dataset generation, cleaning, feature engineering
    ├── ml_pipeline.py     # train/compare LR·DT·RF, persist best model, predict
    ├── simulation.py      # SimPy terminal simulation + animation frame builder
    └── visualizations.py  # all Plotly chart builders (plotly_dark)
```

## Tech stack

Python 3.11+ · Streamlit · scikit-learn · SimPy · Plotly · pandas · numpy · joblib

> All charts are Plotly (`template="plotly_dark"`); there is no matplotlib in the UI.
