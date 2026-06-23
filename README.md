# Customer Churn Prediction Platform

A fresh end-to-end customer churn prediction project built from scratch with a modern Streamlit deployment experience.

## What This Project Includes

- Data ingestion and synthetic fallback dataset generation
- Cleaning, feature engineering, and model training pipeline
- Best model selection using business-focused evaluation
- Saved production pipeline and model artifacts
- Streamlit app for live churn scoring, retention recommendations, and business risk analysis

## Architecture

- `data/raw/` — original dataset input or generated synthetic sample
- `data/processed/` — cleaned and feature-engineered dataset ready for modeling
- `models/trained_models/` — saved production model pipeline and metrics
- `src/` — reusable pipeline modules:
  - `data_loader.py` — raw data loading and synthetic dataset generator
  - `preprocessing.py` — cleaning, feature engineering, transformer pipeline
  - `model_builder.py` — model training and best-model selection
  - `model_evaluation.py` — evaluation metrics and feature importance
  - `visualization.py` — application-ready charts and business plots
- `run_all.py` — pipeline orchestrator from raw data to saved model
- `app.py` — Streamlit deployment app with prediction, insights, and retention strategy

## Technology Stack

- Python 3.x
- pandas / numpy for data processing
- scikit-learn for preprocessing and modeling
- Streamlit for deployment and user experience
- matplotlib / seaborn for charts
- joblib for model serialization

## Getting Started

1. Open a terminal inside the project folder.
2. Create and activate a Python environment.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Train the pipeline and save artifacts:

```bash
python run_all.py --reports-dir reports
```

4. Launch the interactive dashboard:

```bash
streamlit run app.py
```

## New features added

- CLI option in `run_all.py` to write evaluation reports and plots to `reports/` (use `--reports-dir`).
- Logging helper at `src/logger.py` and a top-level `config.yaml` for common paths.
- The training run now saves metrics JSON and PNG plots (`churn_distribution.png`, `feature_importance.png`, `correlation_matrix.png`) to the reports folder.

## Notes

- If the telecom dataset is not available under `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, the project will generate a realistic synthetic dataset automatically.
- The Streamlit app includes an interactive churn risk simulator, retention recommendation engine, and live what-if analysis.

## Unique Value

- Unique synthetic fallback ensures the project runs even without the original data file.
- Business-ready retention insights turn churn scores into action recommendations.
- Clean modular code and production-ready model artifact saving make this repo deployment friendly.


## LICENCE 
  Author - Deepthish 
  Mail id - deepthishraj@gmail.com
