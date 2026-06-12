from pathlib import Path
import pandas as pd

from src.data_loader import load_raw_data, generate_synthetic_data
from src.model_builder import train_best_model, save_model_pipeline
from src.preprocessing import prepare_training_data


def main():
    root = Path(__file__).resolve().parent
    raw_path = root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    synthetic_path = root / "data" / "raw" / "synthetic_churn_data.csv"
    processed_path = root / "data" / "processed" / "churn_processed.csv"
    model_path = root / "models" / "trained_models" / "churn_pipeline.joblib"

    try:
        df = load_raw_data(raw_path)
        print(f"Loaded raw dataset from {raw_path}")
    except FileNotFoundError:
        df = generate_synthetic_data(rows=7043)
        df.to_csv(synthetic_path, index=False)
        print(f"Raw dataset not found. Generated synthetic dataset at {synthetic_path}")

    X_transformed, y, pipeline = prepare_training_data(df)
    model, metrics = train_best_model(X_transformed, y)
    save_model_pipeline(pipeline, model, model_path)
    df.to_csv(processed_path, index=False)

    print("Training completed and model saved.")
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    print(f"Saved the model pipeline to {model_path}")
    print(f"Saved processed data snapshot to {processed_path}")


if __name__ == "__main__":
    main()
