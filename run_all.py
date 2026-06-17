from pathlib import Path
import argparse
import json
import pandas as pd
from datetime import datetime

from src.data_loader import load_raw_data, generate_synthetic_data
from src.model_builder import train_best_model, save_model_pipeline
from src.preprocessing import prepare_training_data
from src.visualization import plot_churn_balance, plot_feature_importance, plot_correlation_matrix
from src.model_evaluation import interpret_feature_importance, extract_feature_names_from_pipeline


def save_metrics_report(metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"metrics_{ts}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return report_path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train churn model and produce reports")
    parser.add_argument("--reports-dir", default=str(root / "reports"), help="Directory to write reports and plots")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
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

    # Save evaluation metrics and plots
    try:
        feature_names = extract_feature_names_from_pipeline(pipeline)
        importance_df = interpret_feature_importance(model, feature_names)
        reports_dir.mkdir(parents=True, exist_ok=True)
        # save metrics JSON
        report_path = save_metrics_report(metrics, reports_dir)
        print(f"Saved metrics report to {report_path}")

        # save plots
        fig1 = plot_churn_balance(df)
        fig1_path = reports_dir / "churn_distribution.png"
        fig1.savefig(fig1_path, bbox_inches="tight")
        fig2 = plot_feature_importance(importance_df, top_n=12)
        fig2_path = reports_dir / "feature_importance.png"
        fig2.savefig(fig2_path, bbox_inches="tight")
        fig3 = plot_correlation_matrix(df)
        fig3_path = reports_dir / "correlation_matrix.png"
        fig3.savefig(fig3_path, bbox_inches="tight")

        print(f"Saved plots to {reports_dir}")
    except Exception as exc:
        print(f"Warning: could not save full reports: {exc}")

    print("Training completed and model saved.")
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    print(f"Saved the model pipeline to {model_path}")
    print(f"Saved processed data snapshot to {processed_path}")


if __name__ == "__main__":
    main()
