import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_proba)
    return report, auc


def build_metrics_summary(report: dict, auc: float) -> dict:
    summary = {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "support": report["1"]["support"],
        "roc_auc": auc,
    }
    return summary


def interpret_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    try:
        import numpy as np
        import pandas as pd

        importances = model.feature_importances_
        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values(by="importance", ascending=False)
        df["importance_normalized"] = df["importance"] / df["importance"].sum()
        return df
    except AttributeError:
        raise ValueError("The provided model does not support feature_importances_")


def extract_feature_names_from_pipeline(pipeline) -> list[str]:
    """Attempt to reconstruct feature names from a preprocessing pipeline.

    Works with the project's `build_preprocessing_pipeline` structure.
    """
    try:
        numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargesPerMonth"]

        cat_transformer = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        cat_inputs = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure_bucket",
        ]
        cat_features = list(cat_transformer.get_feature_names_out(cat_inputs))
        feature_names = numeric_features + cat_features
        return feature_names
    except Exception as exc:
        raise RuntimeError("Unable to extract feature names from pipeline") from exc
