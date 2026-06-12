from pathlib import Path
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["TotalCharges"] = X["TotalCharges"].fillna(X["MonthlyCharges"] * X["tenure"])
        X["SeniorCitizen"] = X["SeniorCitizen"].astype(int)
        X["tenure_bucket"] = pd.cut(
            X["tenure"],
            bins=[-1, 0, 12, 24, 48, 60, 72],
            labels=["0", "1-12", "13-24", "25-48", "49-60", "61-72"],
        )
        X["AvgChargesPerMonth"] = X["TotalCharges"] / (X["tenure"].replace(0, 1))
        X["MonthlyCharges"] = X["MonthlyCharges"].round(2)
        return X


def build_preprocessing_pipeline() -> Pipeline:
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargesPerMonth"]
    categorical_features = [
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

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([("cleaner", DataCleaner()), ("preprocessor", preprocessor)])
    return pipeline


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    pipeline = build_preprocessing_pipeline()
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X_transformed = pipeline.fit_transform(X)
    return X_transformed, y, pipeline
