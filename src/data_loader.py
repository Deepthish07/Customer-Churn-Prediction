import random
from pathlib import Path
import pandas as pd


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        return df
    raise FileNotFoundError(f"Dataset not found at {raw_path}")


def generate_synthetic_data(rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    import numpy as np

    tenure = np.random.randint(0, 73, size=rows)
    monthly_charges = np.round(np.random.uniform(18.0, 120.0, size=rows), 2)
    total_charges = np.round(np.where(tenure == 0, 0.0, tenure * monthly_charges + np.random.uniform(-30, 30, size=rows)), 2)
    gender = np.random.choice(["Male", "Female"], size=rows, p=[0.51, 0.49])
    senior_citizen = np.random.choice([0, 1], size=rows, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=rows, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=rows, p=[0.31, 0.69])
    phone_service = np.random.choice(["Yes", "No"], size=rows, p=[0.9, 0.1])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], size=rows, p=[0.25, 0.65, 0.1])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=rows, p=[0.34, 0.44, 0.22])
    online_security = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.19, 0.55, 0.26])
    online_backup = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.22, 0.54, 0.24])
    device_protection = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.23, 0.53, 0.24])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.18, 0.56, 0.26])
    streaming_tv = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.29, 0.51, 0.2])
    streaming_movies = np.random.choice(["Yes", "No", "No internet service"], size=rows, p=[0.31, 0.49, 0.2])
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=rows, p=[0.54, 0.24, 0.22])
    paperless_billing = np.random.choice(["Yes", "No"], size=rows, p=[0.59, 0.41])
    payment_method = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ], size=rows, p=[0.34, 0.21, 0.22, 0.23])

    churn_score = (
        0.35 * (tenure < 12).astype(int)
        + 0.25 * (contract == "Month-to-month").astype(int)
        + 0.15 * (paperless_billing == "Yes").astype(int)
        + 0.08 * (internet_service == "Fiber optic").astype(int)
        + 0.12 * senior_citizen
    )
    churn = np.where(churn_score + np.random.normal(0, 0.12, rows) > 0.47, "Yes", "No")

    df = pd.DataFrame(
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )
    return df
