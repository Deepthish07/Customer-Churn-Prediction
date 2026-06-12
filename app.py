from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_loader import generate_synthetic_data, load_raw_data
from src.model_builder import load_model_pipeline
from src.preprocessing import DataCleaner, build_preprocessing_pipeline
from src.model_evaluation import build_metrics_summary
from src.visualization import plot_churn_balance, plot_feature_importance


RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/trained_models/churn_pipeline.joblib")

FEATURE_OPTIONS = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def load_app_data():
    try:
        df = load_raw_data(RAW_PATH)
        st.sidebar.success("Loaded raw dataset from disk.")
    except FileNotFoundError:
        df = generate_synthetic_data(rows=7043)
        st.sidebar.warning("Raw dataset not found. Using generated synthetic data.")
    return df


def load_artifact():
    if MODEL_PATH.exists():
        artifact = load_model_pipeline(MODEL_PATH)
        return artifact["pipeline"], artifact["model"]
    return None, None


def build_feature_vector(form_data: dict) -> pd.DataFrame:
    return pd.DataFrame([form_data])


def retention_recommendation(score: float, inputs: dict) -> list[str]:
    actions = []
    if score > 0.7:
        actions.append("Offer a 2-year contract discount to lock in the customer.")
    elif score > 0.5:
        actions.append("Propose a loyalty promotion or bundle package.")
    else:
        actions.append("Monitor the account and focus on high-value cross-sell opportunities.")
    if inputs["Contract"] == "Month-to-month":
        actions.append("Offer a longer-term plan with savings compared to month-to-month billing.")
    if inputs["PaperlessBilling"] == "Yes":
        actions.append("Provide a digital loyalty reward or billing convenience perk.")
    if inputs["TechSupport"] == "No":
        actions.append("Promote premium technical support to reduce churn risk.")
    return actions


def main():
    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction & Retention Assistant")

    df = load_app_data()
    pipeline, model = load_artifact()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head(10))
        st.write("#### Churn distribution")
        st.pyplot(plot_churn_balance(df))

    with col2:
        st.subheader("Churn Insight")
        st.markdown(
            "This platform estimates the probability of a customer leaving based on service adoption, contract type, and billing behavior."
        )
        st.markdown("- **High churn risk**: focus on retention bundles and service loyalty offers.")
        st.markdown("- **Medium churn risk**: highlight stability and support benefits.")
        st.markdown("- **Low churn risk**: identify up-sell opportunities to grow revenue.")

    st.sidebar.header("Customer Profile")
    form_data = {
        "gender": st.sidebar.selectbox("Gender", FEATURE_OPTIONS["gender"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", FEATURE_OPTIONS["Partner"]),
        "Dependents": st.sidebar.selectbox("Dependents", FEATURE_OPTIONS["Dependents"]),
        "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", FEATURE_OPTIONS["PhoneService"]),
        "MultipleLines": st.sidebar.selectbox("Multiple Lines", FEATURE_OPTIONS["MultipleLines"]),
        "InternetService": st.sidebar.selectbox("Internet Service", FEATURE_OPTIONS["InternetService"]),
        "OnlineSecurity": st.sidebar.selectbox("Online Security", FEATURE_OPTIONS["OnlineSecurity"]),
        "OnlineBackup": st.sidebar.selectbox("Online Backup", FEATURE_OPTIONS["OnlineBackup"]),
        "DeviceProtection": st.sidebar.selectbox("Device Protection", FEATURE_OPTIONS["DeviceProtection"]),
        "TechSupport": st.sidebar.selectbox("Tech Support", FEATURE_OPTIONS["TechSupport"]),
        "StreamingTV": st.sidebar.selectbox("Streaming TV", FEATURE_OPTIONS["StreamingTV"]),
        "StreamingMovies": st.sidebar.selectbox("Streaming Movies", FEATURE_OPTIONS["StreamingMovies"]),
        "Contract": st.sidebar.selectbox("Contract", FEATURE_OPTIONS["Contract"]),
        "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", FEATURE_OPTIONS["PaperlessBilling"]),
        "PaymentMethod": st.sidebar.selectbox("Payment Method", FEATURE_OPTIONS["PaymentMethod"]),
        "MonthlyCharges": float(st.sidebar.slider("Monthly Charges", 18.0, 120.0, 60.0, 0.5)),
        "TotalCharges": float(st.sidebar.slider("Total Charges", 0.0, 9000.0, 1200.0, 5.0)),
    }

    st.sidebar.markdown("---")
    st.sidebar.caption("Run `python run_all.py` first to train the production model if needed.")

    with st.expander("Prediction settings", expanded=True):
        st.write("Enter a customer profile and inspect expected churn risk.")

    if pipeline is None or model is None:
        st.error("The production model is not available. Run `python run_all.py` to train and save the model.")
        return

    if st.sidebar.button("Score this customer"):
        input_df = build_feature_vector(form_data)
        transformed = pipeline.transform(input_df)
        probability = float(model.predict_proba(transformed)[:, 1][0])
        label = "High risk" if probability >= 0.5 else "Low risk"
        st.metric("Churn probability", f"{probability:.2%}", delta=None)
        st.success(f"Predicted risk: {label}")

        st.write("### Recommended Retention Actions")
        recommendations = retention_recommendation(probability, form_data)
        for rec in recommendations:
            st.write(f"- {rec}")

        st.write("### Model Confidence and Business Implications")
        if probability >= 0.7:
            st.warning("The customer is at high risk of churn. Prioritize outreach and incentives.")
        elif probability >= 0.5:
            st.info("The customer has moderate churn risk. Consider targeted retention offers.")
        else:
            st.success("Low churn risk. Maintain satisfaction while exploring cross-sell opportunities.")

    st.write("---")
    st.write("## Model Performance")
    st.write(
        "Use `run_all.py` to generate model evaluation metrics and compare model performance across retention strategies." 
    )

    feature_names = []
    try:
        cat_transformer = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_transformer.get_feature_names_out(
            [
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
        )
        feature_names = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargesPerMonth"] + list(cat_features)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values(by="importance", ascending=False)
        importance_df["importance_normalized"] = importance_df["importance"] / importance_df["importance"].sum()
        st.write("### Model feature importance")
        st.pyplot(plot_feature_importance(importance_df, top_n=12))
    except Exception:
        st.warning("Feature importance is unavailable for this model artifact.")


if __name__ == "__main__":
    main()
