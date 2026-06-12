from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def train_best_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = {
        "n_estimators": [100, 150],
        "max_depth": [6, 10],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(model, grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }
    return best_model, metrics


def save_model_pipeline(pipeline: Pipeline, model, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "model": model}, save_path)


def load_model_pipeline(save_path: Path):
    return joblib.load(save_path)
