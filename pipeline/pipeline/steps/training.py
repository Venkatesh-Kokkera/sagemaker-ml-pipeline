import os
import json
import argparse
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import shap
import joblib

def load_data(train_dir: str):
    """Load training data from SageMaker input channel."""
    train_path = os.path.join(train_dir, "train.csv")
    df = pd.read_csv(train_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    print(f"Training data shape: {X.shape}")
    return X, y

def train_xgboost(X_train, y_train, params: dict):
    """Train XGBoost model."""
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y) -> dict:
    """Evaluate model and return metrics."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y, preds), 4),
        "f1_score": round(f1_score(y, preds, average="weighted"), 4),
        "auc_roc": round(roc_auc_score(y, proba), 4)
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print("\n📋 Classification Report:")
    print(classification_report(y, preds))

    return metrics

def compute_shap(model, X_train):
    """Compute SHAP values for explainability."""
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[:100])
    print("✅ SHAP values computed")
    return shap_values

def save_model(model, output_dir: str):
    """Save model to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

def train(args):
    """Main training function."""
    # Load data
    X_train, y_train = load_data(args.train)

    # MLflow tracking
    mlflow.set_experiment("sagemaker-ml-pipeline")

    with mlflow.start_run():
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample
        }
        mlflow.log_params(params)

        # Train model
        model = train_xgboost(X_train, y_train, params)

        # Evaluate
        metrics = evaluate_model(model, X_train, y_train)
        mlflow.log_metrics(metrics)

        # SHAP explainability
        compute_shap(model, X_train)

        # Save model
        save_model(model, args.model_dir)

        # Save metrics
        metrics_path = os.path.join(
            args.model_dir, "metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump({"metrics": metrics}, f)

        print("✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=os.environ.get("SM_CHANNEL_TRAIN", "./data")
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SM_MODEL_DIR", "./model")
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    args = parser.parse_args()
    train(args)
