import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.catboost
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import os

# ====================
# Parse CLI arguments
# ====================
parser = argparse.ArgumentParser()
parser.add_argument("--border_count", type=int, default=83)
parser.add_argument("--random_strength", type=float, default=7.853702081679818e-08)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--depth", type=int, default=7)
parser.add_argument("--learning_rate", type=float, default=0.09867443636192799)
parser.add_argument("--l2_leaf_reg", type=float, default=9.894379624044467)
parser.add_argument("--verbose", type=int, default=0)
parser.add_argument("--eval_metric", type=str, default="Accuracy")
parser.add_argument("--iterations", type=int, default=232)
parser.add_argument("--dataset", type=str, default="preprocessed_dataset")
args = parser.parse_args()

# ============================
# Load preprocessed dataset
# ============================
X_train = pd.read_csv("preprocessing/dataset/X_train.csv")
X_test = pd.read_csv("preprocessing/dataset/X_test.csv")
y_train = pd.read_csv("preprocessing/dataset/y_train.csv")
y_test = pd.read_csv("preprocessing/dataset/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# ============================
# Set MLflow Experiment
# ============================
mlflow.set_experiment("Catboost Diabetic Prediction")

# ============================
# Run MLflow logging
# ============================

# NOTE: Jangan pakai mlflow.start_run() kalau script ini dijalankan via `mlflow run`
if mlflow.active_run() is None:
    mlflow.start_run(run_name="Final CatBoost Model")  # fallback untuk mode lokal

# Convert argparse args to dict
best_params = {
    "border_count": args.border_count,
    "random_strength": args.random_strength,
    "random_state": args.random_state,
    "depth": args.depth,
    "learning_rate": args.learning_rate,
    "l2_leaf_reg": args.l2_leaf_reg,
    "verbose": args.verbose,
    "eval_metric": args.eval_metric,
    "iterations": args.iterations
}

model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
def classification_metrics(y_true, y_pred, prefix=""):
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, average="weighted"),
        f"{prefix}recall": recall_score(y_true, y_pred, average="weighted"),
        f"{prefix}f1_score": f1_score(y_true, y_pred, average="weighted")
    }

train_metrics = classification_metrics(y_train, y_pred_train, "train_")
test_metrics = classification_metrics(y_test, y_pred_test, "test_")

# Log parameters and metrics
mlflow.log_params(best_params)
mlflow.log_metrics(train_metrics)
mlflow.log_metrics(test_metrics)

# Save & log test metrics
test_metrics_df = pd.DataFrame({
    "metric": list(test_metrics.keys()),
    "value": list(test_metrics.values())
})
test_metrics_df.to_csv("test_classification_metrics.csv", index=False)
mlflow.log_artifact("test_classification_metrics.csv")

# Log model
mlflow.catboost.log_model(model, "model", input_example=X_test.iloc[:1])

# Plot feature importance
feature_importance = model.get_feature_importance()
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title("Feature Importance - CatBoost")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("catboost_feature_importance.png")
mlflow.log_artifact("catboost_feature_importance.png")

# Tutup run jika dimulai secara manual
if mlflow.active_run() is not None:
    mlflow.end_run()
