import pandas as pd
import numpy as np
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import warnings

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
parser.add_argument("--dataset", type=str, default="preprocessing/dataset")
args = parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Load dataset
    X_train = pd.read_csv(f"{args.dataset}/X_train.csv")
    X_test = pd.read_csv(f"{args.dataset}/X_test.csv")
    y_train = pd.read_csv(f"{args.dataset}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{args.dataset}/y_test.csv").values.ravel()

    with mlflow.start_run():
        # Log params
        mlflow.log_param("border_count", args.border_count)
        mlflow.log_param("random_strength", args.random_strength)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("depth", args.depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("l2_leaf_reg", args.l2_leaf_reg)
        mlflow.log_param("verbose", args.verbose)
        mlflow.log_param("eval_metric", args.eval_metric)
        mlflow.log_param("iterations", args.iterations)

        # Train model
        model = CatBoostClassifier(
            border_count=args.border_count,
            random_strength=args.random_strength,
            random_state=args.random_state,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            verbose=args.verbose,
            eval_metric=args.eval_metric,
            iterations=args.iterations
        )
        model.fit(X_train, y_train)

        # Log model
        mlflow.catboost.log_model(model, artifact_path="model", input_example=X_test.iloc[:1])

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics function
        def get_metrics(y_true, y_pred):
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1_score": f1_score(y_true, y_pred, average="weighted"),
            }

        train_metrics = get_metrics(y_train, y_train_pred)
        test_metrics = get_metrics(y_test, y_test_pred)

        # Log training metrics
        for key, val in train_metrics.items():
            mlflow.log_metric(f"train_{key}", val)

        # Log testing metrics
        for key, val in test_metrics.items():
            mlflow.log_metric(f"test_{key}", val)

        # Save test metrics to CSV
        df_test_metrics = pd.DataFrame({
            "metric": list(test_metrics.keys()),
            "value": list(test_metrics.values())
        })
        df_test_metrics.to_csv("test_classification_metrics.csv", index=False)
        mlflow.log_artifact("test_classification_metrics.csv")
