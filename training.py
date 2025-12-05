# train_phishing_model.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump


LEGIT_PATH = "datasets/structured_legitimate_data.csv"
PHISH_PATH = "datasets/structured_phishing_data.csv"
RESULTS_PATH = "results/model_performance.csv"
BEST_MODEL_PATH = "models/best_phishing_model.joblib"


def load_and_label_data():
    """Load two CSVs and add labels: 0 = legit, 1 = phishing."""
    if not os.path.exists(LEGIT_PATH):
        raise FileNotFoundError(f"Missing file: {LEGIT_PATH}")
    if not os.path.exists(PHISH_PATH):
        raise FileNotFoundError(f"Missing file: {PHISH_PATH}")

    legit = pd.read_csv(LEGIT_PATH)
    phish = pd.read_csv(PHISH_PATH)

    legit["label"] = 0
    phish["label"] = 1

    df = pd.concat([legit, phish], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def build_feature_target(df: pd.DataFrame):
    """Split dataframe into X (features) and y (labels)."""
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in dataframe")

    X = df.drop(columns=["URL", "label"])
    y = df["label"]
    return X, y


def get_models():
    """Return a dict of candidate models."""
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "LinearSVM": make_pipeline(StandardScaler(), LinearSVC(max_iter=10000)),
        "GaussianNB": GaussianNB(),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
        ),
    }


def evaluate_models(X, y):
    """Train and evaluate all models; return metrics table and best model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    records = []
    best_model_name = None
    best_accuracy = -1.0
    best_model = None

    for name, model in models.items():
        print(f"\n🔹 Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        records.append(
            {
                "model": name,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    results_df = pd.DataFrame.from_records(records).sort_values(
        by="accuracy", ascending=False
    ).reset_index(drop=True)

    print("\n🏆 Best model:", best_model_name, f"(accuracy = {best_accuracy:.4f})")
    return results_df, best_model, best_model_name


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    df = load_and_label_data()
    print(f"Total samples: {len(df)} (legit={sum(df.label==0)}, phish={sum(df.label==1)})")

    X, y = build_feature_target(df)

    print("\nTraining and evaluating models...")
    results_df, best_model, best_name = evaluate_models(X, y)

    print("\nSaving metrics and best model...")
    results_df.to_csv(RESULTS_PATH, index=False)
    dump(best_model, BEST_MODEL_PATH)

    print(f"Results saved to   : {RESULTS_PATH}")
    print(f"Best model ({best_name}) saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
