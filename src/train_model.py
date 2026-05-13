from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "HR_capstone_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"


def load_data():
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    df = df.rename(columns={
        "Work_accident": "work_accident",
        "average_montly_hours": "average_monthly_hours",
        "time_spend_company": "company_years",
        "Department": "department"
    })

    df = df.drop_duplicates(keep="first")
    return df


def prepare_features(df):
    df_model = df.copy()

    df_model["salary"] = (
        df_model["salary"]
        .astype("category")
        .cat.set_categories(["low", "medium", "high"])
        .cat.codes
    )

    df_model = pd.get_dummies(df_model, drop_first=False)

    y = df_model["left"]
    X = df_model.drop("left", axis=1)

    return X, y


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=0)

    cv_params = {
        "max_depth": [3, 5, None],
        "max_features": [1.0],
        "max_samples": [0.7, 1.0],
        "min_samples_leaf": [1, 2, 3],
        "min_samples_split": [2, 3, 4],
        "n_estimators": [300, 500]
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=cv_params,
        scoring=scoring,
        cv=4,
        refit="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def evaluate_model(model, X_test, y_test):
    preds = model.best_estimator_.predict(X_test)
    probs = model.best_estimator_.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs)
    }

    return results


def save_model(model):
    MODEL_DIR.mkdir(exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def main():
    df = load_data()
    df = clean_data(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=0
    )

    model = train_random_forest(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print("Model evaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    save_model(model)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
