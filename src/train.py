import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german.data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Bad credit rate (train): {y_train.mean():.2%}\n")

    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train, y_train)
    save_model(lr, "logistic_regression")

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    save_model(rf, "random_forest")

    save_model({"scaler": scaler, "columns": list(X_train.columns)}, "metadata")

    print("\nTraining complete. Run evaluate.py to see results.")
