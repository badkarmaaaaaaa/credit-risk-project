import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

from preprocess import load_data, preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german.data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def print_metrics(name: str, y_test, y_pred, y_prob):
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  ROC AUC : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))


def plot_confusion_matrix(ax, y_test, y_pred, title: str):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"]
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


def plot_roc_curves(y_test, models: dict):
    plt.figure(figsize=(7, 5))
    for name, (y_prob, color) in models.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "roc_curves.png"), dpi=150)
    plt.savefig(os.path.join(ASSETS_DIR, "roc_curves.png"), dpi=150)
    plt.show()


def plot_feature_importance(rf_model, feature_names: list, top_n: int = 15):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="steelblue"
    )
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features – Random Forest", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=150)
    plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Loading data and models...")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, _ = preprocess(df)

    lr = load_model("logistic_regression")
    rf = load_model("random_forest")

    lr_prob = lr.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    print_metrics("Logistic Regression", y_test, lr_pred, lr_prob)
    print_metrics("Random Forest", y_test, rf_pred, rf_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(axes[0], y_test, lr_pred, "Logistic Regression")
    plot_confusion_matrix(axes[1], y_test, rf_pred, "Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrices.png"), dpi=150)
    plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrices.png"), dpi=150)
    plt.show()

    plot_roc_curves(y_test, {
        "Logistic Regression": (lr_prob, "royalblue"),
        "Random Forest": (rf_prob, "darkorange")
    })

    plot_feature_importance(rf, list(X_test.columns))
