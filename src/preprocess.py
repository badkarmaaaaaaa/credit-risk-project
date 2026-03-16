import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
    "target"
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker"
]

NUMERICAL_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents"
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=" ", header=None, names=COLUMN_NAMES)
    # Remap target: 1 (good) -> 0, 2 (bad) -> 1
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train, X_test, y_train, y_test, scaler
