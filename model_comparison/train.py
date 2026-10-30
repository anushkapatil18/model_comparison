from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "dataset.csv"
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"

MODEL_SPECS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Naive Bayes": (GaussianNB(), True),
    "Support Vector Machine": (SVC(probability=True, random_state=42), True),
}


def score_model(model, features, labels):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(
            labels, probabilities, multi_class="ovr", average="weighted"
        ),
        "Precision": precision_score(labels, predictions, average="weighted"),
        "Recall": recall_score(labels, predictions, average="weighted"),
        "F1": f1_score(labels, predictions, average="weighted"),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    if "Class" not in data.columns:
        raise ValueError("dataset.csv must include a 'Class' target column.")

    features = data.drop(columns="Class")
    labels = LabelEncoder()
    encoded_labels = labels.fit_transform(data["Class"])
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(labels, MODEL_DIR / "label_encoder.joblib")
    pd.concat(
        [x_test.reset_index(drop=True),
         pd.Series(labels.inverse_transform(y_test), name="Class")],
        axis=1,
    ).to_csv(TEST_DATA_PATH, index=False)

    comparison = []
    for model_name, (model, uses_scaled_features) in MODEL_SPECS.items():
        training_features = x_train_scaled if uses_scaled_features else x_train
        evaluation_features = x_test_scaled if uses_scaled_features else x_test
        model.fit(training_features, y_train)

        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        comparison.append(
            {"Model": model_name, **score_model(model, evaluation_features, y_test)}
        )

    results = pd.DataFrame(comparison).set_index("Model").round(4)
    print("Test-set comparison (80/20 stratified split):")
    print(results.to_string())
    print(f"\nSaved model files to {MODEL_DIR}")
    print(f"Saved held-out examples to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
