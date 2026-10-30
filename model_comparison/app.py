from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_OPTIONS = {
    "Logistic Regression": ("logistic_regression.joblib", True),
    "Decision Tree": ("decision_tree.joblib", False),
    "KNN": ("knn.joblib", True),
    "Naive Bayes": ("naive_bayes.joblib", True),
    "Support Vector Machine": ("support_vector_machine.joblib", True),
}

st.set_page_config(page_title="Dry Bean Model Explorer", layout="wide")
st.title("Dry Bean Model Explorer")
st.write(
    "Evaluate a saved classifier against labeled bean measurements. "
    "The app loads existing model artifacts; it does not train a model."
)

with st.sidebar:
    st.header("Evaluation setup")
    uploaded_file = st.file_uploader("Choose a labeled CSV file", type="csv")
    selected_model = st.selectbox("Classifier", list(MODEL_OPTIONS))

if uploaded_file is None:
    st.info("Upload a CSV with the bean measurements and a `Class` column to begin.")
    st.stop()

data = pd.read_csv(uploaded_file)
if "Class" not in data.columns:
    st.error("The CSV is missing the required `Class` column.")
    st.stop()

scaler_path = MODEL_DIR / "scaler.joblib"
encoder_path = MODEL_DIR / "label_encoder.joblib"
model_filename, uses_scaling = MODEL_OPTIONS[selected_model]
model_path = MODEL_DIR / model_filename
missing_files = [
    path.name for path in (scaler_path, encoder_path, model_path) if not path.is_file()
]
if missing_files:
    st.error(
        "Required model artifacts are missing: "
        + ", ".join(missing_files)
        + ". Run `python train_and_save.py` from this project directory."
    )
    st.stop()

scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)
model = joblib.load(model_path)

feature_names = list(scaler.feature_names_in_)
missing_features = [name for name in feature_names if name not in data.columns]
unexpected_features = [
    name for name in data.columns if name not in feature_names and name != "Class"
]
if missing_features or unexpected_features:
    messages = []
    if missing_features:
        messages.append("Missing feature columns: " + ", ".join(missing_features))
    if unexpected_features:
        messages.append("Unrecognized columns: " + ", ".join(unexpected_features))
    st.error(" ".join(messages))
    st.stop()

features = data.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
if features.isna().any().any():
    st.error("Feature columns must contain numeric values and no blank cells.")
    st.stop()

try:
    true_labels = label_encoder.transform(data["Class"])
except ValueError as error:
    st.error(f"The CSV contains a class label not present during training: {error}")
    st.stop()

evaluation_features = scaler.transform(features) if uses_scaling else features
predicted_labels = model.predict(evaluation_features)
class_probabilities = model.predict_proba(evaluation_features)

metrics = {
    "Accuracy": accuracy_score(true_labels, predicted_labels),
    "AUC": roc_auc_score(
        true_labels,
        class_probabilities,
        multi_class="ovr",
        average="weighted",
    ),
    "Precision": precision_score(
        true_labels, predicted_labels, average="weighted"
    ),
    "Recall": recall_score(true_labels, predicted_labels, average="weighted"),
    "F1 score": f1_score(true_labels, predicted_labels, average="weighted"),
    "MCC": matthews_corrcoef(true_labels, predicted_labels),
}

st.subheader(f"Evaluation results: {selected_model}")
metric_columns = st.columns(len(metrics))
for column, (metric, value) in zip(metric_columns, metrics.items()):
    column.metric(metric, f"{value:.4f}")

st.divider()
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(true_labels, predicted_labels)
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    st.pyplot(figure)
    plt.close(figure)

with right_column:
    st.subheader("Per-class report")
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))

with st.expander("Preview uploaded rows"):
    st.dataframe(data.head())
