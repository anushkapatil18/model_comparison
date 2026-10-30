# Dry Bean Model Explorer

This project compares five classifiers on measurements of dry bean samples and
provides a small Streamlit app for evaluating saved models on a labeled CSV.
Training and evaluation are separate: the app loads artifacts created by the
training script and does not fit models when a user uploads data.

## Data and attribution

The project uses the Dry Bean dataset, which contains 13,611 samples and 16
numeric measurements derived from bean images. The target column is `Class`,
with seven varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.

Dataset source: [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
The CSV files in this repository are data, not original project code; retain
the dataset's attribution and follow its terms when redistributing it.

## What the project does

`train.py` creates a stratified 80/20 train/test split using a fixed
random seed, encodes the class labels, and fits the following estimators:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

Standard scaling is applied to the four non-tree models. The script saves the
fitted estimators, scaler, and label encoder in `model/`, writes the held-out
rows to `test_data.csv`, and prints weighted Precision, Recall, and F1, along
with Accuracy, one-vs-rest weighted AUC, and Matthews correlation coefficient
(MCC).

## Run locally

From this directory, install the listed dependencies and train the models:

```bash
python -m pip install -r requirements.txt
python train.py
```

Start the interactive evaluation page with:

```bash
streamlit run app.py
```

Upload a labeled CSV containing the same 16 feature columns and a `Class`
column. Choose a classifier in the sidebar to see its scores, confusion matrix,
and per-class report. `test_data.csv` is provided as an example upload.

## Recorded comparison

The following are the results currently recorded for this project's held-out
split. Re-running training may produce slightly different values if the data
or library versions change.

| Classifier | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9329 | 0.8913 | 0.8917 | 0.8913 | 0.8690 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

On these recorded results, the SVM has the strongest scores across the listed
metrics. This is a comparison on one split, not a guarantee of performance on
other samples.

Project repository: [model_comparison](https://github.com/anushkapatil18/model_comparison).
