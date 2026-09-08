from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "umn_temporal_6sec_forecast_features.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR
    / "umn_temporal_6sec_forecast_model.joblib"
)

METRICS_PATH = (
    MODEL_DIR
    / "umn_temporal_6sec_forecast_metrics.txt"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

print()
print("================================================")
print("TEMPORAL 6-SECOND FUTURE CROWD FORECAST MODEL")
print("================================================")

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset Samples : {len(df)}"
)


# =========================================================
# NON-FEATURE / METADATA COLUMNS
# =========================================================

EXCLUDED_COLUMNS = {
    "sequence_id",
    "scene_id",
    "scene_name",
    "current_window_id",
    "history_start_window_id",
    "history_end_window_id",
    "current_start_frame",
    "current_end_frame",
    "frames_until_transition",
    "seconds_until_transition",
    "future_abnormal_within_6sec",
    "future_forecast_label",
    "training_split",
}


# =========================================================
# AUTOMATIC FEATURE LIST
# =========================================================
#
# Dataset contains:
#
# H0_* = oldest history window
# H1_* = middle history window
# H2_* = current history window
# DELTA_* = temporal change
#
# =========================================================

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


if not FEATURE_COLUMNS:

    raise ValueError(
        "No model features found."
    )


print(
    f"Feature Count   : {len(FEATURE_COLUMNS)}"
)


TARGET = "future_abnormal_within_6sec"
SPLIT = "training_split"


# =========================================================
# VALIDATION
# =========================================================

required = [
    TARGET,
    SPLIT,
    "sequence_id",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# =========================================================
# SPLITS
# =========================================================

train_df = df[
    df[SPLIT] == "TRAIN"
].copy()

validation_df = df[
    df[SPLIT] == "VALIDATION"
].copy()

test_df = df[
    df[SPLIT] == "TEST"
].copy()


print("----------------------------------------------")

print(
    f"TRAIN      : {len(train_df)}"
)

print(
    f"VALIDATION : {len(validation_df)}"
)

print(
    f"TEST       : {len(test_df)}"
)


# =========================================================
# SEQUENCE LEAKAGE CHECK
# =========================================================

train_sequences = set(
    train_df[
        "sequence_id"
    ].unique()
)

validation_sequences = set(
    validation_df[
        "sequence_id"
    ].unique()
)

test_sequences = set(
    test_df[
        "sequence_id"
    ].unique()
)


if train_sequences & validation_sequences:

    raise ValueError(
        "TRAIN / VALIDATION sequence leakage."
    )

if train_sequences & test_sequences:

    raise ValueError(
        "TRAIN / TEST sequence leakage."
    )

if validation_sequences & test_sequences:

    raise ValueError(
        "VALIDATION / TEST sequence leakage."
    )


print(
    "Sequence Leakage Check : PASSED"
)


# =========================================================
# PREPARE X / Y
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
].copy()

y_train = train_df[
    TARGET
].astype(int)


X_validation = validation_df[
    FEATURE_COLUMNS
].copy()

y_validation = validation_df[
    TARGET
].astype(int)


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    TARGET
].astype(int)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def show_distribution(
    name,
    y
):

    negative = int(
        (y == 0).sum()
    )

    positive = int(
        (y == 1).sum()
    )

    print()
    print(name)

    print(
        f"  No transition    : {negative}"
    )

    print(
        f"  Abnormal <=6 sec : {positive}"
    )

    print(
        f"  Total            : {len(y)}"
    )


print("----------------------------------------------")
print("CLASS DISTRIBUTION")


show_distribution(
    "TRAIN",
    y_train
)

show_distribution(
    "VALIDATION",
    y_validation
)

show_distribution(
    "TEST",
    y_test
)


# =========================================================
# MODEL
# =========================================================
#
# Logistic Regression baseline:
#
# - interpretable
# - appropriate for small sample setting
# - class_weight handles imbalance
#
# =========================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
                C=0.5,
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

print()
print("----------------------------------------------")
print("TRAINING TEMPORAL FORECAST MODEL")
print("----------------------------------------------")


model.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    name,
    X,
    y
):

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]


    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )


    if len(
        np.unique(y)
    ) == 2:

        auc = roc_auc_score(
            y,
            probabilities
        )

    else:

        auc = float("nan")


    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    print()
    print(
        "================================================"
    )

    print(
        f"{name} RESULTS"
    )

    print(
        "================================================"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )


    print()
    print("Confusion Matrix")

    print(
        f"TN={tn}  FP={fp}"
    )

    print(
        f"FN={fn}  TP={tp}"
    )


    print()

    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                "NO_TRANSITION",
                "ABNORMAL_WITHIN_6SEC",
            ],
            zero_division=0,
        )
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# VALIDATION
# =========================================================

validation_metrics = evaluate(
    "VALIDATION",
    X_validation,
    y_validation
)


# =========================================================
# UNSEEN TEST
# =========================================================

test_metrics = evaluate(
    "UNSEEN TEST",
    X_test,
    y_test
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

classifier = model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]


importance = sorted(
    zip(
        FEATURE_COLUMNS,
        coefficients
    ),
    key=lambda item:
        abs(item[1]),
    reverse=True,
)


print()
print("================================================")
print("TOP TEMPORAL FORECAST FEATURES")
print("================================================")

print(
    "Positive -> higher probability of "
    "abnormal transition within 6 sec"
)

print(
    "Negative -> lower probability"
)

print()


for feature, coefficient in importance[:30]:

    print(
        f"{feature:55s} "
        f"{coefficient:+.6f}"
    )


# =========================================================
# SAVE MODEL PACKAGE
# =========================================================

model_package = {

    "model":
        model,

    "feature_columns":
        FEATURE_COLUMNS,

    "classes": {
        0:
            "NO_TRANSITION_WITHIN_6SEC",

        1:
            "ABNORMAL_WITHIN_6SEC",
    },

    "history_windows":
        3,

    "history_seconds":
        6,

    "forecast_horizon_seconds":
        6,

    "representation":
        (
            "scene-level A/B/C zone features "
            "with temporal history"
        ),

    "label_source":
        "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

    "official_umn_ground_truth_claim":
        False,

    "purpose":
        (
            "Temporal early-warning baseline "
            "for future abnormal crowd activity"
        ),
}


joblib.dump(
    model_package,
    MODEL_PATH
)


# =========================================================
# SAVE METRICS
# =========================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "UMN TEMPORAL 6-SECOND FORECAST MODEL\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        "History: 3 consecutive 2-second windows\n"
    )

    file.write(
        "Forecast horizon: next 6 seconds\n"
    )

    file.write(
        "Input representation: combined "
        "ZONE_A / ZONE_B / ZONE_C scene features\n\n"
    )


    file.write(
        "VALIDATION\n"
    )

    file.write(
        "----------\n"
    )

    for key, value in validation_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nUNSEEN TEST\n"
    )

    file.write(
        "-----------\n"
    )

    for key, value in test_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nTOP FEATURE COEFFICIENTS\n"
    )

    file.write(
        "------------------------\n"
    )

    for feature, coefficient in importance:

        file.write(
            f"{feature}: "
            f"{coefficient:+.6f}\n"
        )


# =========================================================
# FINAL
# =========================================================

print()
print("================================================")
print("TEMPORAL FORECAST MODEL COMPLETE")
print("================================================")

print(
    f"Model   : {MODEL_PATH}"
)

print(
    f"Metrics : {METRICS_PATH}"
)

print("----------------------------------------------")

print(
    "History used: previous 6 seconds."
)

print(
    "Forecast horizon: next 6 seconds."
)

print(
    "ZONE_A/B/C features are combined."
)

print(
    "No LOW/MEDIUM/HIGH thresholds were used."
)

print(
    "Because only 79 temporal samples exist, "
    "results must be treated as a small-data baseline."
)

print("================================================")