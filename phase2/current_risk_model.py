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
    confusion_matrix,
    classification_report,
    roc_auc_score,
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
    / "umn_labeled_movement_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "umn_current_abnormal_classifier.joblib"
)

METRICS_PATH = (
    MODEL_DIR
    / "umn_current_abnormal_metrics.txt"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FEATURES
# =========================================================
#
# Only crowd movement / spatio-temporal information is used.
#
# IMPORTANT:
# We intentionally DO NOT use:
#
# frame_number
# timestamp_sec
# sequence_id
# scene_id
# scene_name
# transition_onset_frame
# behavior_label
# training_split
#
# because they can leak label/sequence information.
#
# =========================================================

FEATURE_COLUMNS = [
    "normalized_x",
    "normalized_y",
    "bbox_width",
    "bbox_height",
    "bbox_area_ratio",
    "detection_confidence",
    "motion_valid",
    "movement_distance_norm",
    "speed_norm_s",
    "acceleration_norm_s2",
    "direction_change_deg",
    "direction_sin",
    "direction_cos",
    "consecutive_frames",
]

TARGET_COLUMN = "behavior_label_value"

SPLIT_COLUMN = "training_split"


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("==============================================")
print("UMN CURRENT ABNORMAL-CROWD CLASSIFIER")
print("==============================================")

print(f"Dataset : {DATASET_PATH}")
print("----------------------------------------------")


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


print(
    f"Total Rows : "
    f"{len(df)}"
)


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = (
    FEATURE_COLUMNS
    +
    [
        TARGET_COLUMN,
        SPLIT_COLUMN,
        "sequence_id",
    ]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# =========================================================
# SPLIT DATA
# =========================================================

train_df = df[
    df[SPLIT_COLUMN] == "TRAIN"
].copy()

validation_df = df[
    df[SPLIT_COLUMN] == "VALIDATION"
].copy()

test_df = df[
    df[SPLIT_COLUMN] == "TEST"
].copy()


print("----------------------------------------------")

print(
    f"TRAIN Rows      : "
    f"{len(train_df)}"
)

print(
    f"VALIDATION Rows : "
    f"{len(validation_df)}"
)

print(
    f"TEST Rows       : "
    f"{len(test_df)}"
)


if (
    len(train_df) == 0
    or len(validation_df) == 0
    or len(test_df) == 0
):

    raise ValueError(
        "TRAIN / VALIDATION / TEST "
        "must all contain rows."
    )


# =========================================================
# CHECK SEQUENCE LEAKAGE
# =========================================================

train_sequences = set(
    train_df["sequence_id"].unique()
)

validation_sequences = set(
    validation_df["sequence_id"].unique()
)

test_sequences = set(
    test_df["sequence_id"].unique()
)


if train_sequences & validation_sequences:

    raise ValueError(
        "Sequence leakage detected between "
        "TRAIN and VALIDATION."
    )


if train_sequences & test_sequences:

    raise ValueError(
        "Sequence leakage detected between "
        "TRAIN and TEST."
    )


if validation_sequences & test_sequences:

    raise ValueError(
        "Sequence leakage detected between "
        "VALIDATION and TEST."
    )


print()
print("Sequence Leakage Check : PASSED")

print(
    "TRAIN      : "
    +
    ", ".join(
        sorted(train_sequences)
    )
)

print(
    "VALIDATION : "
    +
    ", ".join(
        sorted(validation_sequences)
    )
)

print(
    "TEST       : "
    +
    ", ".join(
        sorted(test_sequences)
    )
)


# =========================================================
# PREPARE X / Y
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
].copy()

y_train = train_df[
    TARGET_COLUMN
].astype(int)


X_validation = validation_df[
    FEATURE_COLUMNS
].copy()

y_validation = validation_df[
    TARGET_COLUMN
].astype(int)


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    TARGET_COLUMN
].astype(int)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_distribution(
    name,
    y
):

    normal = int(
        (y == 0).sum()
    )

    abnormal = int(
        (y == 1).sum()
    )

    total = len(y)

    print()
    print(name)

    print(
        f"  NORMAL   : "
        f"{normal}"
    )

    print(
        f"  ABNORMAL : "
        f"{abnormal}"
    )

    print(
        f"  TOTAL    : "
        f"{total}"
    )

    if total > 0:

        abnormal_percentage = (
            abnormal
            /
            total
            *
            100
        )

        print(
            f"  ABNORMAL % : "
            f"{abnormal_percentage:.2f}%"
        )


print("----------------------------------------------")
print("CLASS DISTRIBUTION")
print("----------------------------------------------")

print_distribution(
    "TRAIN",
    y_train
)

print_distribution(
    "VALIDATION",
    y_validation
)

print_distribution(
    "TEST",
    y_test
)


# =========================================================
# MODEL PIPELINE
# =========================================================
#
# SimpleImputer:
#   Protects against missing numeric values.
#
# StandardScaler:
#   Normalizes feature scales.
#
# LogisticRegression:
#   Explainable baseline classifier.
#
# class_weight="balanced":
#   Helps handle strong NORMAL / ABNORMAL imbalance.
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
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

print()
print("----------------------------------------------")
print("TRAINING MODEL")
print("----------------------------------------------")

model.fit(
    X_train,
    y_train
)

print(
    "Model training completed."
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(
    split_name,
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


    if len(np.unique(y)) == 2:

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


    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=[
            "NORMAL",
            "ABNORMAL"
        ],
        zero_division=0
    )


    print()
    print(
        "=============================================="
    )

    print(
        f"{split_name} RESULTS"
    )

    print(
        "=============================================="
    )


    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{auc:.4f}"
    )


    print()
    print("Confusion Matrix")

    print(
        "                 Predicted"
    )

    print(
        "              NORMAL  ABNORMAL"
    )

    print(
        f"Actual NORMAL  "
        f"{tn:6d}  {fp:8d}"
    )

    print(
        f"Actual ABNORMAL"
        f"{fn:6d}  {tp:8d}"
    )


    print()
    print("Classification Report")

    print(
        report
    )


    return {
        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "roc_auc":
            auc,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "report":
            report,
    }


# =========================================================
# VALIDATION EVALUATION
# =========================================================

validation_metrics = evaluate_model(
    "VALIDATION",
    X_validation,
    y_validation
)


# =========================================================
# TEST EVALUATION
# =========================================================
#
# TEST is evaluated separately.
# It was not used for model fitting.
#
# =========================================================

test_metrics = evaluate_model(
    "TEST",
    X_test,
    y_test
)


# =========================================================
# FEATURE COEFFICIENTS
# =========================================================

classifier = model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]


feature_importance = sorted(
    zip(
        FEATURE_COLUMNS,
        coefficients
    ),
    key=lambda item:
        abs(item[1]),
    reverse=True
)


print()
print("==============================================")
print("MODEL FEATURE COEFFICIENTS")
print("==============================================")

print(
    "Positive coefficient -> pushes prediction "
    "toward ABNORMAL."
)

print(
    "Negative coefficient -> pushes prediction "
    "toward NORMAL."
)

print()


for feature, coefficient in feature_importance:

    print(
        f"{feature:28s} "
        f"{coefficient:+.6f}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

model_package = {
    "model":
        model,

    "feature_columns":
        FEATURE_COLUMNS,

    "target":
        TARGET_COLUMN,

    "class_names": {
        0: "NORMAL",
        1: "ABNORMAL",
    },

    "model_type":
        "LogisticRegression",

    "class_weight":
        "balanced",

    "label_source":
        "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

    "official_umn_ground_truth_claim":
        False,

    "purpose":
        (
            "UMN normal-vs-abnormal crowd "
            "activity baseline classifier"
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
        "UMN CURRENT ABNORMAL-CROWD CLASSIFIER\n"
    )

    file.write(
        "=====================================\n\n"
    )


    file.write(
        "IMPORTANT\n"
    )

    file.write(
        "---------\n"
    )

    file.write(
        "This model predicts NORMAL vs ABNORMAL "
        "crowd activity.\n"
    )

    file.write(
        "It must not be described as official "
        "LOW/HIGH crowd-risk ground truth.\n"
    )

    file.write(
        "Labels were reconstructed from embedded "
        "red-text onset in the local UMN video.\n\n"
    )


    file.write(
        "VALIDATION METRICS\n"
    )

    file.write(
        "------------------\n"
    )

    file.write(
        f"Accuracy  : "
        f"{validation_metrics['accuracy']:.6f}\n"
    )

    file.write(
        f"Precision : "
        f"{validation_metrics['precision']:.6f}\n"
    )

    file.write(
        f"Recall    : "
        f"{validation_metrics['recall']:.6f}\n"
    )

    file.write(
        f"F1 Score  : "
        f"{validation_metrics['f1']:.6f}\n"
    )

    file.write(
        f"ROC-AUC   : "
        f"{validation_metrics['roc_auc']:.6f}\n\n"
    )


    file.write(
        "TEST METRICS\n"
    )

    file.write(
        "------------\n"
    )

    file.write(
        f"Accuracy  : "
        f"{test_metrics['accuracy']:.6f}\n"
    )

    file.write(
        f"Precision : "
        f"{test_metrics['precision']:.6f}\n"
    )

    file.write(
        f"Recall    : "
        f"{test_metrics['recall']:.6f}\n"
    )

    file.write(
        f"F1 Score  : "
        f"{test_metrics['f1']:.6f}\n"
    )

    file.write(
        f"ROC-AUC   : "
        f"{test_metrics['roc_auc']:.6f}\n\n"
    )


    file.write(
        "TEST CONFUSION MATRIX\n"
    )

    file.write(
        "---------------------\n"
    )

    file.write(
        f"TN: {test_metrics['tn']}\n"
    )

    file.write(
        f"FP: {test_metrics['fp']}\n"
    )

    file.write(
        f"FN: {test_metrics['fn']}\n"
    )

    file.write(
        f"TP: {test_metrics['tp']}\n\n"
    )


    file.write(
        "FEATURE COEFFICIENTS\n"
    )

    file.write(
        "--------------------\n"
    )


    for feature, coefficient in feature_importance:

        file.write(
            f"{feature}: "
            f"{coefficient:+.6f}\n"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==============================================")
print("MODEL DEVELOPMENT COMPLETE")
print("==============================================")

print(
    f"Model   : {MODEL_PATH}"
)

print(
    f"Metrics : {METRICS_PATH}"
)

print("----------------------------------------------")

print(
    "This is a NORMAL-vs-ABNORMAL crowd "
    "activity baseline."
)

print(
    "No LOW/MEDIUM/HIGH risk thresholds "
    "were invented."
)

print(
    "Do not claim final project accuracy "
    "until the evaluation results are reviewed."
)

print("==============================================")