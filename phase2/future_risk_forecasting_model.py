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
    / "umn_6sec_early_forecast_features.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR
    / "umn_6sec_future_forecast_model.joblib"
)

METRICS_PATH = (
    MODEL_DIR
    / "umn_6sec_future_forecast_metrics.txt"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "zone_presence_ratio",
    "unique_person_count",
    "avg_person_count_per_frame",
    "max_person_count_per_frame",
    "avg_bbox_area_ratio",
    "max_bbox_area_ratio",
    "avg_detection_confidence",
    "valid_motion_rows",
    "avg_movement_distance_norm",
    "avg_speed_norm_s",
    "max_speed_norm_s",
    "speed_std_norm_s",
    "avg_acceleration_norm_s2",
    "acceleration_std_norm_s2",
    "avg_direction_change_deg",
    "direction_consistency",
    "movement_activity_ratio",
]

TARGET = "future_abnormal_within_6sec"
SPLIT = "training_split"


# =========================================================
# LOAD DATA
# =========================================================

print()
print("================================================")
print("6-SECOND FUTURE CROWD ACTIVITY FORECAST MODEL")
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
# VALIDATE COLUMNS
# =========================================================

required_columns = (
    FEATURE_COLUMNS
    +
    [
        TARGET,
        SPLIT,
        "sequence_id",
    ]
)


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(missing)
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
# DISTRIBUTION
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
        f"  No transition    : "
        f"{negative}"
    )

    print(
        f"  Abnormal <=6 sec : "
        f"{positive}"
    )

    print(
        f"  Total            : "
        f"{len(y)}"
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
# Explainable baseline:
#
# Logistic Regression
#
# class_weight="balanced" handles class imbalance.
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
                max_iter=3000,
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
print("TRAINING FORECAST MODEL")
print("----------------------------------------------")


model.fit(
    X_train,
    y_train
)


print(
    "Forecast model training completed."
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
# FEATURE COEFFICIENTS
# =========================================================

classifier = model.named_steps[
    "classifier"
]


coefficients = classifier.coef_[
    0
]


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
print("FORECAST FEATURE COEFFICIENTS")
print("================================================")

print(
    "Positive -> increases probability of "
    "abnormal activity within 6 sec."
)

print(
    "Negative -> decreases probability."
)

print()


for feature, coefficient in importance:

    print(
        f"{feature:32s} "
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

    "classes": {
        0:
            "NO_TRANSITION_WITHIN_6SEC",

        1:
            "ABNORMAL_WITHIN_6SEC",
    },

    "forecast_horizon_seconds":
        6,

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
            "Early forecasting of abnormal "
            "crowd activity within 6 seconds"
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
        "UMN 6-SECOND FUTURE CROWD FORECAST MODEL\n"
    )

    file.write(
        "========================================\n\n"
    )


    file.write(
        "TARGET\n"
    )

    file.write(
        "------\n"
    )

    file.write(
        "Predict whether abnormal crowd activity "
        "will begin within the next 6 seconds.\n\n"
    )


    file.write(
        "VALIDATION METRICS\n"
    )

    file.write(
        "------------------\n"
    )

    for key, value in validation_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nUNSEEN TEST METRICS\n"
    )

    file.write(
        "-------------------\n"
    )

    for key, value in test_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nFEATURE COEFFICIENTS\n"
    )

    file.write(
        "--------------------\n"
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
print("FUTURE FORECAST MODEL COMPLETE")
print("================================================")

print(
    f"Model   : {MODEL_PATH}"
)

print(
    f"Metrics : {METRICS_PATH}"
)

print("----------------------------------------------")

print(
    "Forecast horizon: 6 seconds."
)

print(
    "Input samples are currently NORMAL "
    "zone windows only."
)

print(
    "Target: abnormal crowd activity "
    "beginning within 6 seconds."
)

print(
    "No LOW/MEDIUM/HIGH thresholds "
    "were invented."
)

print(
    "Review unseen-test results before "
    "accepting the forecasting baseline."
)

print("================================================")