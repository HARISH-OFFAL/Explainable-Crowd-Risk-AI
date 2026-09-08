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
    / "umn_zone_training_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "umn_zone_abnormal_classifier_v2.joblib"
)

METRICS_PATH = (
    MODEL_DIR
    / "umn_zone_abnormal_metrics_v2.txt"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# MODEL FEATURES - V2
# =========================================================
#
# IMPORTANT:
#
# Camera / perspective sensitive features removed:
#
#   avg_bbox_area_ratio
#   max_bbox_area_ratio
#   avg_detection_confidence
#   valid_motion_rows
#
# Model now focuses on crowd behaviour and
# spatio-temporal crowd characteristics.
#
# =========================================================

FEATURE_COLUMNS = [

    # -----------------------------------------------------
    # CROWD OCCUPANCY / DENSITY
    # -----------------------------------------------------

    "zone_presence_ratio",

    "unique_person_count",

    "avg_person_count_per_frame",

    "max_person_count_per_frame",


    # -----------------------------------------------------
    # MOVEMENT
    # -----------------------------------------------------

    "avg_movement_distance_norm",


    # -----------------------------------------------------
    # SPEED
    # -----------------------------------------------------

    "avg_speed_norm_s",

    "max_speed_norm_s",

    "speed_std_norm_s",


    # -----------------------------------------------------
    # ACCELERATION
    # -----------------------------------------------------

    "avg_acceleration_norm_s2",

    "acceleration_std_norm_s2",


    # -----------------------------------------------------
    # DIRECTION / CROWD ORGANIZATION
    # -----------------------------------------------------

    "avg_direction_change_deg",

    "direction_consistency",


    # -----------------------------------------------------
    # MOVEMENT ACTIVITY
    # -----------------------------------------------------

    "movement_activity_ratio",
]


TARGET = "behavior_label_value"

SPLIT = "training_split"


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("================================================")
print("CURRENT CROWD RISK MODEL - V2")
print("================================================")

print(
    "Camera-sensitive features removed."
)

print(
    "Using crowd density + motion + "
    "spatio-temporal behaviour features."
)

print("================================================")


if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n"
        f"{DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


print()
print(
    f"Dataset Rows : {len(df)}"
)


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = (
    FEATURE_COLUMNS
    + [
        TARGET,
        SPLIT,
        "sequence_id",
        "zone_id",
    ]
)


missing_columns = [

    column

    for column
    in required_columns

    if column
    not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


print(
    f"Model Features : "
    f"{len(FEATURE_COLUMNS)}"
)


print()
print(
    "Features used:"
)


for feature in FEATURE_COLUMNS:

    print(
        f"  - {feature}"
    )


# =========================================================
# SPLIT DATA
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


print()
print("----------------------------------------------")

print(
    f"TRAIN      : "
    f"{len(train_df)}"
)

print(
    f"VALIDATION : "
    f"{len(validation_df)}"
)

print(
    f"TEST       : "
    f"{len(test_df)}"
)


# =========================================================
# CHECK EMPTY SPLITS
# =========================================================

if train_df.empty:

    raise ValueError(
        "TRAIN split is empty."
    )


if validation_df.empty:

    raise ValueError(
        "VALIDATION split is empty."
    )


if test_df.empty:

    raise ValueError(
        "TEST split is empty."
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


if (
    train_sequences
    & validation_sequences
):

    raise ValueError(
        "TRAIN / VALIDATION "
        "sequence leakage."
    )


if (
    train_sequences
    & test_sequences
):

    raise ValueError(
        "TRAIN / TEST "
        "sequence leakage."
    )


if (
    validation_sequences
    & test_sequences
):

    raise ValueError(
        "VALIDATION / TEST "
        "sequence leakage."
    )


print()
print(
    "Sequence Leakage Check : PASSED"
)


print(
    f"TRAIN sequences      : "
    f"{sorted(train_sequences)}"
)

print(
    f"VALIDATION sequences : "
    f"{sorted(validation_sequences)}"
)

print(
    f"TEST sequences       : "
    f"{sorted(test_sequences)}"
)


# =========================================================
# PREPARE MODEL DATA
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
# CHECK TARGET CLASSES
# =========================================================

if y_train.nunique() < 2:

    raise ValueError(
        "TRAIN split does not contain "
        "both NORMAL and ABNORMAL classes."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def show_distribution(
    name,
    y
):

    normal = int(
        (y == 0).sum()
    )

    abnormal = int(
        (y == 1).sum()
    )

    print()
    print(name)

    print(
        f"  NORMAL   : {normal}"
    )

    print(
        f"  ABNORMAL : {abnormal}"
    )

    print(
        f"  TOTAL    : {len(y)}"
    )


print()
print("----------------------------------------------")
print("CLASS DISTRIBUTION")
print("----------------------------------------------")


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
# Logistic Regression is retained because:
#
# 1. Dataset is small.
# 2. Explainability is important.
# 3. Coefficients support XAI.
#
# C=0.5 adds slightly stronger regularization
# than default C=1.0 to reduce overfitting.
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

                C=0.5,
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

print()
print("================================================")
print("TRAINING V2 MODEL")
print("================================================")


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

    predictions = (
        model.predict(
            X
        )
    )


    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )


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
        np.unique(
            y
        )
    ) == 2:

        auc = roc_auc_score(
            y,
            probabilities
        )

    else:

        auc = float(
            "nan"
        )


    matrix = confusion_matrix(
        y,
        predictions,
        labels=[
            0,
            1
        ]
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
    print(
        "Confusion Matrix"
    )


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

            labels=[
                0,
                1
            ],

            target_names=[
                "NORMAL",
                "ABNORMAL"
            ],

            zero_division=0,
        )
    )


    return {

        "accuracy":
            float(
                accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "auc":
            float(
                auc
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),
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
# FEATURE COEFFICIENTS / XAI
# =========================================================

classifier = (
    model.named_steps[
        "classifier"
    ]
)


coefficients = (
    classifier.coef_[0]
)


importance = sorted(

    zip(
        FEATURE_COLUMNS,
        coefficients
    ),

    key=lambda item:
        abs(
            item[1]
        ),

    reverse=True
)


print()
print("================================================")
print("V2 FEATURE COEFFICIENTS")
print("================================================")


print(
    "Positive -> pushes model toward ABNORMAL"
)

print(
    "Negative -> pushes model toward NORMAL"
)

print()


for (
    feature,
    coefficient
) in importance:

    print(
        f"{feature:35s} "
        f"{coefficient:+.6f}"
    )


# =========================================================
# GENERALIZATION CHECK
# =========================================================

print()
print("================================================")
print("MODEL REVIEW")
print("================================================")


print(
    f"Validation F1 : "
    f"{validation_metrics['f1']:.4f}"
)

print(
    f"Test F1       : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"Test Recall   : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test Precision: "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test ROC-AUC  : "
    f"{test_metrics['auc']:.4f}"
)


# =========================================================
# SAVE MODEL PACKAGE
# =========================================================

package = {

    "model":
        model,

    "feature_columns":
        FEATURE_COLUMNS,

    "classes": {
        0:
            "NORMAL",

        1:
            "ABNORMAL",
    },

    "model_type":
        (
            "Zone-Level Logistic Regression "
            "V2 - Behaviour Focused"
        ),

    "window_seconds":
        2.0,

    "label_source":
        (
            "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET"
        ),

    "official_umn_ground_truth_claim":
        False,

    "camera_sensitive_features_removed": [
        "avg_bbox_area_ratio",
        "max_bbox_area_ratio",
        "avg_detection_confidence",
        "valid_motion_rows",
    ],

    "behaviour_feature_groups": [
        "zone occupancy",
        "crowd count",
        "movement",
        "speed",
        "speed variability",
        "acceleration",
        "direction change",
        "direction consistency",
        "movement activity",
    ],

    "purpose":
        (
            "Behaviour-focused zone-level "
            "NORMAL vs ABNORMAL crowd "
            "activity research baseline"
        ),

    "model_status":
        "RESEARCH_BASELINE",
}


joblib.dump(
    package,
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
        "UMN CURRENT CROWD ACTIVITY MODEL V2\n"
    )

    file.write(
        "===================================\n\n"
    )


    file.write(
        "MODEL PURPOSE\n"
    )

    file.write(
        "-------------\n"
    )

    file.write(
        "Behaviour-focused zone-level "
        "NORMAL vs ABNORMAL crowd "
        "activity research baseline.\n\n"
    )


    file.write(
        "CAMERA-SENSITIVE FEATURES REMOVED\n"
    )

    file.write(
        "---------------------------------\n"
    )

    file.write(
        "avg_bbox_area_ratio\n"
    )

    file.write(
        "max_bbox_area_ratio\n"
    )

    file.write(
        "avg_detection_confidence\n"
    )

    file.write(
        "valid_motion_rows\n\n"
    )


    file.write(
        "FEATURES USED\n"
    )

    file.write(
        "-------------\n"
    )


    for feature in FEATURE_COLUMNS:

        file.write(
            f"{feature}\n"
        )


    file.write(
        "\nVALIDATION\n"
    )

    file.write(
        "----------\n"
    )


    for (
        key,
        value
    ) in validation_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nUNSEEN TEST\n"
    )

    file.write(
        "-----------\n"
    )


    for (
        key,
        value
    ) in test_metrics.items():

        file.write(
            f"{key}: {value}\n"
        )


    file.write(
        "\nFEATURE COEFFICIENTS\n"
    )

    file.write(
        "--------------------\n"
    )


    for (
        feature,
        coefficient
    ) in importance:

        file.write(
            f"{feature}: "
            f"{coefficient:+.6f}\n"
        )


    file.write(
        "\nIMPORTANT LIMITATION\n"
    )

    file.write(
        "--------------------\n"
    )

    file.write(
        "The UMN labels used here represent "
        "NORMAL versus ABNORMAL crowd activity.\n"
    )

    file.write(
        "This model is not claimed as an "
        "official LOW/MEDIUM/HIGH stampede-risk "
        "classifier.\n"
    )

    file.write(
        "Cross-scene real-world validation "
        "is still required.\n"
    )


# =========================================================
# FINAL
# =========================================================

print()
print("================================================")
print("CURRENT RISK MODEL V2 COMPLETE")
print("================================================")


print(
    f"Model   : "
    f"{MODEL_PATH}"
)

print(
    f"Metrics : "
    f"{METRICS_PATH}"
)


print()
print(
    "No videos were processed."
)

print(
    "No YOLO feature extraction was run."
)

print(
    "Existing zone dataset only was used."
)

print()
print(
    "IMPORTANT:"
)

print(
    "Do not accept this model only from Accuracy."
)

print(
    "Check TEST Precision, Recall, F1 and ROC-AUC."
)

print("================================================")