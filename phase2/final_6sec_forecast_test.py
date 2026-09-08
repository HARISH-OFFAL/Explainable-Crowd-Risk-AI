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
    / "umn_compact_temporal_6sec_forecast_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "umn_final_6sec_forecast_model.joblib"
)

METRICS_PATH = (
    BASE_DIR
    / "models"
    / "umn_final_6sec_forecast_test_metrics.txt"
)


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("FINAL 6-SECOND FORECAST HELD-OUT TEST")
print("================================================")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Total Samples : {len(df)}")


# =========================================================
# FEATURES
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
    "future_forecast_label",
    "future_abnormal_within_6sec",
    "training_split",
}

FEATURE_COLUMNS = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

TARGET = "future_abnormal_within_6sec"

print(f"Feature Count : {len(FEATURE_COLUMNS)}")


# =========================================================
# DEVELOPMENT / FINAL TEST
# =========================================================

development_df = df[
    df["training_split"].isin(
        [
            "TRAIN",
            "VALIDATION",
        ]
    )
].copy()

test_df = df[
    df["training_split"] == "TEST"
].copy()


print("----------------------------------------------")

print(
    f"Development Samples : {len(development_df)}"
)

print(
    f"Final Test Samples  : {len(test_df)}"
)

print(
    "Development Sequences:",
    sorted(
        development_df[
            "sequence_id"
        ].unique()
    )
)

print(
    "Final Test Sequences:",
    sorted(
        test_df[
            "sequence_id"
        ].unique()
    )
)


# =========================================================
# LEAKAGE CHECK
# =========================================================

development_sequences = set(
    development_df[
        "sequence_id"
    ].unique()
)

test_sequences = set(
    test_df[
        "sequence_id"
    ].unique()
)

if development_sequences & test_sequences:
    raise ValueError(
        "Sequence leakage detected."
    )

print(
    "Sequence Leakage Check : PASSED"
)


# =========================================================
# PREPARE DATA
# =========================================================

X_train = development_df[
    FEATURE_COLUMNS
].copy()

y_train = development_df[
    TARGET
].astype(int)


X_test = test_df[
    FEATURE_COLUMNS
].copy()

y_test = test_df[
    TARGET
].astype(int)


print("----------------------------------------------")

print("DEVELOPMENT CLASS DISTRIBUTION")

print(
    f"Negative : {(y_train == 0).sum()}"
)

print(
    f"Positive : {(y_train == 1).sum()}"
)


print()
print("FINAL TEST CLASS DISTRIBUTION")

print(
    f"Negative : {(y_test == 0).sum()}"
)

print(
    f"Positive : {(y_test == 1).sum()}"
)


# =========================================================
# MODEL
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
                C=0.25,
            ),
        ),
    ]
)


# =========================================================
# TRAIN
# =========================================================

print()
print("----------------------------------------------")
print("TRAINING ON DEVELOPMENT SEQUENCES")
print("----------------------------------------------")

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# =========================================================
# FINAL HELD-OUT TEST
# =========================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)


matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()


# =========================================================
# DISPLAY
# =========================================================

print()
print("================================================")
print("FINAL UNTOUCHED TEST RESULTS")
print("================================================")

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
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "NO_TRANSITION",
            "ABNORMAL_WITHIN_6SEC",
        ],
        zero_division=0,
    )
)


# =========================================================
# PER-SEQUENCE TEST RESULTS
# =========================================================

print()
print("================================================")
print("PER TEST SEQUENCE RESULTS")
print("================================================")


test_output = test_df[
    [
        "sequence_id",
        "current_window_id",
        TARGET,
    ]
].copy()

test_output[
    "predicted_label"
] = predictions

test_output[
    "abnormal_probability"
] = probabilities


for sequence_id, group in test_output.groupby(
    "sequence_id"
):

    y_sequence = group[
        TARGET
    ].to_numpy()

    pred_sequence = group[
        "predicted_label"
    ].to_numpy()

    prob_sequence = group[
        "abnormal_probability"
    ].to_numpy()


    seq_accuracy = accuracy_score(
        y_sequence,
        pred_sequence
    )

    seq_precision = precision_score(
        y_sequence,
        pred_sequence,
        zero_division=0,
    )

    seq_recall = recall_score(
        y_sequence,
        pred_sequence,
        zero_division=0,
    )

    seq_f1 = f1_score(
        y_sequence,
        pred_sequence,
        zero_division=0,
    )


    if len(
        np.unique(y_sequence)
    ) == 2:

        seq_auc = roc_auc_score(
            y_sequence,
            prob_sequence
        )

    else:

        seq_auc = float("nan")


    print()
    print(sequence_id)

    print(
        f"  Samples   : {len(group)}"
    )

    print(
        f"  Accuracy  : {seq_accuracy:.4f}"
    )

    print(
        f"  Precision : {seq_precision:.4f}"
    )

    print(
        f"  Recall    : {seq_recall:.4f}"
    )

    print(
        f"  F1        : {seq_f1:.4f}"
    )

    print(
        f"  ROC-AUC   : {seq_auc:.4f}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

model_package = {

    "model":
        model,

    "feature_columns":
        FEATURE_COLUMNS,

    "history_seconds":
        6,

    "forecast_horizon_seconds":
        6,

    "classes": {
        0:
            "NO_TRANSITION_WITHIN_6SEC",

        1:
            "ABNORMAL_WITHIN_6SEC",
    },

    "training_sequences":
        sorted(
            development_sequences
        ),

    "held_out_test_sequences":
        sorted(
            test_sequences
        ),

    "label_source":
        "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

    "official_umn_ground_truth_claim":
        False,

    "model_status":
        "RESEARCH_BASELINE",
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
    encoding="utf-8",
) as file:

    file.write(
        "FINAL 6-SECOND FORECAST "
        "HELD-OUT TEST\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        "Horizon selected using development "
        "sequences only.\n"
    )

    file.write(
        "SEQ_08 and SEQ_11 were not used "
        "during horizon selection.\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {auc:.4f}\n"
    )

    file.write(
        f"TN: {tn}\n"
    )

    file.write(
        f"FP: {fp}\n"
    )

    file.write(
        f"FN: {fn}\n"
    )

    file.write(
        f"TP: {tp}\n"
    )

    file.write(
        "\nImportant limitation:\n"
    )

    file.write(
        "The dataset contains only 11 reconstructed "
        "UMN sequences and 79 temporal samples. "
        "The embedded red-text onset is used as a "
        "local proxy label and is not claimed as "
        "official UMN temporal ground truth.\n"
    )


# =========================================================
# FINAL
# =========================================================

print()
print("================================================")
print("FINAL TEST COMPLETE")
print("================================================")

print(
    f"Model   : {MODEL_PATH}"
)

print(
    f"Metrics : {METRICS_PATH}"
)

print(
    "The result is a held-out research baseline, "
    "not a final real-world crowd-risk accuracy claim."
)

print("================================================")