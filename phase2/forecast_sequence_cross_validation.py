from pathlib import Path

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
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "umn_compact_temporal_6sec_forecast_features.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "umn_compact_temporal_sequence_cv_metrics.txt"
)


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("UMN TEMPORAL FORECAST SEQUENCE CROSS VALIDATION")
print("================================================")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Samples   : {len(df)}")
print(
    f"Sequences : "
    f"{df['sequence_id'].nunique()}"
)


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

print(
    f"Features  : {len(FEATURE_COLUMNS)}"
)


# =========================================================
# PREPARE DATA
# =========================================================

X = df[
    FEATURE_COLUMNS
].copy()

y = df[
    TARGET
].astype(int).to_numpy()

groups = df[
    "sequence_id"
].astype(str).to_numpy()


# =========================================================
# LEAVE ONE SEQUENCE OUT
# =========================================================

logo = LeaveOneGroupOut()

all_true = []
all_pred = []
all_prob = []

fold_results = []


print()
print("----------------------------------------------")
print("LEAVE-ONE-SEQUENCE-OUT EVALUATION")
print("----------------------------------------------")


for fold_number, (
    train_index,
    test_index,
) in enumerate(
    logo.split(
        X,
        y,
        groups,
    ),
    start=1,
):

    held_out_sequence = np.unique(
        groups[test_index]
    )[0]

    X_train = X.iloc[
        train_index
    ]

    y_train = y[
        train_index
    ]

    X_test = X.iloc[
        test_index
    ]

    y_test = y[
        test_index
    ]


    # =====================================================
    # MODEL CREATED FRESH FOR EACH SEQUENCE
    # =====================================================

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


    model.fit(
        X_train,
        y_train
    )


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
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )


    if len(
        np.unique(y_test)
    ) == 2:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    else:

        auc = float("nan")


    positive_count = int(
        (y_test == 1).sum()
    )

    negative_count = int(
        (y_test == 0).sum()
    )


    print()
    print(
        f"Fold {fold_number:02d} "
        f"- {held_out_sequence}"
    )

    print(
        f"Samples   : {len(y_test)}"
    )

    print(
        f"Positive  : {positive_count}"
    )

    print(
        f"Negative  : {negative_count}"
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
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )


    fold_results.append(
        {
            "sequence":
                held_out_sequence,

            "samples":
                len(y_test),

            "positive":
                positive_count,

            "negative":
                negative_count,

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
        }
    )


    all_true.extend(
        y_test.tolist()
    )

    all_pred.extend(
        predictions.tolist()
    )

    all_prob.extend(
        probabilities.tolist()
    )


# =========================================================
# POOLED OUT-OF-FOLD RESULTS
# =========================================================

all_true = np.array(
    all_true
)

all_pred = np.array(
    all_pred
)

all_prob = np.array(
    all_prob
)


accuracy = accuracy_score(
    all_true,
    all_pred
)

precision = precision_score(
    all_true,
    all_pred,
    zero_division=0,
)

recall = recall_score(
    all_true,
    all_pred,
    zero_division=0,
)

f1 = f1_score(
    all_true,
    all_pred,
    zero_division=0,
)

auc = roc_auc_score(
    all_true,
    all_prob
)


matrix = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = matrix.ravel()


# =========================================================
# FOLD STABILITY
# =========================================================

fold_df = pd.DataFrame(
    fold_results
)

mean_accuracy = fold_df[
    "accuracy"
].mean()

mean_recall = fold_df[
    "recall"
].mean()

mean_f1 = fold_df[
    "f1"
].mean()

valid_auc = fold_df[
    "roc_auc"
].dropna()

mean_auc = (
    valid_auc.mean()
    if not valid_auc.empty
    else float("nan")
)


# =========================================================
# FINAL RESULTS
# =========================================================

print()
print("================================================")
print("POOLED OUT-OF-SEQUENCE RESULTS")
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
print("----------------------------------------------")
print("MEAN PER-SEQUENCE PERFORMANCE")
print("----------------------------------------------")

print(
    f"Mean Accuracy : "
    f"{mean_accuracy:.4f}"
)

print(
    f"Mean Recall   : "
    f"{mean_recall:.4f}"
)

print(
    f"Mean F1       : "
    f"{mean_f1:.4f}"
)

print(
    f"Mean ROC-AUC  : "
    f"{mean_auc:.4f}"
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "UMN COMPACT TEMPORAL FORECAST\n"
    )

    file.write(
        "LEAVE-ONE-SEQUENCE-OUT CROSS VALIDATION\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        f"Samples: {len(df)}\n"
    )

    file.write(
        f"Sequences: {df['sequence_id'].nunique()}\n"
    )

    file.write(
        f"Features: {len(FEATURE_COLUMNS)}\n\n"
    )


    for result in fold_results:

        file.write(
            f"{result['sequence']}\n"
        )

        file.write(
            f"  Samples   : {result['samples']}\n"
        )

        file.write(
            f"  Positive  : {result['positive']}\n"
        )

        file.write(
            f"  Negative  : {result['negative']}\n"
        )

        file.write(
            f"  Accuracy  : {result['accuracy']:.4f}\n"
        )

        file.write(
            f"  Precision : {result['precision']:.4f}\n"
        )

        file.write(
            f"  Recall    : {result['recall']:.4f}\n"
        )

        file.write(
            f"  F1        : {result['f1']:.4f}\n"
        )

        file.write(
            f"  ROC-AUC   : {result['roc_auc']:.4f}\n\n"
        )


    file.write(
        "POOLED OUT-OF-SEQUENCE RESULTS\n"
    )

    file.write(
        "------------------------------\n"
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


print()
print("================================================")
print("CROSS VALIDATION COMPLETE")
print("================================================")

print(
    "Every sample was predicted while its "
    "entire sequence was unseen during training."
)

print(
    "No random frame/window split was used."
)

print(
    "This gives stronger generalization evidence "
    "than evaluating only one small fixed split."
)

print(
    f"Metrics saved to:\n{OUTPUT_PATH}"
)

print("================================================")