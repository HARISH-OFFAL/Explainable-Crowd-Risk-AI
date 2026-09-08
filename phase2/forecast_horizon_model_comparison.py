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
)
from sklearn.model_selection import LeaveOneGroupOut
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

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "umn_forecast_horizon_comparison.txt"
)


# =========================================================
# SETTINGS
# =========================================================

HORIZONS = [
    2,
    4,
    6,
    8,
]


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("UMN FORECAST HORIZON MODEL COMPARISON")
print("================================================")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print(f"Total Samples : {len(df)}")
print(
    f"Sequences     : "
    f"{df['sequence_id'].nunique()}"
)


# =========================================================
# IMPORTANT:
# HORIZON SELECTION MUST NOT USE FINAL TEST SEQUENCES
# =========================================================

development_df = df[
    df["training_split"].isin(
        [
            "TRAIN",
            "VALIDATION",
        ]
    )
].copy()

final_test_df = df[
    df["training_split"] == "TEST"
].copy()


print("----------------------------------------------")

print(
    f"Development Samples : "
    f"{len(development_df)}"
)

print(
    f"Final Test Samples  : "
    f"{len(final_test_df)}"
)

print(
    "Development Sequences :",
    sorted(
        development_df[
            "sequence_id"
        ].unique()
    )
)

print(
    "Held-Out Test Sequences:",
    sorted(
        final_test_df[
            "sequence_id"
        ].unique()
    )
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

print(
    f"Feature Count : "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# MODEL FACTORY
# =========================================================

def create_model():

    return Pipeline(
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
# HORIZON COMPARISON
# =========================================================

logo = LeaveOneGroupOut()

comparison_results = []


print()
print("================================================")
print("DEVELOPMENT-SEQUENCE CROSS VALIDATION")
print("================================================")


for horizon in HORIZONS:

    print()
    print("----------------------------------------------")
    print(
        f"FORECAST HORIZON : {horizon} SECONDS"
    )
    print("----------------------------------------------")


    horizon_df = development_df.copy()


    # =====================================================
    # CREATE TARGET
    # =====================================================
    #
    # Current sample is already NORMAL.
    #
    # Positive:
    # abnormal annotation onset occurs
    # within chosen future horizon.
    #
    # =====================================================

    target_column = (
        f"future_abnormal_within_{horizon}sec"
    )


    horizon_df[target_column] = (
        (
            horizon_df[
                "seconds_until_transition"
            ] > 0
        )
        &
        (
            horizon_df[
                "seconds_until_transition"
            ] <= horizon
        )
    ).astype(int)


    positive = int(
        horizon_df[
            target_column
        ].sum()
    )

    negative = int(
        len(horizon_df)
        - positive
    )


    print(
        f"Positive Samples : {positive}"
    )

    print(
        f"Negative Samples : {negative}"
    )


    X = horizon_df[
        FEATURE_COLUMNS
    ].copy()

    y = horizon_df[
        target_column
    ].astype(int).to_numpy()

    groups = horizon_df[
        "sequence_id"
    ].astype(str).to_numpy()


    all_true = []
    all_pred = []
    all_prob = []

    fold_auc = []
    fold_f1 = []


    for (
        train_index,
        validation_index,
    ) in logo.split(
        X,
        y,
        groups,
    ):

        X_train = X.iloc[
            train_index
        ]

        y_train = y[
            train_index
        ]

        X_validation = X.iloc[
            validation_index
        ]

        y_validation = y[
            validation_index
        ]


        # If training fold accidentally has only one class,
        # this horizon is not usable for this fold.
        if len(
            np.unique(y_train)
        ) < 2:

            continue


        model = create_model()

        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_validation
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]


        all_true.extend(
            y_validation.tolist()
        )

        all_pred.extend(
            predictions.tolist()
        )

        all_prob.extend(
            probabilities.tolist()
        )


        fold_f1.append(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        )


        if len(
            np.unique(y_validation)
        ) == 2:

            fold_auc.append(
                roc_auc_score(
                    y_validation,
                    probabilities,
                )
            )


    # =====================================================
    # POOLED DEVELOPMENT CV METRICS
    # =====================================================

    all_true = np.array(
        all_true
    )

    all_pred = np.array(
        all_pred
    )

    all_prob = np.array(
        all_prob
    )


    if len(all_true) == 0:

        print(
            "No valid folds."
        )

        continue


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


    if len(
        np.unique(all_true)
    ) == 2:

        auc = roc_auc_score(
            all_true,
            all_prob
        )

    else:

        auc = float("nan")


    mean_fold_auc = (
        float(
            np.mean(fold_auc)
        )
        if fold_auc
        else float("nan")
    )

    mean_fold_f1 = (
        float(
            np.mean(fold_f1)
        )
        if fold_f1
        else float("nan")
    )


    print()
    print("Development CV Results")

    print(
        f"Accuracy      : {accuracy:.4f}"
    )

    print(
        f"Precision     : {precision:.4f}"
    )

    print(
        f"Recall        : {recall:.4f}"
    )

    print(
        f"F1            : {f1:.4f}"
    )

    print(
        f"ROC-AUC       : {auc:.4f}"
    )

    print(
        f"Mean Fold F1  : {mean_fold_f1:.4f}"
    )

    print(
        f"Mean Fold AUC : {mean_fold_auc:.4f}"
    )


    comparison_results.append(
        {
            "horizon_seconds":
                horizon,

            "positive_samples":
                positive,

            "negative_samples":
                negative,

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

            "mean_fold_f1":
                mean_fold_f1,

            "mean_fold_auc":
                mean_fold_auc,
        }
    )


# =========================================================
# RESULTS TABLE
# =========================================================

result_df = pd.DataFrame(
    comparison_results
)


if result_df.empty:
    raise ValueError(
        "No horizon results generated."
    )


print()
print("================================================")
print("HORIZON COMPARISON SUMMARY")
print("================================================")


for _, row in result_df.iterrows():

    print(
        f"{int(row['horizon_seconds'])} sec"
    )

    print(
        f"  Positive Samples : "
        f"{int(row['positive_samples'])}"
    )

    print(
        f"  F1               : "
        f"{row['f1']:.4f}"
    )

    print(
        f"  Recall           : "
        f"{row['recall']:.4f}"
    )

    print(
        f"  ROC-AUC          : "
        f"{row['roc_auc']:.4f}"
    )

    print(
        f"  Mean Fold AUC    : "
        f"{row['mean_fold_auc']:.4f}"
    )

    print()


# =========================================================
# RANKING
# =========================================================
#
# This is exploratory model comparison.
#
# We primarily rank using ROC-AUC,
# then F1.
#
# TEST sequences are still untouched.
#
# =========================================================

ranked_df = result_df.sort_values(
    [
        "roc_auc",
        "f1",
    ],
    ascending=False,
).reset_index(drop=True)


best = ranked_df.iloc[0]


print("----------------------------------------------")

print(
    "BEST DEVELOPMENT HORIZON"
)

print(
    f"Horizon : "
    f"{int(best['horizon_seconds'])} sec"
)

print(
    f"ROC-AUC : "
    f"{best['roc_auc']:.4f}"
)

print(
    f"F1      : "
    f"{best['f1']:.4f}"
)

print("----------------------------------------------")

print(
    "IMPORTANT: final TEST sequences were NOT "
    "used to select this horizon."
)


# =========================================================
# SAVE REPORT
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "UMN FORECAST HORIZON MODEL COMPARISON\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        "Horizon selection uses only "
        "TRAIN + VALIDATION sequences.\n"
    )

    file.write(
        "Final TEST sequences remain held out.\n\n"
    )


    for _, row in result_df.iterrows():

        file.write(
            f"{int(row['horizon_seconds'])} SECOND HORIZON\n"
        )

        file.write(
            f"Positive samples: "
            f"{int(row['positive_samples'])}\n"
        )

        file.write(
            f"Negative samples: "
            f"{int(row['negative_samples'])}\n"
        )

        file.write(
            f"Accuracy: "
            f"{row['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{row['precision']:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{row['recall']:.4f}\n"
        )

        file.write(
            f"F1: "
            f"{row['f1']:.4f}\n"
        )

        file.write(
            f"ROC-AUC: "
            f"{row['roc_auc']:.4f}\n"
        )

        file.write(
            f"Mean Fold F1: "
            f"{row['mean_fold_f1']:.4f}\n"
        )

        file.write(
            f"Mean Fold ROC-AUC: "
            f"{row['mean_fold_auc']:.4f}\n\n"
        )


    file.write(
        "BEST DEVELOPMENT HORIZON\n"
    )

    file.write(
        "------------------------\n"
    )

    file.write(
        f"Horizon: "
        f"{int(best['horizon_seconds'])} seconds\n"
    )

    file.write(
        f"ROC-AUC: "
        f"{best['roc_auc']:.4f}\n"
    )

    file.write(
        f"F1: "
        f"{best['f1']:.4f}\n"
    )

    file.write(
        "\nFinal TEST sequences were not used "
        "during horizon selection.\n"
    )


print()
print("================================================")
print("HORIZON COMPARISON COMPLETE")
print("================================================")

print(
    f"Report : {OUTPUT_PATH}"
)

print(
    "Do NOT yet claim the selected horizon "
    "as final performance."
)

print(
    "Next step is evaluation on the untouched "
    "TEST sequences."
)

print("================================================")