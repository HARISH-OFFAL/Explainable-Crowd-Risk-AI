from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "datasets"
    / "umn_zone_training_features.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "umn_zone_abnormal_classifier.joblib"
)

OUTPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "umn_current_risk_xai_results.csv"
)


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("CURRENT RISK XAI")
print("================================================")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

model_package = joblib.load(
    MODEL_PATH
)

model = model_package["model"]

feature_columns = model_package[
    "feature_columns"
]


print(f"Rows     : {len(df)}")
print(f"Features : {len(feature_columns)}")


# =========================================================
# MODEL COMPONENTS
# =========================================================

imputer = model.named_steps[
    "imputer"
]

scaler = model.named_steps[
    "scaler"
]

classifier = model.named_steps[
    "classifier"
]


coefficients = classifier.coef_[0]


# =========================================================
# SAMPLE SELECTION
# =========================================================
#
# We explain TEST rows first.
# If TEST is unavailable, use all rows.
#
# =========================================================

if "training_split" in df.columns:

    explain_df = df[
        df["training_split"] == "TEST"
    ].copy()

else:

    explain_df = df.copy()


if explain_df.empty:
    explain_df = df.copy()


X = explain_df[
    feature_columns
].copy()


# =========================================================
# TRANSFORM SAME WAY AS MODEL
# =========================================================

X_imputed = imputer.transform(
    X
)

X_scaled = scaler.transform(
    X_imputed
)


# =========================================================
# PREDICTIONS
# =========================================================

probabilities = model.predict_proba(
    X
)[:, 1]

predictions = model.predict(
    X
)


# =========================================================
# FEATURE CONTRIBUTIONS
# =========================================================
#
# Logistic Regression:
#
# contribution =
# standardized feature value * coefficient
#
# Positive contribution:
# pushes toward ABNORMAL
#
# Negative contribution:
# pushes toward NORMAL
#
# =========================================================

contributions = (
    X_scaled
    * coefficients
)


results = []


for row_index in range(
    len(explain_df)
):

    row = explain_df.iloc[
        row_index
    ]


    row_contributions = contributions[
        row_index
    ]


    contribution_pairs = list(
        zip(
            feature_columns,
            row_contributions,
        )
    )


    positive_features = sorted(
        contribution_pairs,
        key=lambda item:
            item[1],
        reverse=True,
    )


    negative_features = sorted(
        contribution_pairs,
        key=lambda item:
            item[1],
    )


    top_positive = [
        item
        for item in positive_features
        if item[1] > 0
    ][:3]


    top_negative = [
        item
        for item in negative_features
        if item[1] < 0
    ][:3]


    positive_text = "; ".join(
        [
            f"{feature} ({value:+.3f})"
            for feature, value
            in top_positive
        ]
    )


    negative_text = "; ".join(
        [
            f"{feature} ({value:+.3f})"
            for feature, value
            in top_negative
        ]
    )


    probability = float(
        probabilities[
            row_index
        ]
    )


    prediction = int(
        predictions[
            row_index
        ]
    )


    predicted_label = (
        "ABNORMAL"
        if prediction == 1
        else "NORMAL"
    )


    if prediction == 1:

        explanation = (
            "Prediction is pushed toward ABNORMAL "
            f"mainly by: {positive_text}"
        )

    else:

        explanation = (
            "Prediction is pushed toward NORMAL "
            f"mainly by: {negative_text}"
        )


    results.append(
        {
            "sequence_id":
                row.get(
                    "sequence_id",
                    ""
                ),

            "zone_id":
                row.get(
                    "zone_id",
                    ""
                ),

            "window_id":
                row.get(
                    "window_id",
                    ""
                ),

            "actual_label":
                row.get(
                    "behavior_label",
                    ""
                ),

            "predicted_label":
                predicted_label,

            "abnormal_probability":
                round(
                    probability,
                    6
                ),

            "top_abnormal_drivers":
                positive_text,

            "top_normal_drivers":
                negative_text,

            "xai_explanation":
                explanation,
        }
    )


# =========================================================
# SAVE
# =========================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY EXAMPLES
# =========================================================

print()
print("----------------------------------------------")
print("EXAMPLE EXPLANATIONS")
print("----------------------------------------------")


for _, row in result_df.head(
    10
).iterrows():

    print()

    print(
        f"Sequence : "
        f"{row['sequence_id']}"
    )

    print(
        f"Zone     : "
        f"{row['zone_id']}"
    )

    print(
        f"Window   : "
        f"{row['window_id']}"
    )

    print(
        f"Actual   : "
        f"{row['actual_label']}"
    )

    print(
        f"Predicted: "
        f"{row['predicted_label']}"
    )

    print(
        f"Probability: "
        f"{row['abnormal_probability']:.4f}"
    )

    print(
        f"Why      : "
        f"{row['xai_explanation']}"
    )


# =========================================================
# GLOBAL MODEL IMPORTANCE
# =========================================================

importance = sorted(
    zip(
        feature_columns,
        coefficients,
    ),
    key=lambda item:
        abs(
            item[1]
        ),
    reverse=True,
)


print()
print("================================================")
print("GLOBAL MODEL FEATURE IMPORTANCE")
print("================================================")

for feature, coefficient in importance:

    direction = (
        "ABNORMAL"
        if coefficient > 0
        else "NORMAL"
    )

    print(
        f"{feature:45s} "
        f"{coefficient:+.6f} "
        f"-> {direction}"
    )


# =========================================================
# FINAL
# =========================================================

print()
print("================================================")
print("CURRENT RISK XAI COMPLETE")
print("================================================")

print(
    f"Output : {OUTPUT_PATH}"
)

print(
    "Positive contribution means the feature "
    "pushes the prediction toward ABNORMAL."
)

print(
    "Negative contribution means the feature "
    "pushes the prediction toward NORMAL."
)

print(
    "This explains the model decision; "
    "it does not prove causation."
)

print("================================================")