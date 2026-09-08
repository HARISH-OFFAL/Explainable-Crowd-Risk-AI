from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent

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

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "umn_future_forecasting_xai_summary.txt"
)


print()
print("================================================")
print("FUTURE FORECASTING XAI SUMMARY")
print("================================================")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Metrics not found:\n{METRICS_PATH}"
    )


model_package = joblib.load(
    MODEL_PATH
)

model = model_package["model"]

feature_columns = model_package[
    "feature_columns"
]

classifier = model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]


importance = sorted(
    zip(
        feature_columns,
        coefficients
    ),
    key=lambda item:
        abs(item[1]),
    reverse=True,
)


print(
    "Forecast Horizon : "
    f"{model_package['forecast_horizon_seconds']} sec"
)

print(
    "History Used     : "
    f"{model_package['history_seconds']} sec"
)

print(
    f"Feature Count    : "
    f"{len(feature_columns)}"
)


print()
print("----------------------------------------------")
print("TOP FORECAST MODEL DRIVERS")
print("----------------------------------------------")


for feature, coefficient in importance[:10]:

    direction = (
        "toward future abnormal transition"
        if coefficient > 0
        else "away from future abnormal transition"
    )

    print(
        f"{feature:55s} "
        f"{coefficient:+.6f} "
        f"-> {direction}"
    )


summary = """
FUTURE FORECASTING INTERPRETATION

The forecasting model uses recent spatio-temporal crowd behaviour
to estimate whether an abnormal crowd transition may occur within
the next 6 seconds.

The model considers temporal crowd information such as person count,
movement activity, speed variation, acceleration behaviour, direction
consistency, and related zone-derived global summaries.

However, the final held-out evaluation showed insufficient
generalization to unseen UMN sequences.

FINAL HELD-OUT RESULT

Accuracy  : 0.4375
Precision : 0.0000
Recall    : 0.0000
F1 Score  : 0.0000
ROC-AUC   : 0.4667

Therefore:

- The forecasting component is retained as a research baseline.
- It must NOT be presented as an accurate real-world early-warning model.
- The result shows that more independent labelled crowd sequences are
  required for reliable future-risk forecasting.
- No LOW/MEDIUM/HIGH threshold was created from this result.
- Embedded red-text annotation onset in the local UMN video was used
  only as a proxy temporal label.
- It is not claimed as official UMN temporal ground truth.

XAI LIMITATION

Feature coefficients describe how the trained Logistic Regression model
uses each feature. They explain model behaviour, but they do not prove
that a feature causes abnormal crowd behaviour.
"""


print()
print("================================================")
print(summary.strip())
print("================================================")


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "UMN FUTURE FORECASTING XAI SUMMARY\n"
    )

    file.write(
        "==================================\n\n"
    )

    file.write(
        summary.strip()
    )

    file.write(
        "\n\nTOP MODEL DRIVERS\n"
    )

    file.write(
        "-----------------\n"
    )

    for feature, coefficient in importance:

        direction = (
            "toward future abnormal transition"
            if coefficient > 0
            else "away from future abnormal transition"
        )

        file.write(
            f"{feature}: "
            f"{coefficient:+.6f} "
            f"-> {direction}\n"
        )


print()
print(
    f"Output : {OUTPUT_PATH}"
)

print()
print("================================================")
print("FORECASTING XAI SUMMARY COMPLETE")
print("================================================")