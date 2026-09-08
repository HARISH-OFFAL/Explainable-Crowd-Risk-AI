from pathlib import Path

import pandas as pd


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_zone_training_features.csv"
)


# =========================================================
# FORECAST HORIZONS TO ANALYZE
# =========================================================
#
# We are NOT selecting the final horizon here.
#
# We only measure how many genuine
# NORMAL -> future ABNORMAL transition samples
# are available for different lead times.
#
# =========================================================

HORIZONS_SECONDS = [
    2,
    4,
    6,
    8,
]


# UMN local video verified earlier at 30 FPS.
FPS = 30.0


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("UMN FORECAST HORIZON ANALYZER")
print("================================================")


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


print(
    f"Zone Rows : {len(df)}"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required = [
    "sequence_id",
    "zone_id",
    "start_frame",
    "end_frame",
    "behavior_label",
    "behavior_label_value",
    "transition_onset_frame",
    "training_split",
]


missing = [
    column
    for column in required
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns:\n"
        +
        "\n".join(missing)
    )


# =========================================================
# CLEAN
# =========================================================

numeric_columns = [
    "start_frame",
    "end_frame",
    "behavior_label_value",
    "transition_onset_frame",
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=numeric_columns
).copy()


# =========================================================
# IMPORTANT:
# EARLY FORECASTING MUST START FROM NORMAL STATE
# =========================================================
#
# If the current window is already ABNORMAL,
# predicting ABNORMAL later is not early prediction.
#
# Therefore only currently NORMAL windows are examined.
#
# =========================================================

normal_df = df[
    df["behavior_label_value"] == 0
].copy()


print(
    f"Current NORMAL Zone Rows : "
    f"{len(normal_df)}"
)

print("----------------------------------------------")


# =========================================================
# ANALYZE EACH HORIZON
# =========================================================

summary_rows = []


for horizon_seconds in HORIZONS_SECONDS:

    horizon_frames = int(
        round(
            horizon_seconds
            *
            FPS
        )
    )


    # -----------------------------------------------------
    # Frames remaining until abnormal annotation onset
    # -----------------------------------------------------

    frames_until_transition = (
        normal_df[
            "transition_onset_frame"
        ]
        -
        normal_df[
            "end_frame"
        ]
    )


    # -----------------------------------------------------
    # Positive future transition:
    #
    # transition must occur AFTER current window
    # but within selected future horizon.
    # -----------------------------------------------------

    future_positive = (
        (frames_until_transition > 0)
        &
        (
            frames_until_transition
            <= horizon_frames
        )
    )


    temp = normal_df.copy()

    temp[
        "future_transition"
    ] = future_positive.astype(int)


    temp[
        "frames_until_transition"
    ] = frames_until_transition


    # =====================================================
    # SPLIT SUMMARY
    # =====================================================

    print(
        f"FORECAST HORIZON : "
        f"{horizon_seconds} seconds"
    )

    print(
        f"Horizon Frames   : "
        f"{horizon_frames}"
    )


    total_positive = 0
    total_negative = 0


    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        split_df = temp[
            temp[
                "training_split"
            ] == split
        ]


        positive = int(
            (
                split_df[
                    "future_transition"
                ] == 1
            ).sum()
        )


        negative = int(
            (
                split_df[
                    "future_transition"
                ] == 0
            ).sum()
        )


        total_positive += positive
        total_negative += negative


        positive_sequences = sorted(
            split_df.loc[
                split_df[
                    "future_transition"
                ] == 1,
                "sequence_id"
            ]
            .unique()
            .tolist()
        )


        print(
            f"{split}"
        )

        print(
            f"  Transition YES : "
            f"{positive}"
        )

        print(
            f"  Transition NO  : "
            f"{negative}"
        )

        print(
            "  Positive Sequences : "
            +
            (
                ", ".join(
                    positive_sequences
                )
                if positive_sequences
                else "NONE"
            )
        )


    print(
        f"TOTAL Transition YES : "
        f"{total_positive}"
    )

    print(
        f"TOTAL Transition NO  : "
        f"{total_negative}"
    )

    print("----------------------------------------------")


    summary_rows.append(
        {
            "horizon_seconds":
                horizon_seconds,

            "horizon_frames":
                horizon_frames,

            "positive_samples":
                total_positive,

            "negative_samples":
                total_negative,

            "total_samples":
                (
                    total_positive
                    +
                    total_negative
                ),
        }
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("================================================")
print("HORIZON SUMMARY")
print("================================================")


for item in summary_rows:

    total = item[
        "total_samples"
    ]

    positive = item[
        "positive_samples"
    ]


    percentage = (
        positive
        /
        total
        *
        100
        if total > 0
        else 0
    )


    print(
        f"{item['horizon_seconds']:>2} sec"
        f" | Positive: "
        f"{positive:>3}"
        f" | Negative: "
        f"{item['negative_samples']:>3}"
        f" | Positive %: "
        f"{percentage:6.2f}%"
    )


print("----------------------------------------------")

print(
    "No forecasting model was trained."
)

print(
    "No forecast horizon was selected automatically."
)

print(
    "This step only measures available "
    "early-transition examples."
)

print(
    "Current windows are restricted to NORMAL "
    "crowd activity."
)

print("================================================")