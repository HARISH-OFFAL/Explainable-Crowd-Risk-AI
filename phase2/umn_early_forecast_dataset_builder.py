from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_zone_training_features.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_6sec_early_forecast_features.csv"
)


# =========================================================
# SETTINGS
# =========================================================

FORECAST_HORIZON_SECONDS = 6

FPS = 30.0

FORECAST_HORIZON_FRAMES = int(
    FORECAST_HORIZON_SECONDS * FPS
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


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("UMN 6-SECOND EARLY FORECAST DATASET BUILDER")
print("================================================")

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(INPUT_CSV)

print(
    f"Input Zone Rows : {len(df)}"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = (
    FEATURE_COLUMNS
    +
    [
        "sequence_id",
        "scene_id",
        "scene_name",
        "zone_id",
        "window_id",
        "start_frame",
        "end_frame",
        "behavior_label_value",
        "transition_onset_frame",
        "training_split",
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
# NUMERIC CLEANING
# =========================================================

numeric_columns = (
    FEATURE_COLUMNS
    +
    [
        "window_id",
        "start_frame",
        "end_frame",
        "behavior_label_value",
        "transition_onset_frame",
    ]
)


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "sequence_id",
        "zone_id",
        "end_frame",
        "behavior_label_value",
        "transition_onset_frame",
        "training_split",
    ]
).copy()


# =========================================================
# ONLY CURRENTLY NORMAL WINDOWS
# =========================================================
#
# If a window is already ABNORMAL,
# it is not an early-warning sample.
#
# =========================================================

df = df[
    df[
        "behavior_label_value"
    ] == 0
].copy()


print(
    f"Current NORMAL Rows : {len(df)}"
)


# =========================================================
# TIME UNTIL TRANSITION
# =========================================================

df[
    "frames_until_transition"
] = (
    df[
        "transition_onset_frame"
    ]
    -
    df[
        "end_frame"
    ]
)


df[
    "seconds_until_transition"
] = (
    df[
        "frames_until_transition"
    ]
    /
    FPS
)


# =========================================================
# FUTURE TARGET
# =========================================================
#
# 1:
# abnormal transition occurs AFTER current window
# and within next 6 seconds.
#
# 0:
# transition is farther than 6 seconds away.
#
# =========================================================

df[
    "future_abnormal_within_6sec"
] = (
    (
        df[
            "frames_until_transition"
        ] > 0
    )
    &
    (
        df[
            "frames_until_transition"
        ]
        <= FORECAST_HORIZON_FRAMES
    )
).astype(int)


df[
    "future_forecast_label"
] = df[
    "future_abnormal_within_6sec"
].map(
    {
        0: "NO_TRANSITION_WITHIN_6SEC",
        1: "ABNORMAL_WITHIN_6SEC",
    }
)


# =========================================================
# OUTPUT COLUMNS
# =========================================================

metadata_columns = [
    "sequence_id",
    "scene_id",
    "scene_name",
    "zone_id",
    "window_id",
    "start_frame",
    "end_frame",
]


target_columns = [
    "frames_until_transition",
    "seconds_until_transition",
    "future_forecast_label",
    "future_abnormal_within_6sec",
    "training_split",
]


output_columns = (
    metadata_columns
    +
    FEATURE_COLUMNS
    +
    target_columns
)


output_df = df[
    output_columns
].copy()


# =========================================================
# SEQUENCE LEAKAGE CHECK
# =========================================================

sequence_split_counts = (
    output_df.groupby(
        "sequence_id"
    )[
        "training_split"
    ]
    .nunique()
)


if (
    sequence_split_counts > 1
).any():

    raise ValueError(
        "Sequence leakage detected."
    )


# =========================================================
# VERIFY CLASSES
# =========================================================

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = output_df[
        output_df[
            "training_split"
        ] == split
    ]


    if split_df.empty:

        raise ValueError(
            f"{split} has no samples."
        )


    labels = set(
        split_df[
            "future_abnormal_within_6sec"
        ].unique()
    )


    if labels != {0, 1}:

        raise ValueError(
            f"{split} does not contain "
            f"both forecast classes."
        )


# =========================================================
# SAVE
# =========================================================

output_df = output_df.sort_values(
    by=[
        "sequence_id",
        "window_id",
        "zone_id",
    ]
).reset_index(
    drop=True
)


output_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("----------------------------------------------")

print(
    f"Forecast Horizon : "
    f"{FORECAST_HORIZON_SECONDS} seconds"
)

print(
    f"Output Samples   : "
    f"{len(output_df)}"
)

print("----------------------------------------------")


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = output_df[
        output_df[
            "training_split"
        ] == split
    ]


    positive = int(
        (
            split_df[
                "future_abnormal_within_6sec"
            ] == 1
        ).sum()
    )


    negative = int(
        (
            split_df[
                "future_abnormal_within_6sec"
            ] == 0
        ).sum()
    )


    print(split)

    print(
        f"  ABNORMAL within 6 sec : "
        f"{positive}"
    )

    print(
        f"  No transition         : "
        f"{negative}"
    )

    print(
        f"  TOTAL                 : "
        f"{len(split_df)}"
    )

    print()


print("----------------------------------------------")

print(
    "Sequence Leakage Check : PASSED"
)

print(
    "Only currently NORMAL windows were used."
)

print(
    "Target = abnormal activity beginning "
    "within the next 6 seconds."
)

print(
    "No LOW/MEDIUM/HIGH thresholds were used."
)

print(
    f"Output : {OUTPUT_CSV}"
)

print("================================================")