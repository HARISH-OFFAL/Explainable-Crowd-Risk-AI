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
    / "umn_future_forecast_features.csv"
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================
#
# Current zone window = 2 seconds.
#
# Forecast target:
# next consecutive 2-second window of SAME:
#
# - sequence
# - zone
#
# Therefore initial forecast horizon = one window ahead.
#
# =========================================================

FORECAST_STEPS = 1


# =========================================================
# LOAD DATA
# =========================================================

print()
print("================================================")
print("UMN FUTURE FORECAST DATASET BUILDER")
print("================================================")


if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(
    INPUT_CSV
)


print(
    f"Input Zone Rows : {len(df)}"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "sequence_id",
    "scene_id",
    "scene_name",
    "zone_id",
    "window_id",
    "window_seconds",
    "start_frame",
    "end_frame",
    "behavior_label",
    "behavior_label_value",
    "training_split",
]


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
# CLEAN TYPES
# =========================================================

df["window_id"] = pd.to_numeric(
    df["window_id"],
    errors="coerce"
)

df["window_seconds"] = pd.to_numeric(
    df["window_seconds"],
    errors="coerce"
)

df["behavior_label_value"] = pd.to_numeric(
    df["behavior_label_value"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "sequence_id",
        "zone_id",
        "window_id",
        "window_seconds",
        "behavior_label_value",
        "training_split",
    ]
).copy()


df["window_id"] = df[
    "window_id"
].astype(int)

df["behavior_label_value"] = df[
    "behavior_label_value"
].astype(int)


# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    by=[
        "sequence_id",
        "zone_id",
        "window_id",
    ]
).reset_index(
    drop=True
)


# =========================================================
# FEATURE COLUMNS
# =========================================================
#
# These are current-time zone features.
#
# We intentionally do NOT use:
#
# behavior_label
# future label
# sequence ID
# frame number
# transition onset
# training split
#
# as predictive feature inputs.
#
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


for column in FEATURE_COLUMNS:

    if column not in df.columns:

        raise ValueError(
            f"Missing forecasting feature: {column}"
        )


# =========================================================
# BUILD LOOKUP
# =========================================================

lookup = {}


for index, row in df.iterrows():

    key = (
        row["sequence_id"],
        row["zone_id"],
        int(row["window_id"]),
    )

    lookup[key] = row


# =========================================================
# BUILD CURRENT → FUTURE PAIRS
# =========================================================

forecast_rows = []

skipped_no_future = 0
skipped_split_mismatch = 0


for _, current in df.iterrows():

    sequence_id = current[
        "sequence_id"
    ]

    zone_id = current[
        "zone_id"
    ]

    current_window = int(
        current["window_id"]
    )

    future_window = (
        current_window
        +
        FORECAST_STEPS
    )


    future_key = (
        sequence_id,
        zone_id,
        future_window,
    )


    # -----------------------------------------------------
    # No consecutive future window
    # -----------------------------------------------------

    if future_key not in lookup:

        skipped_no_future += 1
        continue


    future = lookup[
        future_key
    ]


    # -----------------------------------------------------
    # Safety:
    # same reconstructed sequence must stay in same split
    # -----------------------------------------------------

    if (
        current["training_split"]
        != future["training_split"]
    ):

        skipped_split_mismatch += 1
        continue


    window_seconds = float(
        current["window_seconds"]
    )

    forecast_horizon_sec = (
        window_seconds
        *
        FORECAST_STEPS
    )


    # =====================================================
    # BASE METADATA
    # =====================================================

    output = {
        "sequence_id":
            sequence_id,

        "scene_id":
            current["scene_id"],

        "scene_name":
            current["scene_name"],

        "zone_id":
            zone_id,

        "current_window_id":
            current_window,

        "future_window_id":
            future_window,

        "current_start_frame":
            int(current["start_frame"]),

        "current_end_frame":
            int(current["end_frame"]),

        "future_start_frame":
            int(future["start_frame"]),

        "future_end_frame":
            int(future["end_frame"]),

        "forecast_horizon_sec":
            forecast_horizon_sec,

        "current_behavior_label":
            current["behavior_label"],

        "current_behavior_value":
            int(
                current[
                    "behavior_label_value"
                ]
            ),

        "future_behavior_label":
            future["behavior_label"],

        "future_behavior_value":
            int(
                future[
                    "behavior_label_value"
                ]
            ),

        "training_split":
            current["training_split"],

        "label_source":
            "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

        "official_umn_ground_truth_claim":
            "NO",
    }


    # =====================================================
    # CURRENT-TIME FEATURES
    # =====================================================

    for feature in FEATURE_COLUMNS:

        output[
            feature
        ] = current[
            feature
        ]


    # =====================================================
    # TRANSITION TYPE
    # =====================================================

    current_value = int(
        current[
            "behavior_label_value"
        ]
    )

    future_value = int(
        future[
            "behavior_label_value"
        ]
    )


    if (
        current_value == 0
        and future_value == 0
    ):

        transition_type = (
            "NORMAL_TO_NORMAL"
        )


    elif (
        current_value == 0
        and future_value == 1
    ):

        transition_type = (
            "NORMAL_TO_ABNORMAL"
        )


    elif (
        current_value == 1
        and future_value == 1
    ):

        transition_type = (
            "ABNORMAL_TO_ABNORMAL"
        )


    else:

        transition_type = (
            "ABNORMAL_TO_NORMAL"
        )


    output[
        "transition_type"
    ] = transition_type


    forecast_rows.append(
        output
    )


# =========================================================
# CREATE DATAFRAME
# =========================================================

forecast_df = pd.DataFrame(
    forecast_rows
)


if forecast_df.empty:

    raise ValueError(
        "No current-to-future forecasting "
        "pairs were generated."
    )


# =========================================================
# SEQUENCE LEAKAGE CHECK
# =========================================================

sequence_split_counts = (
    forecast_df.groupby(
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
# VALIDATE BOTH FUTURE CLASSES
# =========================================================

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = forecast_df[
        forecast_df[
            "training_split"
        ] == split
    ]


    if split_df.empty:

        raise ValueError(
            f"{split} contains no "
            f"forecast samples."
        )


    labels = set(
        split_df[
            "future_behavior_label"
        ].unique()
    )


    if not {
        "NORMAL",
        "ABNORMAL"
    }.issubset(labels):

        print(
            f"WARNING: {split} does not "
            f"contain both future classes."
        )


# =========================================================
# SAVE
# =========================================================

forecast_df = forecast_df.sort_values(
    by=[
        "sequence_id",
        "zone_id",
        "current_window_id",
    ]
).reset_index(
    drop=True
)


forecast_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("----------------------------------------------")

print(
    f"Forecast Pairs        : "
    f"{len(forecast_df)}"
)

print(
    f"No Future Window Skip : "
    f"{skipped_no_future}"
)

print(
    f"Split Mismatch Skip   : "
    f"{skipped_split_mismatch}"
)

print("----------------------------------------------")


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = forecast_df[
        forecast_df[
            "training_split"
        ] == split
    ]


    normal_future = int(
        (
            split_df[
                "future_behavior_label"
            ] == "NORMAL"
        ).sum()
    )


    abnormal_future = int(
        (
            split_df[
                "future_behavior_label"
            ] == "ABNORMAL"
        ).sum()
    )


    early_transition = int(
        (
            split_df[
                "transition_type"
            ]
            == "NORMAL_TO_ABNORMAL"
        ).sum()
    )


    print(split)

    print(
        f"  Future NORMAL       : "
        f"{normal_future}"
    )

    print(
        f"  Future ABNORMAL     : "
        f"{abnormal_future}"
    )

    print(
        f"  NORMAL -> ABNORMAL  : "
        f"{early_transition}"
    )

    print(
        f"  TOTAL               : "
        f"{len(split_df)}"
    )

    print()


print("----------------------------------------------")

print(
    "Sequence Leakage Check : PASSED"
)

print(
    f"Forecast Horizon        : "
    f"{FORECAST_STEPS} window ahead"
)

print(
    "No LOW/MEDIUM/HIGH thresholds were used."
)

print(
    "Future target remains NORMAL / ABNORMAL "
    "crowd activity."
)

print(
    f"Output : {OUTPUT_CSV}"
)

print("================================================")