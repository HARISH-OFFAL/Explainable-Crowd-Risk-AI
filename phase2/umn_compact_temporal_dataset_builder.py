from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_temporal_6sec_forecast_features.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_compact_temporal_6sec_forecast_features.csv"
)


# =========================================================
# LOAD
# =========================================================

print()
print("================================================")
print("UMN COMPACT TEMPORAL FORECAST DATASET BUILDER")
print("================================================")


if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(INPUT_CSV)

print(f"Input Samples : {len(df)}")
print(f"Input Columns : {len(df.columns)}")


# =========================================================
# METADATA / TARGET
# =========================================================

METADATA_COLUMNS = [
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
]


# =========================================================
# COMPACT FEATURE DESIGN
# =========================================================
#
# Instead of using all 188 features,
# retain a compact set representing:
#
# 1. Crowd count / density
# 2. Speed
# 3. Speed variation
# 4. Acceleration
# 5. Direction change
# 6. Direction consistency
# 7. Movement activity
# 8. Temporal change from oldest -> current
#
# H0 = approximately 4-6 sec earlier
# H1 = middle
# H2 = current
#
# =========================================================


GLOBAL_BASE_FEATURES = [
    "global_unique_person_count",
    "global_avg_person_count_per_frame",
    "global_max_zone_person_count",
    "global_avg_speed_norm_s",
    "global_speed_std_norm_s",
    "global_avg_acceleration_norm_s2",
    "global_avg_direction_change_deg",
    "global_direction_consistency",
    "global_movement_activity_ratio",
]


COMPACT_FEATURE_COLUMNS = []


# =========================================================
# HISTORY FEATURES
# =========================================================

for history_prefix in [
    "H0",
    "H1",
    "H2",
]:

    for feature in GLOBAL_BASE_FEATURES:

        COMPACT_FEATURE_COLUMNS.append(
            f"{history_prefix}_{feature}"
        )


# =========================================================
# TEMPORAL DELTA FEATURES
# =========================================================

DELTA_FEATURES = [
    "DELTA_global_unique_person_count",
    "DELTA_global_avg_person_count_per_frame",
    "DELTA_global_avg_speed_norm_s",
    "DELTA_global_speed_std_norm_s",
    "DELTA_global_avg_acceleration_norm_s2",
    "DELTA_global_avg_direction_change_deg",
    "DELTA_global_direction_consistency",
    "DELTA_global_movement_activity_ratio",
]


COMPACT_FEATURE_COLUMNS.extend(
    DELTA_FEATURES
)


# =========================================================
# VERIFY COLUMNS
# =========================================================

required_columns = (
    METADATA_COLUMNS
    + COMPACT_FEATURE_COLUMNS
)


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# =========================================================
# CREATE COMPACT DATASET
# =========================================================

output_columns = (
    METADATA_COLUMNS
    + COMPACT_FEATURE_COLUMNS
)


compact_df = df[
    output_columns
].copy()


# =========================================================
# NUMERIC CLEANING
# =========================================================

for feature in COMPACT_FEATURE_COLUMNS:

    compact_df[feature] = pd.to_numeric(
        compact_df[feature],
        errors="coerce",
    )


# =========================================================
# TARGET CLEANING
# =========================================================

compact_df[
    "future_abnormal_within_6sec"
] = pd.to_numeric(
    compact_df[
        "future_abnormal_within_6sec"
    ],
    errors="coerce",
)


compact_df = compact_df.dropna(
    subset=[
        "sequence_id",
        "training_split",
        "future_abnormal_within_6sec",
    ]
).copy()


compact_df[
    "future_abnormal_within_6sec"
] = compact_df[
    "future_abnormal_within_6sec"
].astype(int)


# =========================================================
# SEQUENCE LEAKAGE CHECK
# =========================================================

sequence_split_counts = (
    compact_df.groupby(
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
# CHECK CLASSES
# =========================================================

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = compact_df[
        compact_df[
            "training_split"
        ] == split
    ]


    if split_df.empty:

        raise ValueError(
            f"{split} contains no samples."
        )


    classes = set(
        split_df[
            "future_abnormal_within_6sec"
        ].unique()
    )


    if classes != {0, 1}:

        raise ValueError(
            f"{split} does not contain "
            f"both forecast classes."
        )


# =========================================================
# SAVE
# =========================================================

compact_df = compact_df.sort_values(
    [
        "sequence_id",
        "current_window_id",
    ]
).reset_index(
    drop=True
)


compact_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

print("----------------------------------------------")

print(
    f"Compact Samples  : "
    f"{len(compact_df)}"
)

print(
    f"Model Features   : "
    f"{len(COMPACT_FEATURE_COLUMNS)}"
)

print("----------------------------------------------")


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = compact_df[
        compact_df[
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
    "Only compact global spatio-temporal "
    "features were retained."
)

print(
    "ZONE_A/B/C information remains indirectly "
    "represented through global zone summaries."
)

print(
    "No target/frame-transition information "
    "is included as a model feature."
)

print(
    "No LOW/MEDIUM/HIGH thresholds were used."
)

print(
    f"Output : {OUTPUT_CSV}"
)

print("================================================")