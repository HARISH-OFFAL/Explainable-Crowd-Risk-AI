from pathlib import Path

import numpy as np
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
    / "umn_temporal_6sec_forecast_features.csv"
)


# =========================================================
# SETTINGS
# =========================================================
#
# Zone window = 2 sec
# History = previous 3 windows = about 6 sec
# Forecast = next 6 sec
#
# =========================================================

WINDOW_SECONDS = 2.0
HISTORY_WINDOWS = 3
FORECAST_SECONDS = 6.0

FPS = 30.0

FORECAST_FRAMES = int(
    FORECAST_SECONDS * FPS
)

ZONES = [
    "ZONE_A",
    "ZONE_B",
    "ZONE_C",
]


# =========================================================
# ZONE FEATURES
# =========================================================

ZONE_FEATURES = [
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
print("UMN TEMPORAL FUTURE FORECAST DATASET BUILDER")
print("================================================")

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print(f"Input Zone Rows : {len(df)}")


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required = (
    ZONE_FEATURES
    + [
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
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# =========================================================
# CLEAN TYPES
# =========================================================

numeric_columns = (
    ZONE_FEATURES
    + [
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
        errors="coerce",
    )

df = df.dropna(
    subset=[
        "sequence_id",
        "zone_id",
        "window_id",
        "start_frame",
        "end_frame",
        "behavior_label_value",
        "transition_onset_frame",
        "training_split",
    ]
).copy()

df["window_id"] = df[
    "window_id"
].astype(int)


# =========================================================
# BUILD ONE ROW PER SEQUENCE-WINDOW
# =========================================================
#
# Instead of treating each zone as separate target sample,
# combine A/B/C into one scene-time representation.
#
# Missing zone -> zero features.
#
# =========================================================

scene_rows = []


grouped = df.groupby(
    [
        "sequence_id",
        "window_id",
    ],
    sort=True,
)


for (
    sequence_id,
    window_id,
), group in grouped:

    group = group.sort_values(
        "zone_id"
    )

    base = group.iloc[0]

    output = {
        "sequence_id":
            sequence_id,

        "scene_id":
            base["scene_id"],

        "scene_name":
            base["scene_name"],

        "window_id":
            int(window_id),

        "start_frame":
            int(group["start_frame"].min()),

        "end_frame":
            int(group["end_frame"].max()),

        "transition_onset_frame":
            int(base["transition_onset_frame"]),

        "training_split":
            base["training_split"],
    }


    # =====================================================
    # CREATE ZONE-SPECIFIC FEATURE COLUMNS
    # =====================================================

    for zone in ZONES:

        zone_group = group[
            group["zone_id"] == zone
        ]

        if zone_group.empty:

            for feature in ZONE_FEATURES:
                output[
                    f"{zone}_{feature}"
                ] = 0.0

        else:

            row = zone_group.iloc[0]

            for feature in ZONE_FEATURES:

                value = row[feature]

                if pd.isna(value):
                    value = 0.0

                output[
                    f"{zone}_{feature}"
                ] = float(value)


    # =====================================================
    # GLOBAL CURRENT-WINDOW SUMMARY
    # =====================================================

    output[
        "global_unique_person_count"
    ] = float(
        group[
            "unique_person_count"
        ].sum()
    )

    output[
        "global_avg_person_count_per_frame"
    ] = float(
        group[
            "avg_person_count_per_frame"
        ].sum()
    )

    output[
        "global_max_zone_person_count"
    ] = float(
        group[
            "max_person_count_per_frame"
        ].max()
    )

    output[
        "global_avg_speed_norm_s"
    ] = float(
        group[
            "avg_speed_norm_s"
        ].mean()
    )

    output[
        "global_speed_std_norm_s"
    ] = float(
        group[
            "speed_std_norm_s"
        ].mean()
    )

    output[
        "global_avg_acceleration_norm_s2"
    ] = float(
        group[
            "avg_acceleration_norm_s2"
        ].mean()
    )

    output[
        "global_avg_direction_change_deg"
    ] = float(
        group[
            "avg_direction_change_deg"
        ].mean()
    )

    output[
        "global_direction_consistency"
    ] = float(
        group[
            "direction_consistency"
        ].mean()
    )

    output[
        "global_movement_activity_ratio"
    ] = float(
        group[
            "movement_activity_ratio"
        ].mean()
    )


    # =====================================================
    # CURRENT BEHAVIOR LABEL
    # =====================================================

    labels = set(
        group[
            "behavior_label_value"
        ].astype(int)
    )

    if len(labels) != 1:
        raise ValueError(
            f"Mixed labels found in "
            f"{sequence_id} window {window_id}"
        )

    output[
        "current_behavior_value"
    ] = int(
        next(iter(labels))
    )

    scene_rows.append(output)


scene_df = pd.DataFrame(
    scene_rows
)

scene_df = scene_df.sort_values(
    [
        "sequence_id",
        "window_id",
    ]
).reset_index(drop=True)


print(
    f"Scene-Level Windows : "
    f"{len(scene_df)}"
)


# =========================================================
# SEQUENCE LEAKAGE CHECK
# =========================================================

sequence_split_counts = (
    scene_df.groupby(
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
# CURRENT FEATURE COLUMNS
# =========================================================

metadata_columns = {
    "sequence_id",
    "scene_id",
    "scene_name",
    "window_id",
    "start_frame",
    "end_frame",
    "transition_onset_frame",
    "training_split",
    "current_behavior_value",
}

current_feature_columns = [
    column
    for column in scene_df.columns
    if column not in metadata_columns
]


# =========================================================
# BUILD TEMPORAL HISTORY SAMPLES
# =========================================================
#
# Need 3 consecutive scene windows:
#
# t-4 sec
# t-2 sec
# t
#
# Then predict abnormal onset within NEXT 6 sec.
#
# Only current window must still be NORMAL.
#
# =========================================================

temporal_rows = []

skipped_history = 0
skipped_non_normal_current = 0
skipped_transition_already_started = 0


for sequence_id, seq_df in scene_df.groupby(
    "sequence_id"
):

    seq_df = seq_df.sort_values(
        "window_id"
    ).reset_index(drop=True)

    window_lookup = {
        int(row["window_id"]): row
        for _, row in seq_df.iterrows()
    }


    for _, current in seq_df.iterrows():

        current_window_id = int(
            current["window_id"]
        )


        # -------------------------------------------------
        # Current window must be NORMAL
        # -------------------------------------------------

        if int(
            current[
                "current_behavior_value"
            ]
        ) != 0:

            skipped_non_normal_current += 1
            continue


        history_ids = [
            current_window_id - 2,
            current_window_id - 1,
            current_window_id,
        ]


        # -------------------------------------------------
        # Require 3 consecutive windows
        # -------------------------------------------------

        if not all(
            wid in window_lookup
            for wid in history_ids
        ):

            skipped_history += 1
            continue


        history_rows = [
            window_lookup[wid]
            for wid in history_ids
        ]


        # -------------------------------------------------
        # All history windows must be NORMAL
        # -------------------------------------------------

        if any(
            int(
                row[
                    "current_behavior_value"
                ]
            ) != 0
            for row in history_rows
        ):

            skipped_non_normal_current += 1
            continue


        transition_frame = int(
            current[
                "transition_onset_frame"
            ]
        )

        current_end_frame = int(
            current[
                "end_frame"
            ]
        )


        frames_until_transition = (
            transition_frame
            -
            current_end_frame
        )


        if frames_until_transition <= 0:

            skipped_transition_already_started += 1
            continue


        # =================================================
        # TARGET
        # =================================================

        future_abnormal = int(
            frames_until_transition
            <= FORECAST_FRAMES
        )


        output = {
            "sequence_id":
                sequence_id,

            "scene_id":
                current["scene_id"],

            "scene_name":
                current["scene_name"],

            "current_window_id":
                current_window_id,

            "history_start_window_id":
                history_ids[0],

            "history_end_window_id":
                history_ids[-1],

            "current_start_frame":
                int(current["start_frame"]),

            "current_end_frame":
                current_end_frame,

            "frames_until_transition":
                frames_until_transition,

            "seconds_until_transition":
                frames_until_transition / FPS,

            "future_abnormal_within_6sec":
                future_abnormal,

            "future_forecast_label":
                (
                    "ABNORMAL_WITHIN_6SEC"
                    if future_abnormal == 1
                    else "NO_TRANSITION_WITHIN_6SEC"
                ),

            "training_split":
                current["training_split"],
        }


        # =================================================
        # HISTORY FEATURES
        # =================================================
        #
        # H0 = oldest
        # H1 = middle
        # H2 = current
        #
        # =================================================

        for history_index, row in enumerate(
            history_rows
        ):

            prefix = f"H{history_index}"

            for feature in current_feature_columns:

                value = row[feature]

                if pd.isna(value):
                    value = 0.0

                output[
                    f"{prefix}_{feature}"
                ] = float(value)


        # =================================================
        # TEMPORAL CHANGE FEATURES
        # =================================================
        #
        # Current minus oldest window.
        #
        # =================================================

        oldest = history_rows[0]
        newest = history_rows[-1]


        trend_features = [
            "global_unique_person_count",
            "global_avg_person_count_per_frame",
            "global_avg_speed_norm_s",
            "global_speed_std_norm_s",
            "global_avg_acceleration_norm_s2",
            "global_avg_direction_change_deg",
            "global_direction_consistency",
            "global_movement_activity_ratio",
        ]


        for feature in trend_features:

            oldest_value = float(
                oldest[feature]
            )

            newest_value = float(
                newest[feature]
            )

            output[
                f"DELTA_{feature}"
            ] = (
                newest_value
                -
                oldest_value
            )


        temporal_rows.append(output)


# =========================================================
# CREATE OUTPUT DATAFRAME
# =========================================================

temporal_df = pd.DataFrame(
    temporal_rows
)


if temporal_df.empty:
    raise ValueError(
        "No temporal forecasting samples generated."
    )


# =========================================================
# FINAL LEAKAGE CHECK
# =========================================================

sequence_split_counts = (
    temporal_df.groupby(
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
        "Temporal dataset sequence leakage detected."
    )


# =========================================================
# VERIFY SPLITS
# =========================================================

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = temporal_df[
        temporal_df[
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
        print(
            f"WARNING: {split} does not "
            f"contain both forecast classes."
        )


# =========================================================
# SAVE
# =========================================================

temporal_df = temporal_df.sort_values(
    [
        "sequence_id",
        "current_window_id",
    ]
).reset_index(drop=True)


temporal_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("----------------------------------------------")

print(
    f"Temporal Forecast Samples : "
    f"{len(temporal_df)}"
)

print(
    f"History Length            : "
    f"{HISTORY_WINDOWS} windows"
)

print(
    f"History Duration          : "
    f"{HISTORY_WINDOWS * WINDOW_SECONDS:.1f} sec"
)

print(
    f"Forecast Horizon          : "
    f"{FORECAST_SECONDS:.1f} sec"
)

print("----------------------------------------------")


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    split_df = temporal_df[
        temporal_df[
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
    f"Skipped insufficient history : "
    f"{skipped_history}"
)

print(
    f"Skipped non-normal history   : "
    f"{skipped_non_normal_current}"
)

print(
    f"Skipped transition started   : "
    f"{skipped_transition_already_started}"
)

print("----------------------------------------------")

print(
    "Sequence Leakage Check : PASSED"
)

print(
    "Zones A/B/C were combined into "
    "one scene-level time window."
)

print(
    "Three consecutive windows were used "
    "as temporal history."
)

print(
    "Target = abnormal crowd activity "
    "starting within the next 6 seconds."
)

print(
    "No LOW/MEDIUM/HIGH thresholds were used."
)

print(
    f"Output : {OUTPUT_CSV}"
)

print("================================================")