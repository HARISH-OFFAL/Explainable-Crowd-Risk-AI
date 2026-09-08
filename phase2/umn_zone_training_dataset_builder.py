from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_labeled_movement_features.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_zone_training_features.csv"
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

WINDOW_SECONDS = 2.0

ZONE_NAMES = [
    "ZONE_A",
    "ZONE_B",
    "ZONE_C",
]


# =========================================================
# LOAD DATA
# =========================================================

print()
print("================================================")
print("UMN ZONE-LEVEL TRAINING DATASET BUILDER")
print("================================================")

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_CSV}"
    )


df = pd.read_csv(INPUT_CSV)

print(f"Input Rows : {len(df)}")


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "sequence_id",
    "scene_id",
    "scene_name",
    "frame_number",
    "timestamp_sec",
    "track_id",
    "video_fps",
    "normalized_x",
    "normalized_y",
    "bbox_area_ratio",
    "detection_confidence",
    "motion_valid",
    "movement_distance_norm",
    "speed_px_s",
    "speed_norm_s",
    "acceleration_norm_s2",
    "direction_change_deg",
    "direction_sin",
    "direction_cos",
    "behavior_label",
    "behavior_label_value",
    "transition_onset_frame",
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
        + "\n".join(missing)
    )


# =========================================================
# CLEAN NUMERIC DATA
# =========================================================

numeric_columns = [
    "frame_number",
    "timestamp_sec",
    "video_fps",
    "normalized_x",
    "normalized_y",
    "bbox_area_ratio",
    "detection_confidence",
    "motion_valid",
    "movement_distance_norm",
    "speed_px_s",
    "speed_norm_s",
    "acceleration_norm_s2",
    "direction_change_deg",
    "direction_sin",
    "direction_cos",
    "behavior_label_value",
]


for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "sequence_id",
        "frame_number",
        "normalized_x",
        "behavior_label_value",
        "training_split",
    ]
).copy()


# =========================================================
# ASSIGN SPATIAL ZONES
# =========================================================
#
# Frame normalized horizontally:
#
# 0.00 -------- 0.33 -------- 0.66 -------- 1.00
#      ZONE_A         ZONE_B         ZONE_C
#
# Same normalized zone concept as our zone-analysis stage.
#
# =========================================================

def assign_zone(normalized_x):

    if normalized_x < (1.0 / 3.0):
        return "ZONE_A"

    if normalized_x < (2.0 / 3.0):
        return "ZONE_B"

    return "ZONE_C"


df["zone_id"] = df[
    "normalized_x"
].apply(assign_zone)


# =========================================================
# CREATE SEQUENCE-LOCAL WINDOWS
# =========================================================
#
# IMPORTANT:
# Window numbering restarts for every reconstructed sequence.
#
# Therefore a window can NEVER cross a sequence boundary.
#
# =========================================================

sequence_start_frames = (
    df.groupby("sequence_id")[
        "frame_number"
    ]
    .min()
    .to_dict()
)


def calculate_window(row):

    fps = float(
        row["video_fps"]
    )

    if fps <= 0:
        fps = 30.0

    sequence_start = sequence_start_frames[
        row["sequence_id"]
    ]

    relative_frame = (
        row["frame_number"]
        - sequence_start
    )

    frames_per_window = (
        fps
        * WINDOW_SECONDS
    )

    return int(
        relative_frame
        // frames_per_window
    )


df["window_id"] = df.apply(
    calculate_window,
    axis=1
)


# =========================================================
# REMOVE WINDOWS THAT CROSS BEHAVIOR TRANSITION
# =========================================================
#
# A 2-second window containing BOTH NORMAL and ABNORMAL
# person rows has an ambiguous label.
#
# We do NOT force a label on such a window.
#
# =========================================================

label_counts = (
    df.groupby(
        [
            "sequence_id",
            "zone_id",
            "window_id",
        ]
    )[
        "behavior_label_value"
    ]
    .nunique()
)


valid_group_index = set(
    label_counts[
        label_counts == 1
    ].index
)


before_rows = len(df)


valid_mask = [
    (
        row.sequence_id,
        row.zone_id,
        row.window_id,
    )
    in valid_group_index

    for row in df.itertuples()
]


df = df[
    valid_mask
].copy()


removed_rows = (
    before_rows
    - len(df)
)


print(
    f"Rows removed from transition-mixed "
    f"zone windows : {removed_rows}"
)


# =========================================================
# CIRCULAR DIRECTION HELPER
# =========================================================

def circular_direction(group):

    mean_sin = group[
        "direction_sin"
    ].mean()

    mean_cos = group[
        "direction_cos"
    ].mean()


    angle = np.degrees(
        np.arctan2(
            mean_sin,
            mean_cos
        )
    )


    if angle < 0:
        angle += 360.0


    consistency = np.sqrt(
        (mean_sin ** 2)
        +
        (mean_cos ** 2)
    )


    return (
        float(angle),
        float(consistency)
    )


# =========================================================
# AGGREGATE ZONE WINDOWS
# =========================================================

output_rows = []


group_columns = [
    "sequence_id",
    "zone_id",
    "window_id",
]


groups = df.groupby(
    group_columns,
    sort=True
)


for (
    sequence_id,
    zone_id,
    window_id
), group in groups:


    # -----------------------------------------------------
    # Verify single label
    # -----------------------------------------------------

    labels = group[
        "behavior_label_value"
    ].unique()


    if len(labels) != 1:
        continue


    label_value = int(
        labels[0]
    )

    behavior_label = (
        "ABNORMAL"
        if label_value == 1
        else "NORMAL"
    )


    # -----------------------------------------------------
    # Verify single split
    # -----------------------------------------------------

    splits = group[
        "training_split"
    ].unique()


    if len(splits) != 1:

        raise ValueError(
            f"Multiple splits detected inside "
            f"{sequence_id} window {window_id}"
        )


    training_split = splits[0]


    # -----------------------------------------------------
    # Frame statistics
    # -----------------------------------------------------

    first_frame = int(
        group["frame_number"].min()
    )

    last_frame = int(
        group["frame_number"].max()
    )

    observed_frames = int(
        group["frame_number"].nunique()
    )


    fps = float(
        group["video_fps"].median()
    )


    expected_frames = max(
        1,
        int(
            round(
                fps
                * WINDOW_SECONDS
            )
        )
    )


    zone_presence_ratio = min(
        1.0,
        observed_frames
        / expected_frames
    )


    # -----------------------------------------------------
    # Person count per frame
    # -----------------------------------------------------

    persons_per_frame = (
        group.groupby(
            "frame_number"
        )[
            "track_id"
        ]
        .nunique()
    )


    unique_person_count = int(
        group["track_id"].nunique()
    )


    avg_person_count = float(
        persons_per_frame.mean()
    )


    max_person_count = int(
        persons_per_frame.max()
    )


    # -----------------------------------------------------
    # Direction
    # -----------------------------------------------------

    (
        mean_direction_deg,
        direction_consistency
    ) = circular_direction(group)


    # -----------------------------------------------------
    # Motion validity
    # -----------------------------------------------------

    valid_motion_rows = int(
        (
            group["motion_valid"] == 1
        ).sum()
    )


    movement_activity_ratio = float(
        (
            group["movement_distance_norm"] > 0
        ).mean()
    )


    # -----------------------------------------------------
    # Aggregate
    # -----------------------------------------------------

    output_rows.append(
        {
            "sequence_id":
                sequence_id,

            "scene_id":
                group["scene_id"].iloc[0],

            "scene_name":
                group["scene_name"].iloc[0],

            "zone_id":
                zone_id,

            "window_id":
                int(window_id),

            "window_seconds":
                WINDOW_SECONDS,

            "start_frame":
                first_frame,

            "end_frame":
                last_frame,

            "observed_frames":
                observed_frames,

            "expected_frames":
                expected_frames,

            "zone_presence_ratio":
                zone_presence_ratio,

            "unique_person_count":
                unique_person_count,

            "avg_person_count_per_frame":
                avg_person_count,

            "max_person_count_per_frame":
                max_person_count,

            "avg_bbox_area_ratio":
                float(
                    group[
                        "bbox_area_ratio"
                    ].mean()
                ),

            "max_bbox_area_ratio":
                float(
                    group[
                        "bbox_area_ratio"
                    ].max()
                ),

            "avg_detection_confidence":
                float(
                    group[
                        "detection_confidence"
                    ].mean()
                ),

            "valid_motion_rows":
                valid_motion_rows,

            "avg_movement_distance_norm":
                float(
                    group[
                        "movement_distance_norm"
                    ].mean()
                ),

            "avg_speed_px_s":
                float(
                    group[
                        "speed_px_s"
                    ].mean()
                ),

            "avg_speed_norm_s":
                float(
                    group[
                        "speed_norm_s"
                    ].mean()
                ),

            "max_speed_norm_s":
                float(
                    group[
                        "speed_norm_s"
                    ].max()
                ),

            "speed_std_norm_s":
                float(
                    group[
                        "speed_norm_s"
                    ].std(ddof=0)
                ),

            "avg_acceleration_norm_s2":
                float(
                    group[
                        "acceleration_norm_s2"
                    ].mean()
                ),

            "acceleration_std_norm_s2":
                float(
                    group[
                        "acceleration_norm_s2"
                    ].std(ddof=0)
                ),

            "avg_direction_change_deg":
                float(
                    group[
                        "direction_change_deg"
                    ].mean()
                ),

            "mean_direction_deg":
                mean_direction_deg,

            "direction_consistency":
                direction_consistency,

            "movement_activity_ratio":
                movement_activity_ratio,

            "behavior_label":
                behavior_label,

            "behavior_label_value":
                label_value,

            "transition_onset_frame":
                int(
                    group[
                        "transition_onset_frame"
                    ].iloc[0]
                ),

            "label_source":
                "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

            "training_split":
                training_split,
        }
    )


# =========================================================
# CREATE OUTPUT DATAFRAME
# =========================================================

output_df = pd.DataFrame(
    output_rows
)


if output_df.empty:

    raise ValueError(
        "No zone-level rows were generated."
    )


# =========================================================
# LEAKAGE VALIDATION
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
        "Sequence leakage detected in "
        "zone-level dataset."
    )


# =========================================================
# CLASS VALIDATION
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
            f"{split} has no zone rows."
        )


    labels = set(
        split_df[
            "behavior_label"
        ].unique()
    )


    if not {
        "NORMAL",
        "ABNORMAL"
    }.issubset(labels):

        raise ValueError(
            f"{split} does not contain "
            f"both NORMAL and ABNORMAL."
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
    f"Zone Rows Generated : "
    f"{len(output_df)}"
)

print(
    f"Sequences           : "
    f"{output_df['sequence_id'].nunique()}"
)

print(
    f"Window Size         : "
    f"{WINDOW_SECONDS:.1f} sec"
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


    normal_count = int(
        (
            split_df[
                "behavior_label"
            ] == "NORMAL"
        ).sum()
    )


    abnormal_count = int(
        (
            split_df[
                "behavior_label"
            ] == "ABNORMAL"
        ).sum()
    )


    print(split)

    print(
        f"  NORMAL   : "
        f"{normal_count}"
    )

    print(
        f"  ABNORMAL : "
        f"{abnormal_count}"
    )

    print(
        f"  TOTAL    : "
        f"{len(split_df)}"
    )

    print()


print("----------------------------------------------")

print(
    "Sequence Leakage Check : PASSED"
)

print(
    "Mixed transition zone windows were excluded."
)

print(
    "No LOW/MEDIUM/HIGH thresholds were created."
)

print(
    "Labels remain NORMAL / ABNORMAL crowd activity."
)

print(
    f"Output : {OUTPUT_CSV}"
)

print("================================================")