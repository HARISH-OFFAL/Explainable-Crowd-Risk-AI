import csv
import math
from pathlib import Path
from collections import defaultdict


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MOVEMENT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_movement_features.csv"
)

GROUND_TRUTH_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_ground_truth.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_zone_labeled_features.csv"
)

TIME_WINDOW_SECONDS = 2.0


# =========================================================
# ZONE ASSIGNMENT
# =========================================================

def assign_zone(normalized_x):

    if normalized_x < (1.0 / 3.0):
        return "ZONE_A"

    elif normalized_x < (2.0 / 3.0):
        return "ZONE_B"

    else:
        return "ZONE_C"


# =========================================================
# SAFE CONVERSION
# =========================================================

def safe_float(value, default=0.0):

    try:
        return float(value)

    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):

    try:
        return int(float(value))

    except (ValueError, TypeError):
        return default


# =========================================================
# STATISTICS
# =========================================================

def calculate_mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_std(values):

    if len(values) < 2:
        return 0.0

    mean_value = calculate_mean(values)

    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / len(values)

    return math.sqrt(variance)


# =========================================================
# CIRCULAR DIRECTION
# =========================================================

def calculate_mean_direction(
    sin_values,
    cos_values
):

    if not sin_values or not cos_values:
        return 0.0

    mean_sin = calculate_mean(sin_values)
    mean_cos = calculate_mean(cos_values)

    angle = math.degrees(
        math.atan2(
            mean_sin,
            mean_cos
        )
    )

    if angle < 0:
        angle += 360.0

    return angle


def calculate_direction_consistency(
    sin_values,
    cos_values
):

    if not sin_values or not cos_values:
        return 0.0

    mean_sin = calculate_mean(sin_values)
    mean_cos = calculate_mean(cos_values)

    resultant = math.sqrt(
        mean_sin ** 2
        +
        mean_cos ** 2
    )

    return max(
        0.0,
        min(
            1.0,
            resultant
        )
    )


# =========================================================
# FILE CHECK
# =========================================================

for required_file in [
    MOVEMENT_CSV,
    GROUND_TRUTH_CSV
]:

    if not required_file.exists():

        print()
        print("==========================================")
        print("ERROR")
        print("==========================================")
        print(f"Missing file: {required_file}")
        print("==========================================")

        raise SystemExit


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

abnormal_intervals = []


with open(
    GROUND_TRUTH_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if (
            row.get(
                "label",
                ""
            ).strip().upper()
            !=
            "ABNORMAL"
        ):
            continue

        start_frame = safe_int(
            row["start_frame"]
        )

        end_frame = safe_int(
            row["end_frame"]
        )

        start_time = safe_float(
            row["start_time_sec"]
        )

        end_time = safe_float(
            row["end_time_sec"]
        )

        interval_id = row.get(
            "interval_id",
            ""
        )

        abnormal_intervals.append({

            "interval_id": interval_id,

            "start_frame": start_frame,

            "end_frame": end_frame,

            "start_time": start_time,

            "end_time": end_time
        })


print()
print("==========================================")
print("UMN ZONE + LABEL PREPARATION")
print("==========================================")
print(f"Movement Dataset : {MOVEMENT_CSV}")
print(f"Ground Truth     : {GROUND_TRUTH_CSV}")
print(f"Abnormal Periods : {len(abnormal_intervals)}")
print(f"Time Window      : {TIME_WINDOW_SECONDS} sec")
print("------------------------------------------")


# =========================================================
# GROUP STORAGE
# =========================================================

groups = defaultdict(
    lambda: {

        "frames": set(),

        "track_ids": set(),

        "frame_persons": defaultdict(set),

        "timestamps": [],

        "bbox_area_ratio": [],

        "confidence": [],

        "valid_motion_rows": 0,

        "movement_norm": [],

        "speed_px": [],

        "speed_norm": [],

        "acceleration_norm": [],

        "direction_change": [],

        "direction_sin": [],

        "direction_cos": []
    }
)


total_rows = 0
valid_rows = 0


# =========================================================
# READ UMN MOVEMENT FEATURES
# =========================================================

with open(
    MOVEMENT_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)


    required_columns = {

        "video_name",
        "video_id",

        "frame_number",
        "timestamp_sec",

        "track_id",

        "normalized_x",

        "motion_valid",

        "movement_distance_norm",

        "speed_px_s",
        "speed_norm_s",

        "acceleration_norm_s2",

        "direction_change_deg",

        "direction_sin",
        "direction_cos",

        "bbox_area_ratio",

        "detection_confidence"
    }


    actual_columns = set(
        reader.fieldnames or []
    )


    missing = (
        required_columns
        -
        actual_columns
    )


    if missing:

        print()
        print("ERROR: Missing required columns:")

        for column in sorted(missing):
            print(f" - {column}")

        raise SystemExit


    for row in reader:

        total_rows += 1


        normalized_x = safe_float(
            row["normalized_x"]
        )


        if (
            normalized_x < 0.0
            or
            normalized_x > 1.0
        ):
            continue


        valid_rows += 1


        video_name = row["video_name"]
        video_id = row["video_id"]

        frame_number = safe_int(
            row["frame_number"]
        )

        timestamp = safe_float(
            row["timestamp_sec"]
        )

        track_id = safe_int(
            row["track_id"]
        )


        zone_id = assign_zone(
            normalized_x
        )


        window_index = int(
            timestamp
            //
            TIME_WINDOW_SECONDS
        )


        key = (

            video_name,

            video_id,

            window_index,

            zone_id
        )


        group = groups[key]


        group[
            "frames"
        ].add(
            frame_number
        )


        group[
            "track_ids"
        ].add(
            track_id
        )


        group[
            "frame_persons"
        ][
            frame_number
        ].add(
            track_id
        )


        group[
            "timestamps"
        ].append(
            timestamp
        )


        group[
            "bbox_area_ratio"
        ].append(

            safe_float(
                row["bbox_area_ratio"]
            )
        )


        group[
            "confidence"
        ].append(

            safe_float(
                row["detection_confidence"]
            )
        )


        motion_valid = safe_int(
            row["motion_valid"]
        )


        if motion_valid == 1:

            group[
                "valid_motion_rows"
            ] += 1


            group[
                "movement_norm"
            ].append(

                safe_float(
                    row[
                        "movement_distance_norm"
                    ]
                )
            )


            group[
                "speed_px"
            ].append(

                safe_float(
                    row["speed_px_s"]
                )
            )


            group[
                "speed_norm"
            ].append(

                safe_float(
                    row["speed_norm_s"]
                )
            )


            group[
                "acceleration_norm"
            ].append(

                safe_float(
                    row[
                        "acceleration_norm_s2"
                    ]
                )
            )


            group[
                "direction_change"
            ].append(

                safe_float(
                    row[
                        "direction_change_deg"
                    ]
                )
            )


            direction_sin = safe_float(
                row["direction_sin"]
            )

            direction_cos = safe_float(
                row["direction_cos"]
            )


            if not (
                direction_sin == 0.0
                and
                direction_cos == 0.0
            ):

                group[
                    "direction_sin"
                ].append(
                    direction_sin
                )

                group[
                    "direction_cos"
                ].append(
                    direction_cos
                )


print(f"Input Rows    : {total_rows}")
print(f"Valid Rows    : {valid_rows}")
print(f"Zone Windows  : {len(groups)}")
print("------------------------------------------")


# =========================================================
# LABEL FUNCTION
# =========================================================

def get_window_label(
    window_start,
    window_end
):

    matching_intervals = []


    for interval in abnormal_intervals:

        abnormal_start = interval[
            "start_time"
        ]

        abnormal_end = interval[
            "end_time"
        ]


        # Time-window overlap check.
        #
        # If any part of the 2-second window overlaps a
        # manually marked abnormal interval, that window
        # receives ABNORMAL.
        #
        # This avoids pretending that a mixed transition
        # window is purely NORMAL.

        overlap = (

            window_start
            <
            abnormal_end

            and

            window_end
            >
            abnormal_start
        )


        if overlap:

            matching_intervals.append(
                interval["interval_id"]
            )


    if matching_intervals:

        return (
            "ABNORMAL",
            "|".join(
                matching_intervals
            )
        )


    return (
        "NORMAL",
        ""
    )


# =========================================================
# OUTPUT HEADER
# =========================================================

OUTPUT_HEADER = [

    "video_name",
    "video_id",

    "window_index",
    "window_start_sec",
    "window_end_sec",

    "zone_id",

    "unique_person_count",

    "avg_person_count_per_frame",
    "max_person_count_per_frame",

    "avg_bbox_area_ratio",
    "total_bbox_area_ratio",

    "avg_detection_confidence",

    "valid_motion_rows",
    "motion_valid_ratio",

    "avg_movement_norm",

    "avg_speed_px_s",
    "avg_speed_norm_s",
    "speed_std_norm_s",

    "avg_acceleration_norm_s2",
    "acceleration_std_norm_s2",

    "avg_direction_change_deg",

    "mean_direction_deg",
    "direction_consistency",

    "ground_truth_label",
    "ground_truth_interval"
]


# =========================================================
# CREATE LABELED ZONE ROWS
# =========================================================

output_rows = []

normal_rows = 0
abnormal_rows = 0


for key in sorted(
    groups.keys()
):

    (
        video_name,
        video_id,
        window_index,
        zone_id
    ) = key


    group = groups[key]


    window_start = (
        window_index
        *
        TIME_WINDOW_SECONDS
    )

    window_end = (
        window_start
        +
        TIME_WINDOW_SECONDS
    )


    # =====================================================
    # PERSON COUNTS
    # =====================================================

    unique_person_count = len(
        group["track_ids"]
    )


    frame_person_counts = [

        len(person_ids)

        for person_ids
        in group[
            "frame_persons"
        ].values()
    ]


    avg_person_count = calculate_mean(
        frame_person_counts
    )


    max_person_count = (

        max(frame_person_counts)

        if frame_person_counts

        else 0
    )


    # =====================================================
    # OCCUPANCY / DETECTION
    # =====================================================

    avg_bbox_area_ratio = calculate_mean(
        group[
            "bbox_area_ratio"
        ]
    )


    total_bbox_area_ratio = sum(
        group[
            "bbox_area_ratio"
        ]
    )


    avg_detection_confidence = calculate_mean(
        group[
            "confidence"
        ]
    )


    # =====================================================
    # MOTION QUALITY
    # =====================================================

    total_observations = len(
        group[
            "timestamps"
        ]
    )


    valid_motion_rows = group[
        "valid_motion_rows"
    ]


    motion_valid_ratio = (

        valid_motion_rows
        /
        total_observations

        if total_observations > 0

        else 0.0
    )


    # =====================================================
    # SPATIO-TEMPORAL MOTION
    # =====================================================

    avg_movement_norm = calculate_mean(
        group[
            "movement_norm"
        ]
    )


    avg_speed_px_s = calculate_mean(
        group[
            "speed_px"
        ]
    )


    avg_speed_norm_s = calculate_mean(
        group[
            "speed_norm"
        ]
    )


    speed_std_norm_s = calculate_std(
        group[
            "speed_norm"
        ]
    )


    avg_acceleration_norm_s2 = calculate_mean(
        group[
            "acceleration_norm"
        ]
    )


    acceleration_std_norm_s2 = calculate_std(
        group[
            "acceleration_norm"
        ]
    )


    avg_direction_change_deg = calculate_mean(
        group[
            "direction_change"
        ]
    )


    mean_direction_deg = calculate_mean_direction(

        group[
            "direction_sin"
        ],

        group[
            "direction_cos"
        ]
    )


    direction_consistency = calculate_direction_consistency(

        group[
            "direction_sin"
        ],

        group[
            "direction_cos"
        ]
    )


    # =====================================================
    # GROUND TRUTH
    # =====================================================

    (
        ground_truth_label,
        ground_truth_interval
    ) = get_window_label(

        window_start,
        window_end
    )


    if ground_truth_label == "ABNORMAL":
        abnormal_rows += 1

    else:
        normal_rows += 1


    # =====================================================
    # SAVE OUTPUT ROW
    # =====================================================

    output_rows.append([

        video_name,
        video_id,

        window_index,

        round(
            window_start,
            4
        ),

        round(
            window_end,
            4
        ),

        zone_id,

        unique_person_count,

        round(
            avg_person_count,
            4
        ),

        max_person_count,

        round(
            avg_bbox_area_ratio,
            8
        ),

        round(
            total_bbox_area_ratio,
            8
        ),

        round(
            avg_detection_confidence,
            6
        ),

        valid_motion_rows,

        round(
            motion_valid_ratio,
            6
        ),

        round(
            avg_movement_norm,
            8
        ),

        round(
            avg_speed_px_s,
            4
        ),

        round(
            avg_speed_norm_s,
            8
        ),

        round(
            speed_std_norm_s,
            8
        ),

        round(
            avg_acceleration_norm_s2,
            8
        ),

        round(
            acceleration_std_norm_s2,
            8
        ),

        round(
            avg_direction_change_deg,
            4
        ),

        round(
            mean_direction_deg,
            4
        ),

        round(
            direction_consistency,
            6
        ),

        ground_truth_label,

        ground_truth_interval
    ])


# =========================================================
# WRITE OUTPUT
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(
        OUTPUT_HEADER
    )

    writer.writerows(
        output_rows
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

total_output_rows = len(
    output_rows
)


normal_percentage = (

    normal_rows
    /
    total_output_rows
    *
    100

    if total_output_rows > 0

    else 0.0
)


abnormal_percentage = (

    abnormal_rows
    /
    total_output_rows
    *
    100

    if total_output_rows > 0

    else 0.0
)


print()
print("==========================================")
print("UMN LABELED ZONE DATASET COMPLETE")
print("==========================================")
print(f"Movement Rows   : {total_rows}")
print(f"Output Rows     : {total_output_rows}")
print("------------------------------------------")
print(
    f"NORMAL          : "
    f"{normal_rows} "
    f"({normal_percentage:.2f}%)"
)
print(
    f"ABNORMAL        : "
    f"{abnormal_rows} "
    f"({abnormal_percentage:.2f}%)"
)
print("------------------------------------------")
print(f"Output Dataset  : {OUTPUT_CSV}")
print("------------------------------------------")
print(
    "Ground truth is based on manually inspected "
    "UMN abnormal intervals."
)
print(
    "No LOW/MEDIUM/HIGH crowd-risk labels were created."
)
print("==========================================")