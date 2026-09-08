import csv
import math
from pathlib import Path
from collections import defaultdict


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "movement_features.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "zone_features.csv"
)

# Each aggregation window = 2 seconds
TIME_WINDOW_SECONDS = 2.0


# =========================================================
# AUTOMATIC NORMALIZED ZONES
# =========================================================
#
# Frame width is divided into 3 normalized zones.
#
# 0.00 - 0.333  -> ZONE_A
# 0.333 - 0.666 -> ZONE_B
# 0.666 - 1.00  -> ZONE_C
#
# Because normalized_x is used, this works across
# different video resolutions.
# =========================================================

def assign_zone(normalized_x):

    if normalized_x < (1.0 / 3.0):

        return "ZONE_A"

    elif normalized_x < (2.0 / 3.0):

        return "ZONE_B"

    else:

        return "ZONE_C"


# =========================================================
# SAFE NUMBER CONVERSION
# =========================================================

def safe_float(value, default=0.0):

    try:

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return default


def safe_int(value, default=0):

    try:

        return int(float(value))

    except (
        ValueError,
        TypeError
    ):

        return default


# =========================================================
# BASIC STATISTICS
# =========================================================

def calculate_mean(values):

    if not values:

        return 0.0

    return sum(values) / len(values)


def calculate_std(values):

    if len(values) < 2:

        return 0.0

    mean_value = calculate_mean(
        values
    )

    variance = sum(
        (
            value - mean_value
        ) ** 2

        for value in values

    ) / len(values)

    return math.sqrt(
        variance
    )


# =========================================================
# CIRCULAR DIRECTION STATISTICS
# =========================================================
#
# Direction is circular.
#
# Example:
# 359 degrees and 1 degree are almost the same direction.
#
# Therefore raw degree standard deviation should NOT
# be used directly.
#
# We use direction_sin and direction_cos.
# =========================================================

def calculate_direction_consistency(
    sin_values,
    cos_values
):

    if not sin_values or not cos_values:

        return 0.0


    mean_sin = calculate_mean(
        sin_values
    )

    mean_cos = calculate_mean(
        cos_values
    )


    resultant_length = math.sqrt(
        mean_sin ** 2
        +
        mean_cos ** 2
    )


    # Numerical safety

    resultant_length = max(
        0.0,
        min(
            1.0,
            resultant_length
        )
    )


    return resultant_length


def calculate_mean_direction(
    sin_values,
    cos_values
):

    if not sin_values or not cos_values:

        return 0.0


    mean_sin = calculate_mean(
        sin_values
    )

    mean_cos = calculate_mean(
        cos_values
    )


    angle = math.degrees(
        math.atan2(
            mean_sin,
            mean_cos
        )
    )


    if angle < 0:

        angle += 360


    return angle


# =========================================================
# CHECK INPUT DATASET
# =========================================================

if not INPUT_CSV.exists():

    print()
    print("==========================================")
    print("ERROR")
    print("==========================================")

    print(
        "movement_features.csv was not found."
    )

    print(
        f"Expected location: {INPUT_CSV}"
    )

    print("==========================================")

    raise SystemExit


# =========================================================
# READ MOVEMENT FEATURES
# =========================================================

print()
print("==========================================")
print("ZONE-BASED ANALYSIS")
print("==========================================")

print(
    f"Input Dataset : {INPUT_CSV}"
)

print(
    f"Time Window   : {TIME_WINDOW_SECONDS} seconds"
)

print("------------------------------------------")


# =========================================================
# GROUP STORAGE
# =========================================================
#
# Key:
#
# (
#   video_name,
#   video_id,
#   window_index,
#   zone_id
# )
#
# Each key contains all person-frame observations
# belonging to that video/time-window/zone.
# =========================================================

groups = defaultdict(
    lambda: {

        "timestamps": [],

        "frames": set(),

        "track_ids": set(),

        "frame_persons": defaultdict(set),

        "valid_motion_rows": 0,

        "speed_norm": [],

        "speed_px": [],

        "acceleration_norm": [],

        "direction_change": [],

        "direction_sin": [],

        "direction_cos": [],

        "bbox_area_ratio": [],

        "confidence": [],

        "movement_norm": []
    }
)


total_input_rows = 0

valid_input_rows = 0


# =========================================================
# PROCESS INPUT CSV
# =========================================================

with open(
    INPUT_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as csv_file:

    reader = csv.DictReader(
        csv_file
    )


    # =====================================================
    # REQUIRED COLUMN CHECK
    # =====================================================

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


    missing_columns = (
        required_columns
        -
        actual_columns
    )


    if missing_columns:

        print()
        print("ERROR: Required columns are missing:")

        for column in sorted(
            missing_columns
        ):

            print(
                f" - {column}"
            )

        print()
        print(
            "Generate movement_features.csv using the improved"
        )

        print(
            "feature_dataset_generator.py first."
        )

        raise SystemExit


    # =====================================================
    # ROW LOOP
    # =====================================================

    for row in reader:

        total_input_rows += 1


        video_name = (
            row["video_name"]
        )

        video_id = (
            row["video_id"]
        )


        frame_number = safe_int(
            row["frame_number"]
        )


        timestamp = safe_float(
            row["timestamp_sec"]
        )


        track_id = safe_int(
            row["track_id"]
        )


        normalized_x = safe_float(
            row["normalized_x"]
        )


        # ================================================
        # NORMALIZED POSITION VALIDATION
        # ================================================

        if (
            normalized_x < 0.0
            or
            normalized_x > 1.0
        ):

            continue


        valid_input_rows += 1


        # ================================================
        # ASSIGN ZONE
        # ================================================

        zone_id = assign_zone(
            normalized_x
        )


        # ================================================
        # TIME WINDOW
        # ================================================

        window_index = int(
            timestamp
            //
            TIME_WINDOW_SECONDS
        )


        window_start = (
            window_index
            *
            TIME_WINDOW_SECONDS
        )


        # ================================================
        # GROUP KEY
        # ================================================

        key = (

            video_name,

            video_id,

            window_index,

            zone_id
        )


        group = groups[
            key
        ]


        # ================================================
        # BASIC OBSERVATIONS
        # ================================================

        group[
            "timestamps"
        ].append(
            timestamp
        )


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


        # ================================================
        # VISUAL OCCUPANCY SIGNAL
        # ================================================

        bbox_area_ratio = safe_float(
            row["bbox_area_ratio"]
        )


        if bbox_area_ratio >= 0:

            group[
                "bbox_area_ratio"
            ].append(
                bbox_area_ratio
            )


        # ================================================
        # DETECTION CONFIDENCE
        # ================================================

        confidence = safe_float(
            row["detection_confidence"]
        )


        group[
            "confidence"
        ].append(
            confidence
        )


        # ================================================
        # TEMPORAL FEATURES
        # ================================================

        motion_valid = safe_int(
            row["motion_valid"]
        )


        # Only consecutive-frame temporal observations
        # are used for motion statistics.

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
                    row[
                        "speed_px_s"
                    ]
                )
            )


            group[
                "speed_norm"
            ].append(

                safe_float(
                    row[
                        "speed_norm_s"
                    ]
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
                row[
                    "direction_sin"
                ]
            )


            direction_cos = safe_float(
                row[
                    "direction_cos"
                ]
            )


            # sin=0 and cos=0 means no valid direction
            # was available for that observation.

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


print(
    f"Input Rows   : {total_input_rows}"
)

print(
    f"Valid Rows   : {valid_input_rows}"
)

print(
    f"Zone Windows : {len(groups)}"
)

print("------------------------------------------")


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

    "zone_presence_ratio",

    "avg_bbox_area_ratio",
    "total_bbox_area_ratio",

    "avg_detection_confidence",

    "valid_motion_rows",

    "avg_movement_norm",

    "avg_speed_px_s",
    "avg_speed_norm_s",
    "speed_std_norm_s",

    "avg_acceleration_norm_s2",
    "acceleration_std_norm_s2",

    "avg_direction_change_deg",

    "mean_direction_deg",
    "direction_consistency",

    "movement_activity"
]


# =========================================================
# CREATE ZONE DATASET
# =========================================================

output_rows = []


for key in sorted(
    groups.keys()
):

    (
        video_name,
        video_id,
        window_index,
        zone_id
    ) = key


    group = groups[
        key
    ]


    # =====================================================
    # WINDOW
    # =====================================================

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
        group[
            "track_ids"
        ]
    )


    frame_person_counts = [

        len(person_ids)

        for person_ids
        in group[
            "frame_persons"
        ].values()
    ]


    avg_person_count = (
        calculate_mean(
            frame_person_counts
        )
    )


    max_person_count = (
        max(
            frame_person_counts
        )

        if frame_person_counts

        else 0
    )


    # =====================================================
    # ZONE PRESENCE RATIO
    # =====================================================
    #
    # Number of frames where this zone had at least
    # one tracked person relative to the expected number
    # of frames in the window.
    #
    # FPS is not explicitly needed here because we use
    # observed frames in the source data.
    # =====================================================

    observed_frames = len(
        group[
            "frames"
        ]
    )


    zone_presence_ratio = (
        1.0
        if observed_frames > 0
        else 0.0
    )


    # =====================================================
    # VISUAL OCCUPANCY
    # =====================================================

    avg_bbox_area_ratio = (
        calculate_mean(
            group[
                "bbox_area_ratio"
            ]
        )
    )


    total_bbox_area_ratio = sum(
        group[
            "bbox_area_ratio"
        ]
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    avg_detection_confidence = (
        calculate_mean(
            group[
                "confidence"
            ]
        )
    )


    # =====================================================
    # MOTION
    # =====================================================

    avg_movement_norm = (
        calculate_mean(
            group[
                "movement_norm"
            ]
        )
    )


    avg_speed_px_s = (
        calculate_mean(
            group[
                "speed_px"
            ]
        )
    )


    avg_speed_norm_s = (
        calculate_mean(
            group[
                "speed_norm"
            ]
        )
    )


    speed_std_norm_s = (
        calculate_std(
            group[
                "speed_norm"
            ]
        )
    )


    # =====================================================
    # ACCELERATION
    # =====================================================

    avg_acceleration_norm_s2 = (
        calculate_mean(
            group[
                "acceleration_norm"
            ]
        )
    )


    acceleration_std_norm_s2 = (
        calculate_std(
            group[
                "acceleration_norm"
            ]
        )
    )


    # =====================================================
    # DIRECTION CHANGE
    # =====================================================

    avg_direction_change_deg = (
        calculate_mean(
            group[
                "direction_change"
            ]
        )
    )


    # =====================================================
    # CIRCULAR DIRECTION
    # =====================================================

    mean_direction_deg = (
        calculate_mean_direction(

            group[
                "direction_sin"
            ],

            group[
                "direction_cos"
            ]
        )
    )


    direction_consistency = (
        calculate_direction_consistency(

            group[
                "direction_sin"
            ],

            group[
                "direction_cos"
            ]
        )
    )


    # =====================================================
    # MOVEMENT ACTIVITY
    # =====================================================
    #
    # Fraction of observations in this zone/window
    # that have valid consecutive-frame motion.
    #
    # This is NOT a risk score.
    # =====================================================

    total_observations = len(
        group[
            "timestamps"
        ]
    )


    movement_activity = (

        group[
            "valid_motion_rows"
        ]
        /
        total_observations

        if total_observations > 0

        else 0.0
    )


    # =====================================================
    # OUTPUT ROW
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
            zone_presence_ratio,
            6
        ),

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

        group[
            "valid_motion_rows"
        ],

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

        round(
            movement_activity,
            6
        )
    ])


# =========================================================
# WRITE OUTPUT CSV
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as output_file:

    writer = csv.writer(
        output_file
    )

    writer.writerow(
        OUTPUT_HEADER
    )

    writer.writerows(
        output_rows
    )


# =========================================================
# PER VIDEO SUMMARY
# =========================================================

video_summary = defaultdict(
    lambda: {

        "rows": 0,
        "zones": set(),
        "windows": set()
    }
)


for row in output_rows:

    video_name = row[0]

    window_index = row[2]

    zone_id = row[5]


    video_summary[
        video_name
    ][
        "rows"
    ] += 1


    video_summary[
        video_name
    ][
        "zones"
    ].add(
        zone_id
    )


    video_summary[
        video_name
    ][
        "windows"
    ].add(
        window_index
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==========================================")
print("ZONE-BASED ANALYSIS COMPLETE")
print("==========================================")

print(
    f"Input Rows       : {total_input_rows}"
)

print(
    f"Output Zone Rows : {len(output_rows)}"
)

print(
    f"Videos           : {len(video_summary)}"
)

print(
    f"Output Dataset   : {OUTPUT_CSV}"
)

print("------------------------------------------")


for video_name in sorted(
    video_summary.keys()
):

    summary = video_summary[
        video_name
    ]


    print(
        f"Video   : {video_name}"
    )

    print(
        f"Windows : {len(summary['windows'])}"
    )

    print(
        f"Zones   : {', '.join(sorted(summary['zones']))}"
    )

    print(
        f"Rows    : {summary['rows']}"
    )

    print("------------------------------------------")


print(
    "No LOW/MEDIUM/HIGH risk labels were generated."
)

print(
    "This dataset contains zone-level spatio-temporal features only."
)

print("==========================================")