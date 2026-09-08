import csv
import math
from collections import defaultdict
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_movement_features.csv"
)

SEQUENCE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_verified_sequences.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_sequence_abnormal_candidates.csv"
)


# =========================================================
# CONFIG
# =========================================================

WINDOW_SECONDS = 1.0

SMOOTH_RADIUS = 2

# Avoid first/last tiny part of sequence.
EDGE_MARGIN_SECONDS = 2.0

# Number of transition candidates kept per sequence.
TOP_PER_SEQUENCE = 3

# Keep candidates sufficiently separated.
MIN_GAP_SECONDS = 3.0


# =========================================================
# HELPERS
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


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    if len(values) < 2:
        return 0.0

    avg = mean(values)

    return math.sqrt(
        mean(
            [
                (value - avg) ** 2
                for value in values
            ]
        )
    )


def zscore(values):
    if not values:
        return []

    avg = mean(values)
    deviation = std(values)

    if deviation == 0:
        return [0.0 for _ in values]

    return [
        (value - avg) / deviation
        for value in values
    ]


def moving_average(values, radius):
    output = []

    for index in range(len(values)):

        start = max(
            0,
            index - radius
        )

        end = min(
            len(values),
            index + radius + 1
        )

        output.append(
            mean(values[start:end])
        )

    return output


# =========================================================
# CHECK FILES
# =========================================================

if not FEATURE_CSV.exists():

    print()
    print("ERROR: UMN movement feature CSV not found.")
    print(FEATURE_CSV)

    raise SystemExit


if not SEQUENCE_CSV.exists():

    print()
    print("ERROR: Verified sequence CSV not found.")
    print(SEQUENCE_CSV)

    raise SystemExit


# =========================================================
# LOAD SEQUENCES
# =========================================================

sequences = []


with open(
    SEQUENCE_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        sequences.append(
            {
                "sequence_id":
                    row["sequence_id"],

                "scene_id":
                    row["scene_id"],

                "scene_name":
                    row["scene_name"],

                "start_frame":
                    safe_int(
                        row["start_frame"]
                    ),

                "end_frame":
                    safe_int(
                        row["end_frame"]
                    )
            }
        )


# =========================================================
# FRAME → SEQUENCE LOOKUP
# =========================================================

def get_sequence(frame_number):

    for sequence in sequences:

        if (
            sequence["start_frame"]
            <= frame_number
            <= sequence["end_frame"]
        ):
            return sequence

    return None


# =========================================================
# COLLECT FEATURES BY SEQUENCE / WINDOW
# =========================================================

sequence_windows = defaultdict(
    lambda: defaultdict(
        lambda: {
            "frames": set(),
            "track_ids": set(),
            "speed": [],
            "max_speed_values": [],
            "acceleration": [],
            "direction_change": [],
            "movement": [],
            "valid_motion_rows": 0,
            "total_rows": 0
        }
    )
)


fps = 30.0


print()
print("==========================================")
print("UMN SEQUENCE ABNORMAL DETECTOR")
print("==========================================")
print(f"Features  : {FEATURE_CSV}")
print(f"Sequences : {SEQUENCE_CSV}")
print("------------------------------------------")


rows_read = 0
rows_used = 0


with open(
    FEATURE_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        rows_read += 1

        frame_number = safe_int(
            row["frame_number"]
        )

        sequence = get_sequence(
            frame_number
        )

        if sequence is None:
            continue

        rows_used += 1

        fps = safe_float(
            row["video_fps"],
            30.0
        )

        relative_frame = (
            frame_number
            -
            sequence["start_frame"]
        )

        relative_time = (
            relative_frame
            /
            fps
        )

        window_index = int(
            relative_time
            //
            WINDOW_SECONDS
        )

        group = sequence_windows[
            sequence["sequence_id"]
        ][
            window_index
        ]

        group[
            "frames"
        ].add(
            frame_number
        )

        group[
            "track_ids"
        ].add(
            safe_int(
                row["track_id"]
            )
        )

        group[
            "total_rows"
        ] += 1


        motion_valid = safe_int(
            row["motion_valid"]
        )

        if motion_valid != 1:
            continue


        group[
            "valid_motion_rows"
        ] += 1


        speed = safe_float(
            row["speed_norm_s"]
        )

        group[
            "speed"
        ].append(
            speed
        )

        group[
            "max_speed_values"
        ].append(
            speed
        )

        group[
            "acceleration"
        ].append(
            abs(
                safe_float(
                    row[
                        "acceleration_norm_s2"
                    ]
                )
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

        group[
            "movement"
        ].append(
            safe_float(
                row[
                    "movement_distance_norm"
                ]
            )
        )


print(f"Rows Read : {rows_read}")
print(f"Rows Used : {rows_used}")
print("------------------------------------------")


# =========================================================
# ANALYZE EACH SEQUENCE
# =========================================================

all_candidates = []


for sequence in sequences:

    sequence_id = sequence[
        "sequence_id"
    ]

    windows = sequence_windows[
        sequence_id
    ]


    records = []


    for window_index in sorted(
        windows.keys()
    ):

        group = windows[
            window_index
        ]

        start_frame = (
            sequence["start_frame"]
            +
            int(
                window_index
                *
                WINDOW_SECONDS
                *
                fps
            )
        )

        end_frame = min(
            sequence["end_frame"],
            start_frame
            +
            int(
                WINDOW_SECONDS
                *
                fps
            )
            -
            1
        )


        records.append(
            {
                "window_index":
                    window_index,

                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "person_count":
                    len(
                        group[
                            "track_ids"
                        ]
                    ),

                "avg_speed":
                    mean(
                        group[
                            "speed"
                        ]
                    ),

                "max_speed":
                    max(
                        group[
                            "max_speed_values"
                        ]
                    )
                    if group[
                        "max_speed_values"
                    ]
                    else 0.0,

                "avg_acceleration":
                    mean(
                        group[
                            "acceleration"
                        ]
                    ),

                "avg_direction_change":
                    mean(
                        group[
                            "direction_change"
                        ]
                    ),

                "avg_movement":
                    mean(
                        group[
                            "movement"
                        ]
                    ),

                "motion_ratio":
                    (
                        group[
                            "valid_motion_rows"
                        ]
                        /
                        group[
                            "total_rows"
                        ]

                        if group[
                            "total_rows"
                        ] > 0

                        else 0.0
                    )
            }
        )


    if len(records) < 4:

        print(
            f"{sequence_id}: "
            f"Not enough windows."
        )

        continue


    # =====================================================
    # NORMALIZE WITHIN THIS SEQUENCE
    # =====================================================

    speed_z = zscore(
        [
            row["avg_speed"]
            for row in records
        ]
    )

    max_speed_z = zscore(
        [
            row["max_speed"]
            for row in records
        ]
    )

    acc_z = zscore(
        [
            row["avg_acceleration"]
            for row in records
        ]
    )

    direction_z = zscore(
        [
            row["avg_direction_change"]
            for row in records
        ]
    )

    movement_z = zscore(
        [
            row["avg_movement"]
            for row in records
        ]
    )


    raw_activity = []


    for index in range(
        len(records)
    ):

        activity_score = (
            0.30
            *
            speed_z[index]
            +
            0.20
            *
            max_speed_z[index]
            +
            0.20
            *
            acc_z[index]
            +
            0.15
            *
            direction_z[index]
            +
            0.15
            *
            movement_z[index]
        )

        raw_activity.append(
            activity_score
        )


    smoothed_activity = moving_average(
        raw_activity,
        SMOOTH_RADIUS
    )


    # =====================================================
    # TRANSITION SCORE
    # =====================================================

    transition_candidates = []


    margin_frames = int(
        EDGE_MARGIN_SECONDS
        *
        fps
    )


    for index in range(
        1,
        len(records)
    ):

        current = records[
            index
        ].copy()

        current_activity = (
            smoothed_activity[
                index
            ]
        )

        previous_activity = (
            smoothed_activity[
                index - 1
            ]
        )

        transition_score = (
            current_activity
            -
            previous_activity
        )


        # We care only about upward activity transitions.
        if transition_score <= 0:
            continue


        if (
            current["start_frame"]
            <
            sequence["start_frame"]
            +
            margin_frames
        ):
            continue


        if (
            current["start_frame"]
            >
            sequence["end_frame"]
            -
            margin_frames
        ):
            continue


        current[
            "activity_score"
        ] = current_activity

        current[
            "previous_activity_score"
        ] = previous_activity

        current[
            "transition_score"
        ] = transition_score


        transition_candidates.append(
            current
        )


    # =====================================================
    # SELECT DISTINCT TOP CANDIDATES
    # =====================================================

    ranked = sorted(
        transition_candidates,
        key=lambda row:
            row[
                "transition_score"
            ],
        reverse=True
    )


    selected = []

    min_gap_frames = int(
        MIN_GAP_SECONDS
        *
        fps
    )


    for candidate in ranked:

        candidate_frame = candidate[
            "start_frame"
        ]

        too_close = any(
            abs(
                candidate_frame
                -
                existing[
                    "start_frame"
                ]
            )
            <
            min_gap_frames
            for existing in selected
        )

        if too_close:
            continue


        selected.append(
            candidate
        )


        if len(
            selected
        ) >= TOP_PER_SEQUENCE:
            break


    selected = sorted(
        selected,
        key=lambda row:
            row[
                "transition_score"
            ],
        reverse=True
    )


    # =====================================================
    # PRINT
    # =====================================================

    print()
    print(
        f"{sequence_id} | "
        f"{sequence['scene_name']} | "
        f"Frames "
        f"{sequence['start_frame']}-"
        f"{sequence['end_frame']}"
    )

    if not selected:

        print(
            "  No positive candidate found."
        )


    for rank, candidate in enumerate(
        selected,
        start=1
    ):

        print(
            f"  {rank}. "
            f"Frame "
            f"{candidate['start_frame']} | "
            f"Activity "
            f"{candidate['activity_score']:.3f} | "
            f"Increase "
            f"{candidate['transition_score']:.3f}"
        )


        all_candidates.append(
            {
                "sequence_id":
                    sequence_id,

                "scene_id":
                    sequence["scene_id"],

                "scene_name":
                    sequence["scene_name"],

                "sequence_start_frame":
                    sequence["start_frame"],

                "sequence_end_frame":
                    sequence["end_frame"],

                "candidate_rank":
                    rank,

                "candidate_start_frame":
                    candidate["start_frame"],

                "candidate_end_frame":
                    candidate["end_frame"],

                "timestamp_sec":
                    (
                        candidate[
                            "start_frame"
                        ]
                        - 1
                    )
                    /
                    fps,

                "person_count":
                    candidate[
                        "person_count"
                    ],

                "avg_speed_norm_s":
                    candidate[
                        "avg_speed"
                    ],

                "max_speed_norm_s":
                    candidate[
                        "max_speed"
                    ],

                "avg_acceleration_norm_s2":
                    candidate[
                        "avg_acceleration"
                    ],

                "avg_direction_change_deg":
                    candidate[
                        "avg_direction_change"
                    ],

                "avg_movement_norm":
                    candidate[
                        "avg_movement"
                    ],

                "motion_ratio":
                    candidate[
                        "motion_ratio"
                    ],

                "previous_activity_score":
                    candidate[
                        "previous_activity_score"
                    ],

                "activity_score":
                    candidate[
                        "activity_score"
                    ],

                "transition_score":
                    candidate[
                        "transition_score"
                    ]
            }
        )


# =========================================================
# SAVE
# =========================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = [
    "sequence_id",
    "scene_id",
    "scene_name",
    "sequence_start_frame",
    "sequence_end_frame",
    "candidate_rank",
    "candidate_start_frame",
    "candidate_end_frame",
    "timestamp_sec",
    "person_count",
    "avg_speed_norm_s",
    "max_speed_norm_s",
    "avg_acceleration_norm_s2",
    "avg_direction_change_deg",
    "avg_movement_norm",
    "motion_ratio",
    "previous_activity_score",
    "activity_score",
    "transition_score",
    "ground_truth_status",
    "training_allowed"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    for row in all_candidates:

        output_row = row.copy()

        output_row[
            "timestamp_sec"
        ] = round(
            output_row[
                "timestamp_sec"
            ],
            4
        )

        numeric_fields = [
            "avg_speed_norm_s",
            "max_speed_norm_s",
            "avg_acceleration_norm_s2",
            "avg_direction_change_deg",
            "avg_movement_norm",
            "motion_ratio",
            "previous_activity_score",
            "activity_score",
            "transition_score"
        ]


        for field in numeric_fields:

            output_row[field] = round(
                output_row[field],
                6
            )


        output_row[
            "ground_truth_status"
        ] = "AUTOMATIC_CANDIDATE_ONLY"

        output_row[
            "training_allowed"
        ] = "NO"


        writer.writerow(
            output_row
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("ABNORMAL CANDIDATE DETECTION COMPLETE")
print("==========================================")

print(
    f"Sequences analyzed : "
    f"{len(sequences)}"
)

print(
    f"Candidates saved   : "
    f"{len(all_candidates)}"
)

print(
    f"Output             : "
    f"{OUTPUT_CSV}"
)

print("------------------------------------------")

print(
    "These are NORMAL-to-high-activity "
    "transition candidates only."
)

print(
    "They are NOT official UMN "
    "abnormal ground truth."
)

print(
    "Training allowed: NO"
)

print("==========================================")