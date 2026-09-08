import csv
import math
from collections import defaultdict
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_movement_features.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_scene3_transition_candidates.csv"
)

# Scene 3 visually confirmed range.
SCENE_START_FRAME = 5597
SCENE_END_FRAME = 7739

# 1-second windows.
WINDOW_SECONDS = 1.0

# Smoothing over neighbouring windows.
SMOOTHING_RADIUS = 2

# Minimum separation between transition candidates.
MIN_GAP_SECONDS = 8.0

# Number of strongest automatic candidates to display.
TOP_CANDIDATES = 10


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
# HELPERS
# =========================================================

def mean(values):

    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):

    if len(values) < 2:
        return 0.0

    avg = mean(values)

    variance = mean([
        (value - avg) ** 2
        for value in values
    ])

    return math.sqrt(variance)


def normalize_series(values):

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

    smoothed = []

    for index in range(len(values)):

        start = max(
            0,
            index - radius
        )

        end = min(
            len(values),
            index + radius + 1
        )

        smoothed.append(
            mean(
                values[start:end]
            )
        )

    return smoothed


# =========================================================
# CHECK FILE
# =========================================================

if not INPUT_CSV.exists():

    print()
    print("ERROR: UMN movement dataset not found.")
    print(INPUT_CSV)

    raise SystemExit


# =========================================================
# STORAGE
# =========================================================

windows = defaultdict(
    lambda: {
        "frames": set(),
        "track_ids": set(),
        "speed": [],
        "acceleration": [],
        "direction_change": [],
        "movement": [],
        "valid_motion_rows": 0,
        "total_rows": 0
    }
)


# =========================================================
# READ SCENE 3 FEATURES
# =========================================================

print()
print("==========================================")
print("UMN SCENE 3 AUTO TRANSITION DETECTOR")
print("==========================================")
print(f"Input CSV    : {INPUT_CSV}")
print(
    f"Scene Frames : "
    f"{SCENE_START_FRAME} - {SCENE_END_FRAME}"
)
print(f"Window       : {WINDOW_SECONDS:.1f} sec")
print("------------------------------------------")


rows_read = 0
rows_used = 0
fps_value = 30.0


with open(
    INPUT_CSV,
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

        if not (
            SCENE_START_FRAME
            <=
            frame_number
            <=
            SCENE_END_FRAME
        ):
            continue

        rows_used += 1

        fps_value = safe_float(
            row["video_fps"],
            30.0
        )

        timestamp = safe_float(
            row["timestamp_sec"]
        )

        scene_relative_time = (
            (frame_number - SCENE_START_FRAME)
            /
            fps_value
        )

        window_index = int(
            scene_relative_time
            //
            WINDOW_SECONDS
        )

        group = windows[
            window_index
        ]

        track_id = safe_int(
            row["track_id"]
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

        group[
            "speed"
        ].append(
            safe_float(
                row["speed_norm_s"]
            )
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


print(f"Rows Read     : {rows_read}")
print(f"Scene 3 Rows  : {rows_used}")
print(f"Windows       : {len(windows)}")
print("------------------------------------------")


# =========================================================
# BUILD WINDOW FEATURES
# =========================================================

records = []


for window_index in sorted(
    windows.keys()
):

    group = windows[
        window_index
    ]

    window_start_frame = (
        SCENE_START_FRAME
        +
        int(
            window_index
            *
            WINDOW_SECONDS
            *
            fps_value
        )
    )

    window_end_frame = min(
        SCENE_END_FRAME,
        window_start_frame
        +
        int(
            WINDOW_SECONDS
            *
            fps_value
        )
        -
        1
    )

    start_time_sec = (
        window_start_frame - 1
    ) / fps_value

    end_time_sec = (
        window_end_frame - 1
    ) / fps_value


    avg_speed = mean(
        group[
            "speed"
        ]
    )

    max_speed = (
        max(
            group[
                "speed"
            ]
        )
        if group[
            "speed"
        ]
        else 0.0
    )

    avg_acceleration = mean(
        group[
            "acceleration"
        ]
    )

    avg_direction_change = mean(
        group[
            "direction_change"
        ]
    )

    avg_movement = mean(
        group[
            "movement"
        ]
    )

    person_count = len(
        group[
            "track_ids"
        ]
    )

    motion_ratio = (

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


    records.append({

        "window_index": window_index,

        "start_frame": window_start_frame,

        "end_frame": window_end_frame,

        "start_time_sec": start_time_sec,

        "end_time_sec": end_time_sec,

        "person_count": person_count,

        "avg_speed": avg_speed,

        "max_speed": max_speed,

        "avg_acceleration": avg_acceleration,

        "avg_direction_change": avg_direction_change,

        "avg_movement": avg_movement,

        "motion_ratio": motion_ratio
    })


# =========================================================
# NORMALIZE FEATURES
# =========================================================

speed_values = [
    row["avg_speed"]
    for row in records
]

max_speed_values = [
    row["max_speed"]
    for row in records
]

acc_values = [
    row["avg_acceleration"]
    for row in records
]

direction_values = [
    row["avg_direction_change"]
    for row in records
]

movement_values = [
    row["avg_movement"]
    for row in records
]


z_speed = normalize_series(
    speed_values
)

z_max_speed = normalize_series(
    max_speed_values
)

z_acc = normalize_series(
    acc_values
)

z_direction = normalize_series(
    direction_values
)

z_movement = normalize_series(
    movement_values
)


# =========================================================
# ACTIVITY SCORE
# =========================================================

activity_scores = []


for index in range(
    len(records)
):

    # This is not a risk score.
    #
    # It is only a motion-activity ranking score used
    # for locating sudden changes in the video.

    activity = (
        0.30
        *
        z_speed[index]
        +
        0.20
        *
        z_max_speed[index]
        +
        0.20
        *
        z_acc[index]
        +
        0.15
        *
        z_direction[index]
        +
        0.15
        *
        z_movement[index]
    )

    activity_scores.append(
        activity
    )


smoothed_activity = moving_average(
    activity_scores,
    SMOOTHING_RADIUS
)


# =========================================================
# TRANSITION SCORE
# =========================================================

transition_candidates = []


for index in range(
    1,
    len(records)
):

    previous_activity = (
        smoothed_activity[
            index - 1
        ]
    )

    current_activity = (
        smoothed_activity[
            index
        ]
    )


    increase = (
        current_activity
        -
        previous_activity
    )


    # We care about upward motion/activity transitions.
    if increase <= 0:
        continue


    record = records[
        index
    ].copy()

    record[
        "activity_score"
    ] = current_activity

    record[
        "previous_activity_score"
    ] = previous_activity

    record[
        "transition_score"
    ] = increase


    transition_candidates.append(
        record
    )


# =========================================================
# RANK CANDIDATES
# =========================================================

ranked = sorted(
    transition_candidates,
    key=lambda row: row[
        "transition_score"
    ],
    reverse=True
)


selected = []


minimum_gap_frames = int(
    MIN_GAP_SECONDS
    *
    fps_value
)


for candidate in ranked:

    frame = candidate[
        "start_frame"
    ]


    too_close = False


    for existing in selected:

        if abs(
            frame
            -
            existing[
                "start_frame"
            ]
        ) < minimum_gap_frames:

            too_close = True
            break


    if too_close:
        continue


    selected.append(
        candidate
    )


    if len(
        selected
    ) >= TOP_CANDIDATES:

        break


selected = sorted(
    selected,
    key=lambda row: row[
        "start_frame"
    ]
)


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "candidate_id",

        "start_frame",
        "end_frame",

        "start_time_sec",
        "end_time_sec",

        "person_count",

        "avg_speed_norm_s",
        "max_speed_norm_s",

        "avg_acceleration_norm_s2",

        "avg_direction_change_deg",

        "avg_movement_norm",

        "motion_valid_ratio",

        "previous_activity_score",

        "activity_score",

        "transition_score"
    ])


    for index, row in enumerate(
        selected,
        start=1
    ):

        writer.writerow([

            f"TRANSITION_{index:03d}",

            row[
                "start_frame"
            ],

            row[
                "end_frame"
            ],

            round(
                row[
                    "start_time_sec"
                ],
                4
            ),

            round(
                row[
                    "end_time_sec"
                ],
                4
            ),

            row[
                "person_count"
            ],

            round(
                row[
                    "avg_speed"
                ],
                8
            ),

            round(
                row[
                    "max_speed"
                ],
                8
            ),

            round(
                row[
                    "avg_acceleration"
                ],
                8
            ),

            round(
                row[
                    "avg_direction_change"
                ],
                4
            ),

            round(
                row[
                    "avg_movement"
                ],
                8
            ),

            round(
                row[
                    "motion_ratio"
                ],
                6
            ),

            round(
                row[
                    "previous_activity_score"
                ],
                6
            ),

            round(
                row[
                    "activity_score"
                ],
                6
            ),

            round(
                row[
                    "transition_score"
                ],
                6
            )
        ])


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("==========================================")
print("AUTOMATIC TRANSITION CANDIDATES")
print("==========================================")


for index, row in enumerate(
    selected,
    start=1
):

    print(
        f"{index:02d}. "
        f"Frame {row['start_frame']:4d} | "
        f"{row['start_time_sec']:7.2f}s | "
        f"Activity {row['activity_score']:.3f} | "
        f"Increase {row['transition_score']:.3f}"
    )


print()
print("==========================================")
print("DETECTION COMPLETE")
print("==========================================")
print(
    f"Candidates : {len(selected)}"
)
print(
    f"Output     : {OUTPUT_CSV}"
)
print("------------------------------------------")
print(
    "These are automatic motion-transition "
    "candidates only."
)
print(
    "They are not official UMN ground-truth labels."
)
print(
    "No LOW/MEDIUM/HIGH crowd-risk labels "
    "were created."
)
print("==========================================")