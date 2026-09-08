import csv
import math
from collections import defaultdict
from pathlib import Path


# =========================================================
# CONFIG
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
    / "umn_scene3_fine_transition.csv"
)

START_FRAME = 7600
END_FRAME = 7670

SMOOTH_RADIUS = 2


# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
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


def smooth(values, radius):
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
            mean(
                values[start:end]
            )
        )

    return output


# =========================================================
# CHECK FILE
# =========================================================

if not INPUT_CSV.exists():

    print()
    print("ERROR: Input CSV not found.")
    print(INPUT_CSV)

    raise SystemExit


# =========================================================
# READ FRAME FEATURES
# =========================================================

frame_data = defaultdict(
    lambda: {
        "track_ids": set(),
        "speed": [],
        "acceleration": [],
        "direction_change": [],
        "movement": [],
        "motion_rows": 0,
        "total_rows": 0
    }
)


fps = 30.0


with open(
    INPUT_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        frame_number = safe_int(
            row["frame_number"]
        )

        if not (
            START_FRAME
            <= frame_number
            <= END_FRAME
        ):
            continue

        fps = safe_float(
            row["video_fps"],
            30.0
        )

        data = frame_data[
            frame_number
        ]

        data[
            "track_ids"
        ].add(
            safe_int(
                row["track_id"]
            )
        )

        data[
            "total_rows"
        ] += 1

        motion_valid = safe_int(
            row["motion_valid"]
        )

        if motion_valid != 1:
            continue

        data[
            "motion_rows"
        ] += 1

        data[
            "speed"
        ].append(
            safe_float(
                row["speed_norm_s"]
            )
        )

        data[
            "acceleration"
        ].append(
            abs(
                safe_float(
                    row["acceleration_norm_s2"]
                )
            )
        )

        data[
            "direction_change"
        ].append(
            safe_float(
                row["direction_change_deg"]
            )
        )

        data[
            "movement"
        ].append(
            safe_float(
                row["movement_distance_norm"]
            )
        )


# =========================================================
# BUILD FRAME RECORDS
# =========================================================

records = []


for frame_number in range(
    START_FRAME,
    END_FRAME + 1
):

    data = frame_data[
        frame_number
    ]

    records.append(
        {
            "frame_number": frame_number,

            "timestamp_sec":
                (frame_number - 1) / fps,

            "person_count":
                len(
                    data[
                        "track_ids"
                    ]
                ),

            "avg_speed":
                mean(
                    data[
                        "speed"
                    ]
                ),

            "max_speed":
                max(
                    data[
                        "speed"
                    ]
                )
                if data[
                    "speed"
                ]
                else 0.0,

            "avg_acceleration":
                mean(
                    data[
                        "acceleration"
                    ]
                ),

            "avg_direction_change":
                mean(
                    data[
                        "direction_change"
                    ]
                ),

            "avg_movement":
                mean(
                    data[
                        "movement"
                    ]
                ),

            "motion_ratio":
                (
                    data[
                        "motion_rows"
                    ]
                    /
                    data[
                        "total_rows"
                    ]

                    if data[
                        "total_rows"
                    ] > 0

                    else 0.0
                )
        }
    )


# =========================================================
# NORMALIZE
# =========================================================

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


# =========================================================
# ACTIVITY SCORE
# =========================================================

raw_activity = []


for index in range(
    len(records)
):

    score = (
        0.30 * speed_z[index]
        +
        0.20 * max_speed_z[index]
        +
        0.20 * acc_z[index]
        +
        0.15 * direction_z[index]
        +
        0.15 * movement_z[index]
    )

    raw_activity.append(
        score
    )


smoothed_activity = smooth(
    raw_activity,
    SMOOTH_RADIUS
)


# =========================================================
# TRANSITION SCORE
# =========================================================

for index, row in enumerate(
    records
):

    row[
        "activity_score"
    ] = smoothed_activity[
        index
    ]

    if index == 0:

        row[
            "transition_score"
        ] = 0.0

    else:

        row[
            "transition_score"
        ] = (
            smoothed_activity[
                index
            ]
            -
            smoothed_activity[
                index - 1
            ]
        )


# =========================================================
# RANK POSITIVE TRANSITIONS
# =========================================================

positive_transitions = [
    row
    for row in records
    if row[
        "transition_score"
    ] > 0
]


ranked = sorted(
    positive_transitions,
    key=lambda row:
        row[
            "transition_score"
        ],
    reverse=True
)


top = ranked[:10]


# =========================================================
# SAVE CSV
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

    writer.writerow(
        [
            "frame_number",
            "timestamp_sec",
            "person_count",
            "avg_speed_norm_s",
            "max_speed_norm_s",
            "avg_acceleration_norm_s2",
            "avg_direction_change_deg",
            "avg_movement_norm",
            "motion_ratio",
            "activity_score",
            "transition_score"
        ]
    )

    for row in records:

        writer.writerow(
            [
                row["frame_number"],
                round(
                    row["timestamp_sec"],
                    4
                ),
                row["person_count"],
                round(
                    row["avg_speed"],
                    8
                ),
                round(
                    row["max_speed"],
                    8
                ),
                round(
                    row["avg_acceleration"],
                    8
                ),
                round(
                    row[
                        "avg_direction_change"
                    ],
                    4
                ),
                round(
                    row["avg_movement"],
                    8
                ),
                round(
                    row["motion_ratio"],
                    6
                ),
                round(
                    row["activity_score"],
                    6
                ),
                round(
                    row["transition_score"],
                    6
                )
            ]
        )


# =========================================================
# PRINT RESULT
# =========================================================

print()
print("==========================================")
print("UMN SCENE 3 FINE TRANSITION ANALYSIS")
print("==========================================")
print(
    f"Frame Range : "
    f"{START_FRAME} - {END_FRAME}"
)
print(
    f"Frames      : "
    f"{len(records)}"
)
print("------------------------------------------")
print("TOP POSITIVE TRANSITIONS")
print("------------------------------------------")


for index, row in enumerate(
    top,
    start=1
):

    print(
        f"{index:02d}. "
        f"Frame {row['frame_number']} | "
        f"{row['timestamp_sec']:.2f}s | "
        f"Persons {row['person_count']:2d} | "
        f"Activity {row['activity_score']:.3f} | "
        f"Increase {row['transition_score']:.3f}"
    )


print()
print("==========================================")
print("ANALYSIS COMPLETE")
print("==========================================")
print(
    f"Output : {OUTPUT_CSV}"
)
print("------------------------------------------")
print(
    "This is only fine-grained motion "
    "transition analysis."
)
print(
    "It does not create official "
    "ground-truth labels."
)
print("==========================================")