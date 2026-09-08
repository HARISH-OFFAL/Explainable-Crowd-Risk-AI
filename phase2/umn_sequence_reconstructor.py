import csv
import math
from collections import defaultdict
from pathlib import Path


# =========================================================
# PATHS
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
    / "umn_reconstructed_sequence_candidates.csv"
)


# =========================================================
# VIDEO / SCENE INFORMATION
# =========================================================

FPS = 30.0
TOTAL_FRAMES = 7739

SCENES = [
    {
        "scene_id": "SCENE_01",
        "start_frame": 1,
        "end_frame": 1453,
        "expected_sequences": 2
    },
    {
        "scene_id": "SCENE_02",
        "start_frame": 1454,
        "end_frame": 5596,
        "expected_sequences": 6
    },
    {
        "scene_id": "SCENE_03",
        "start_frame": 5597,
        "end_frame": 7739,
        "expected_sequences": 3
    }
]

# Within each scene:
# expected internal boundaries = sequences - 1
#
# Scene 1 -> 1
# Scene 2 -> 5
# Scene 3 -> 2


# =========================================================
# CONFIG
# =========================================================

SMOOTH_RADIUS = 5

# Compare average crowd behaviour before and after
# each candidate frame.
COMPARE_WINDOW = 30

# Minimum separation between selected boundaries.
MIN_GAP_FRAMES = 120

# Avoid selecting very beginning/end of scene.
EDGE_MARGIN = 60


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

    variance = mean(
        [
            (value - avg) ** 2
            for value in values
        ]
    )

    return math.sqrt(variance)


def normalize(values):
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

    for i in range(len(values)):

        start = max(
            0,
            i - radius
        )

        end = min(
            len(values),
            i + radius + 1
        )

        output.append(
            mean(values[start:end])
        )

    return output


# =========================================================
# CHECK INPUT
# =========================================================

if not INPUT_CSV.exists():

    print()
    print("ERROR: UMN movement feature CSV not found.")
    print(INPUT_CSV)

    raise SystemExit


# =========================================================
# READ PER-FRAME FEATURES
# =========================================================

frame_data = defaultdict(
    lambda: {
        "track_ids": set(),
        "speeds": [],
        "accelerations": [],
        "direction_changes": [],
        "movement": []
    }
)


print()
print("==========================================")
print("UMN SEQUENCE RECONSTRUCTOR")
print("==========================================")
print(f"Input : {INPUT_CSV}")
print("------------------------------------------")


with open(
    INPUT_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        frame = safe_int(
            row["frame_number"]
        )

        if not (
            1 <= frame <= TOTAL_FRAMES
        ):
            continue

        data = frame_data[
            frame
        ]

        data[
            "track_ids"
        ].add(
            safe_int(
                row["track_id"]
            )
        )

        motion_valid = safe_int(
            row["motion_valid"]
        )

        if motion_valid != 1:
            continue

        data[
            "speeds"
        ].append(
            safe_float(
                row["speed_norm_s"]
            )
        )

        data[
            "accelerations"
        ].append(
            abs(
                safe_float(
                    row["acceleration_norm_s2"]
                )
            )
        )

        data[
            "direction_changes"
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
# BUILD FRAME TABLE
# =========================================================

records = []


for frame in range(
    1,
    TOTAL_FRAMES + 1
):

    data = frame_data[
        frame
    ]

    records.append(
        {
            "frame": frame,

            "person_count":
                len(
                    data[
                        "track_ids"
                    ]
                ),

            "avg_speed":
                mean(
                    data[
                        "speeds"
                    ]
                ),

            "avg_acceleration":
                mean(
                    data[
                        "accelerations"
                    ]
                ),

            "avg_direction_change":
                mean(
                    data[
                        "direction_changes"
                    ]
                ),

            "avg_movement":
                mean(
                    data[
                        "movement"
                    ]
                )
        }
    )


# =========================================================
# SMOOTH GLOBAL FEATURES
# =========================================================

person_count_series = [
    row["person_count"]
    for row in records
]

speed_series = [
    row["avg_speed"]
    for row in records
]

acc_series = [
    row["avg_acceleration"]
    for row in records
]

direction_series = [
    row["avg_direction_change"]
    for row in records
]

movement_series = [
    row["avg_movement"]
    for row in records
]


smoothed_person = moving_average(
    person_count_series,
    SMOOTH_RADIUS
)

smoothed_speed = moving_average(
    speed_series,
    SMOOTH_RADIUS
)

smoothed_acc = moving_average(
    acc_series,
    SMOOTH_RADIUS
)

smoothed_direction = moving_average(
    direction_series,
    SMOOTH_RADIUS
)

smoothed_movement = moving_average(
    movement_series,
    SMOOTH_RADIUS
)


# =========================================================
# SCENE-WISE RECONSTRUCTION
# =========================================================

all_selected = []


for scene in SCENES:

    scene_id = scene[
        "scene_id"
    ]

    scene_start = scene[
        "start_frame"
    ]

    scene_end = scene[
        "end_frame"
    ]

    expected_internal = (
        scene[
            "expected_sequences"
        ]
        - 1
    )


    print()
    print(
        f"{scene_id} | "
        f"{scene_start}-{scene_end}"
    )

    print(
        f"Expected internal boundaries : "
        f"{expected_internal}"
    )


    candidates = []


    # =====================================================
    # COMPARE BEFORE / AFTER WINDOWS
    # =====================================================

    candidate_start = (
        scene_start
        +
        EDGE_MARGIN
        +
        COMPARE_WINDOW
    )

    candidate_end = (
        scene_end
        -
        EDGE_MARGIN
        -
        COMPARE_WINDOW
    )


    for frame in range(
        candidate_start,
        candidate_end + 1
    ):

        index = frame - 1


        before_start = max(
            scene_start - 1,
            index - COMPARE_WINDOW
        )

        before_end = index


        after_start = index

        after_end = min(
            scene_end,
            index + COMPARE_WINDOW
        )


        before_person = mean(
            smoothed_person[
                before_start:
                before_end
            ]
        )

        after_person = mean(
            smoothed_person[
                after_start:
                after_end
            ]
        )


        before_speed = mean(
            smoothed_speed[
                before_start:
                before_end
            ]
        )

        after_speed = mean(
            smoothed_speed[
                after_start:
                after_end
            ]
        )


        before_acc = mean(
            smoothed_acc[
                before_start:
                before_end
            ]
        )

        after_acc = mean(
            smoothed_acc[
                after_start:
                after_end
            ]
        )


        before_direction = mean(
            smoothed_direction[
                before_start:
                before_end
            ]
        )

        after_direction = mean(
            smoothed_direction[
                after_start:
                after_end
            ]
        )


        before_movement = mean(
            smoothed_movement[
                before_start:
                before_end
            ]
        )

        after_movement = mean(
            smoothed_movement[
                after_start:
                after_end
            ]
        )


        # =================================================
        # RESET FEATURES
        # =================================================
        #
        # In UMN, one sequence often ends with people
        # escaping and the next sequence restarts with
        # another crowd configuration.
        #
        # Therefore a sequence boundary may show:
        #
        # - sudden person-count reset
        # - speed drop/change
        # - motion reset
        # - acceleration reset
        # - direction change
        #
        # This is NOT a ground-truth score.
        # It is only a candidate ranking signal.
        # =================================================

        person_change = (
            after_person
            -
            before_person
        )

        speed_change = abs(
            after_speed
            -
            before_speed
        )

        acceleration_change = abs(
            after_acc
            -
            before_acc
        )

        direction_change = abs(
            after_direction
            -
            before_direction
        )

        movement_change = abs(
            after_movement
            -
            before_movement
        )


        candidates.append(
            {
                "scene_id": scene_id,
                "frame": frame,

                "before_person":
                    before_person,

                "after_person":
                    after_person,

                "person_change":
                    person_change,

                "speed_change":
                    speed_change,

                "acceleration_change":
                    acceleration_change,

                "direction_change":
                    direction_change,

                "movement_change":
                    movement_change
            }
        )


    # =====================================================
    # NORMALIZE CANDIDATE FEATURES
    # =====================================================

    person_z = normalize(
        [
            abs(
                item[
                    "person_change"
                ]
            )
            for item in candidates
        ]
    )

    speed_z = normalize(
        [
            item[
                "speed_change"
            ]
            for item in candidates
        ]
    )

    acc_z = normalize(
        [
            item[
                "acceleration_change"
            ]
            for item in candidates
        ]
    )

    direction_z = normalize(
        [
            item[
                "direction_change"
            ]
            for item in candidates
        ]
    )

    movement_z = normalize(
        [
            item[
                "movement_change"
            ]
            for item in candidates
        ]
    )


    # =====================================================
    # RESET SCORE
    # =====================================================

    for i, candidate in enumerate(
        candidates
    ):

        score = (
            0.40
            *
            person_z[i]
            +
            0.20
            *
            speed_z[i]
            +
            0.15
            *
            acc_z[i]
            +
            0.10
            *
            direction_z[i]
            +
            0.15
            *
            movement_z[i]
        )


        # Positive person reappearance after crowd exits
        # is useful evidence of a restart.

        if (
            candidate[
                "person_change"
            ] > 0
        ):

            score += 0.25


        candidate[
            "reset_score"
        ] = score


    # =====================================================
    # RANK
    # =====================================================

    ranked = sorted(
        candidates,
        key=lambda row:
            row[
                "reset_score"
            ],
        reverse=True
    )


    selected = []


    for candidate in ranked:

        frame = candidate[
            "frame"
        ]

        too_close = False


        for existing in selected:

            if abs(
                frame
                -
                existing[
                    "frame"
                ]
            ) < MIN_GAP_FRAMES:

                too_close = True
                break


        if too_close:
            continue


        selected.append(
            candidate
        )


        if len(
            selected
        ) >= expected_internal:

            break


    selected = sorted(
        selected,
        key=lambda row:
            row[
                "frame"
            ]
    )


    # =====================================================
    # PRINT SCENE RESULT
    # =====================================================

    for index, candidate in enumerate(
        selected,
        start=1
    ):

        print(
            f"  Candidate {index}: "
            f"Frame {candidate['frame']} | "
            f"Reset Score "
            f"{candidate['reset_score']:.3f} | "
            f"Persons "
            f"{candidate['before_person']:.1f}"
            f" -> "
            f"{candidate['after_person']:.1f}"
        )


        candidate[
            "boundary_rank"
        ] = index

        candidate[
            "status"
        ] = "AUTOMATIC_CANDIDATE"

        all_selected.append(
            candidate
        )


# =========================================================
# ADD KNOWN MAJOR SCENE BOUNDARIES
# =========================================================

known_scene_boundaries = [

    {
        "scene_id": "SCENE_CHANGE",
        "frame": 1454,
        "boundary_rank": 0,
        "before_person": 0.0,
        "after_person": 0.0,
        "person_change": 0.0,
        "speed_change": 0.0,
        "acceleration_change": 0.0,
        "direction_change": 0.0,
        "movement_change": 0.0,
        "reset_score": 0.0,
        "status": "VISUALLY_CONFIRMED_SCENE_CHANGE"
    },

    {
        "scene_id": "SCENE_CHANGE",
        "frame": 5597,
        "boundary_rank": 0,
        "before_person": 0.0,
        "after_person": 0.0,
        "person_change": 0.0,
        "speed_change": 0.0,
        "acceleration_change": 0.0,
        "direction_change": 0.0,
        "movement_change": 0.0,
        "reset_score": 0.0,
        "status": "VISUALLY_CONFIRMED_SCENE_CHANGE"
    }
]


all_output = (
    all_selected
    +
    known_scene_boundaries
)


all_output = sorted(
    all_output,
    key=lambda row:
        row[
            "frame"
        ]
)


# =========================================================
# SAVE
# =========================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = [

    "scene_id",

    "boundary_rank",

    "frame",

    "timestamp_sec",

    "before_person_count",

    "after_person_count",

    "person_count_change",

    "speed_change",

    "acceleration_change",

    "direction_change",

    "movement_change",

    "reset_score",

    "status",

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


    for row in all_output:

        frame = row[
            "frame"
        ]

        writer.writerow(
            {
                "scene_id":
                    row[
                        "scene_id"
                    ],

                "boundary_rank":
                    row[
                        "boundary_rank"
                    ],

                "frame":
                    frame,

                "timestamp_sec":
                    round(
                        (frame - 1)
                        /
                        FPS,
                        4
                    ),

                "before_person_count":
                    round(
                        row[
                            "before_person"
                        ],
                        4
                    ),

                "after_person_count":
                    round(
                        row[
                            "after_person"
                        ],
                        4
                    ),

                "person_count_change":
                    round(
                        row[
                            "person_change"
                        ],
                        4
                    ),

                "speed_change":
                    round(
                        row[
                            "speed_change"
                        ],
                        8
                    ),

                "acceleration_change":
                    round(
                        row[
                            "acceleration_change"
                        ],
                        8
                    ),

                "direction_change":
                    round(
                        row[
                            "direction_change"
                        ],
                        4
                    ),

                "movement_change":
                    round(
                        row[
                            "movement_change"
                        ],
                        8
                    ),

                "reset_score":
                    round(
                        row[
                            "reset_score"
                        ],
                        6
                    ),

                "status":
                    row[
                        "status"
                    ],

                "training_allowed":
                    "NO"
            }
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("RECONSTRUCTION COMPLETE")
print("==========================================")

print(
    f"Automatic internal candidates : "
    f"{len(all_selected)}"
)

print(
    f"Confirmed scene boundaries     : "
    f"{len(known_scene_boundaries)}"
)

print(
    f"Total boundary records         : "
    f"{len(all_output)}"
)

print("------------------------------------------")

print(
    "Expected internal boundaries:"
)

print(
    "Scene 1 = 1"
)

print(
    "Scene 2 = 5"
)

print(
    "Scene 3 = 2"
)

print("------------------------------------------")

print(
    f"Output : {OUTPUT_CSV}"
)

print(
    "Training allowed : NO"
)

print("------------------------------------------")

print(
    "Automatic candidates must be "
    "verified before final sequence labels."
)

print("==========================================")