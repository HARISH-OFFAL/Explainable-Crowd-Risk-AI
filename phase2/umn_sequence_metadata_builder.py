import csv
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "umn_sequence_metadata.csv"
)


# =========================================================
# LOCAL VIDEO INFORMATION
# =========================================================
#
# Verified from our downloaded UMN combined AVI:
#
# Total frames : 7739
# FPS          : 30
# Resolution   : 320 x 240
#
# Important:
# Frame numbering used by our project is 1-based.
# =========================================================

TOTAL_FRAMES = 7739
FPS = 30.0


# =========================================================
# SCENE STRUCTURE
# =========================================================
#
# Visual inspection of our local video showed:
#
# Scene 1 -> Lawn
# Scene 2 -> Indoor
# Scene 3 -> Outdoor Plaza
#
# Major visual changes:
#
# frame 1454 -> Lawn → Indoor
# frame 5597 -> Indoor → Plaza
#
# These are LOCAL project frame observations.
#
# We deliberately do NOT infer all 11 sequence boundaries
# from these two scene boundaries.
# =========================================================

SCENES = [

    {
        "scene_id": "SCENE_01",
        "scene_name": "Lawn",
        "start_frame": 1,
        "end_frame": 1453,
        "expected_sequences": 2,
        "boundary_status": "VISUALLY_CONFIRMED"
    },

    {
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 1454,
        "end_frame": 5596,
        "expected_sequences": 6,
        "boundary_status": "VISUALLY_CONFIRMED_LOCAL_INDEXING"
    },

    {
        "scene_id": "SCENE_03",
        "scene_name": "Outdoor_Plaza",
        "start_frame": 5597,
        "end_frame": 7739,
        "expected_sequences": 3,
        "boundary_status": "VISUALLY_CONFIRMED_LOCAL_INDEXING"
    }
]


# =========================================================
# HELPER
# =========================================================

def frame_to_time(frame_number):

    return (
        frame_number - 1
    ) / FPS


# =========================================================
# VALIDATION
# =========================================================

expected_total_sequences = sum(
    scene["expected_sequences"]
    for scene in SCENES
)


if expected_total_sequences != 11:

    raise ValueError(
        "Expected UMN sequence count must equal 11."
    )


# Check that scenes are continuous.

for index in range(
    1,
    len(SCENES)
):

    previous_scene = SCENES[
        index - 1
    ]

    current_scene = SCENES[
        index
    ]

    expected_start = (
        previous_scene["end_frame"]
        + 1
    )

    if (
        current_scene["start_frame"]
        != expected_start
    ):

        raise ValueError(
            "Scene frame ranges are not continuous."
        )


if SCENES[0]["start_frame"] != 1:

    raise ValueError(
        "First scene must start at frame 1."
    )


if SCENES[-1]["end_frame"] != TOTAL_FRAMES:

    raise ValueError(
        "Last scene must end at final video frame."
    )


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# WRITE METADATA
# =========================================================

fieldnames = [

    "scene_id",
    "scene_name",

    "scene_start_frame",
    "scene_end_frame",

    "scene_start_time_sec",
    "scene_end_time_sec",

    "expected_sequence_count",

    "sequence_boundaries_status",

    "ground_truth_status",

    "training_allowed",

    "notes"
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    for scene in SCENES:

        writer.writerow({

            "scene_id":
                scene["scene_id"],

            "scene_name":
                scene["scene_name"],

            "scene_start_frame":
                scene["start_frame"],

            "scene_end_frame":
                scene["end_frame"],

            "scene_start_time_sec":
                round(
                    frame_to_time(
                        scene["start_frame"]
                    ),
                    4
                ),

            "scene_end_time_sec":
                round(
                    frame_to_time(
                        scene["end_frame"]
                    ),
                    4
                ),

            "expected_sequence_count":
                scene[
                    "expected_sequences"
                ],

            "sequence_boundaries_status":
                "VERIFICATION_REQUIRED",

            "ground_truth_status":
                "NOT_OFFICIAL",

            "training_allowed":
                "NO",

            "notes":
                (
                    "Scene boundary based on local "
                    "video inspection. Individual "
                    "sequence boundaries and abnormal "
                    "ground-truth intervals are not "
                    "yet verified."
                )
        })


# =========================================================
# PRINT SUMMARY
# =========================================================

print()
print("==========================================")
print("UMN SEQUENCE METADATA BUILDER")
print("==========================================")

print(
    f"Total Video Frames      : "
    f"{TOTAL_FRAMES}"
)

print(
    f"FPS                     : "
    f"{FPS}"
)

print(
    f"Scenes                  : "
    f"{len(SCENES)}"
)

print(
    f"Expected Sequences      : "
    f"{expected_total_sequences}"
)

print("------------------------------------------")


for scene in SCENES:

    print(
        f"{scene['scene_id']} | "
        f"{scene['scene_name']:<15} | "
        f"Frames "
        f"{scene['start_frame']:4d}"
        f" - "
        f"{scene['end_frame']:4d}"
        f" | "
        f"Expected sequences: "
        f"{scene['expected_sequences']}"
    )


print("------------------------------------------")

print(
    "Individual sequence boundaries : "
    "VERIFICATION REQUIRED"
)

print(
    "Official abnormal ground truth : "
    "NOT VERIFIED"
)

print(
    "Training allowed               : "
    "NO"
)

print("------------------------------------------")

print(
    f"Output : {OUTPUT_FILE}"
)

print("==========================================")