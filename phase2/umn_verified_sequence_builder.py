from pathlib import Path
import csv


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_verified_sequences.csv"
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# VIDEO INFORMATION
# =========================================================

VIDEO_NAME = "Crowd-Activity-All.avi"
TOTAL_FRAMES = 7739
FPS = 30.0


# =========================================================
# VERIFIED / RECONSTRUCTED SEQUENCES
# =========================================================
#
# IMPORTANT:
# These are NOT claimed as official UMN per-sequence
# annotations.
#
# They were reconstructed from the local concatenated
# UMN video using:
#
# 1. automatic reset/cut detection
# 2. local consecutive-frame refinement
# 3. visual verification of crowd reset
# 4. confirmed major scene changes
#
# =========================================================

SEQUENCES = [

    # -------------------------
    # SCENE 1 - LAWN
    # -------------------------

    {
        "sequence_id": "SEQ_01",
        "scene_id": "SCENE_01",
        "scene_name": "Lawn",
        "start_frame": 1,
        "end_frame": 625,
        "start_reason": "VIDEO_START",
        "boundary_evidence": "VIDEO_START"
    },

    {
        "sequence_id": "SEQ_02",
        "scene_id": "SCENE_01",
        "scene_name": "Lawn",
        "start_frame": 626,
        "end_frame": 1453,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },


    # -------------------------
    # SCENE 2 - INDOOR
    # -------------------------

    {
        "sequence_id": "SEQ_03",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 1454,
        "end_frame": 2002,
        "start_reason": "SCENE_CHANGE",
        "boundary_evidence":
            "HARD_SCENE_CUT_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_04",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 2003,
        "end_frame": 2687,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_05",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 2688,
        "end_frame": 3455,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_06",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 3456,
        "end_frame": 4034,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_07",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 4035,
        "end_frame": 4929,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_08",
        "scene_id": "SCENE_02",
        "scene_name": "Indoor",
        "start_frame": 4930,
        "end_frame": 5596,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },


    # -------------------------
    # SCENE 3 - OUTDOOR PLAZA
    # -------------------------

    {
        "sequence_id": "SEQ_09",
        "scene_id": "SCENE_03",
        "scene_name": "Outdoor_Plaza",
        "start_frame": 5597,
        "end_frame": 6254,
        "start_reason": "SCENE_CHANGE",
        "boundary_evidence":
            "HARD_SCENE_CUT_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_10",
        "scene_id": "SCENE_03",
        "scene_name": "Outdoor_Plaza",
        "start_frame": 6255,
        "end_frame": 6931,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    },

    {
        "sequence_id": "SEQ_11",
        "scene_id": "SCENE_03",
        "scene_name": "Outdoor_Plaza",
        "start_frame": 6932,
        "end_frame": 7739,
        "start_reason": "CROWD_RESET",
        "boundary_evidence":
            "AUTO_RECONSTRUCTED_VISUALLY_VERIFIED"
    }
]


# =========================================================
# VALIDATION
# =========================================================

print()
print("==========================================")
print("UMN VERIFIED SEQUENCE BUILDER")
print("==========================================")

print(f"Video              : {VIDEO_NAME}")
print(f"Total Frames       : {TOTAL_FRAMES}")
print(f"FPS                : {FPS}")
print(f"Expected Sequences : 11")
print("------------------------------------------")


if len(SEQUENCES) != 11:

    raise ValueError(
        "Expected exactly 11 reconstructed sequences."
    )


# Check first and last frame.

if SEQUENCES[0]["start_frame"] != 1:

    raise ValueError(
        "First sequence must start at frame 1."
    )


if SEQUENCES[-1]["end_frame"] != TOTAL_FRAMES:

    raise ValueError(
        "Last sequence must end at frame 7739."
    )


# Check sequence continuity.

for index in range(1, len(SEQUENCES)):

    previous = SEQUENCES[index - 1]
    current = SEQUENCES[index]

    expected_start = (
        previous["end_frame"] + 1
    )

    if current["start_frame"] != expected_start:

        raise ValueError(
            f"Frame gap/overlap detected between "
            f"{previous['sequence_id']} and "
            f"{current['sequence_id']}."
        )


# Check sequence lengths.

for sequence in SEQUENCES:

    sequence["frame_count"] = (
        sequence["end_frame"]
        -
        sequence["start_frame"]
        +
        1
    )

    sequence["start_time_sec"] = (
        sequence["start_frame"] - 1
    ) / FPS

    sequence["end_time_sec"] = (
        sequence["end_frame"] - 1
    ) / FPS

    sequence["duration_sec"] = (
        sequence["frame_count"]
        / FPS
    )


# Verify total frame coverage.

covered_frames = sum(
    sequence["frame_count"]
    for sequence in SEQUENCES
)


if covered_frames != TOTAL_FRAMES:

    raise ValueError(
        f"Frame coverage error. "
        f"Expected {TOTAL_FRAMES}, "
        f"got {covered_frames}."
    )


# =========================================================
# DISPLAY
# =========================================================

for sequence in SEQUENCES:

    print(
        f"{sequence['sequence_id']} | "
        f"{sequence['scene_id']} | "
        f"{sequence['scene_name']:<14} | "
        f"Frames "
        f"{sequence['start_frame']:4d} - "
        f"{sequence['end_frame']:4d} | "
        f"{sequence['frame_count']:4d} frames | "
        f"{sequence['duration_sec']:.2f}s"
    )


# =========================================================
# SAVE CSV
# =========================================================

fieldnames = [
    "video_name",
    "sequence_id",
    "scene_id",
    "scene_name",
    "start_frame",
    "end_frame",
    "frame_count",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "start_reason",
    "boundary_evidence",
    "sequence_status",
    "official_boundary_claim",
    "abnormal_ground_truth_status",
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


    for sequence in SEQUENCES:

        writer.writerow(
            {
                "video_name":
                    VIDEO_NAME,

                "sequence_id":
                    sequence["sequence_id"],

                "scene_id":
                    sequence["scene_id"],

                "scene_name":
                    sequence["scene_name"],

                "start_frame":
                    sequence["start_frame"],

                "end_frame":
                    sequence["end_frame"],

                "frame_count":
                    sequence["frame_count"],

                "start_time_sec":
                    round(
                        sequence["start_time_sec"],
                        4
                    ),

                "end_time_sec":
                    round(
                        sequence["end_time_sec"],
                        4
                    ),

                "duration_sec":
                    round(
                        sequence["duration_sec"],
                        4
                    ),

                "start_reason":
                    sequence["start_reason"],

                "boundary_evidence":
                    sequence["boundary_evidence"],

                "sequence_status":
                    "RECONSTRUCTED_AND_VISUALLY_VERIFIED",

                "official_boundary_claim":
                    "NO",

                "abnormal_ground_truth_status":
                    "NOT_VERIFIED",

                "training_allowed":
                    "NO"
            }
        )


# =========================================================
# SUMMARY
# =========================================================

print("------------------------------------------")

print(
    f"Sequences          : {len(SEQUENCES)}"
)

print(
    f"Covered Frames     : {covered_frames}"
)

print(
    f"Expected Frames    : {TOTAL_FRAMES}"
)

print("------------------------------------------")

print(
    f"Output : {OUTPUT_CSV}"
)

print("------------------------------------------")

print(
    "Sequence boundaries are reconstructed "
    "and visually verified for the local video."
)

print(
    "They are NOT claimed as official UMN "
    "per-sequence annotations."
)

print(
    "Abnormal ground truth is NOT verified."
)

print(
    "Training allowed: NO"
)

print("==========================================")