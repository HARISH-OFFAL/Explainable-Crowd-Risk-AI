from pathlib import Path
import csv
import cv2
import numpy as np


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "public_datasets"
    / "umn"
    / "Crowd-Activity-All.avi"
)

SEQUENCE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_verified_sequences.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_red_annotation_scan.csv"
)


# =========================================================
# CONFIG
# =========================================================

# Only inspect the upper text band.
TEXT_TOP_RATIO = 0.00
TEXT_BOTTOM_RATIO = 0.13

# Ignore first 2 seconds of every reconstructed sequence.
START_IGNORE_SECONDS = 2.0

# Red text HSV ranges.
RED1_LOW = np.array([0, 120, 100], dtype=np.uint8)
RED1_HIGH = np.array([10, 255, 255], dtype=np.uint8)

RED2_LOW = np.array([170, 120, 100], dtype=np.uint8)
RED2_HIGH = np.array([180, 255, 255], dtype=np.uint8)

# Annotation normally contains many small red text pixels.
MIN_RED_PIXELS = 300
MAX_RED_PIXELS = 1200

# Text must remain visible for half a second.
SUSTAINED_FRAMES = 15


# =========================================================
# CHECK FILES
# =========================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not SEQUENCE_CSV.exists():
    raise FileNotFoundError(
        f"Sequence CSV not found: {SEQUENCE_CSV}"
    )


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
                "sequence_id": row["sequence_id"],
                "scene_name": row["scene_name"],
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
            }
        )


# =========================================================
# FRAME -> SEQUENCE
# =========================================================

def find_sequence(frame_number):

    for sequence in sequences:

        if (
            sequence["start_frame"]
            <= frame_number
            <= sequence["end_frame"]
        ):
            return sequence

    return None


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open UMN video."
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

ignore_frames = int(
    START_IGNORE_SECONDS * fps
)


print()
print("==========================================")
print("UMN REFINED RED TEXT SCAN")
print("==========================================")
print(f"Video          : {VIDEO_PATH}")
print(f"FPS            : {fps}")
print(f"Total Frames   : {total_frames}")
print(f"Sequences      : {len(sequences)}")
print(f"Start Ignore   : {ignore_frames} frames")
print(f"Sustained Need : {SUSTAINED_FRAMES} frames")
print("------------------------------------------")


# =========================================================
# SIGNAL STORAGE
# =========================================================

signals = {
    sequence["sequence_id"]: []
    for sequence in sequences
}


# =========================================================
# SCAN VIDEO
# =========================================================

frame_number = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    sequence = find_sequence(
        frame_number
    )

    if sequence is None:
        continue


    height, width = frame.shape[:2]

    y1 = int(
        height * TEXT_TOP_RATIO
    )

    y2 = int(
        height * TEXT_BOTTOM_RATIO
    )

    text_roi = frame[
        y1:y2,
        :
    ]


    hsv = cv2.cvtColor(
        text_roi,
        cv2.COLOR_BGR2HSV
    )


    mask1 = cv2.inRange(
        hsv,
        RED1_LOW,
        RED1_HIGH
    )

    mask2 = cv2.inRange(
        hsv,
        RED2_LOW,
        RED2_HIGH
    )


    mask = cv2.bitwise_or(
        mask1,
        mask2
    )


    red_pixels = int(
        cv2.countNonZero(mask)
    )


    relative_frame = (
        frame_number
        -
        sequence["start_frame"]
    )


    # Ignore sequence beginning.
    eligible = (
        relative_frame
        >=
        ignore_frames
    )


    signal = (
        1
        if (
            eligible
            and
            MIN_RED_PIXELS
            <= red_pixels
            <= MAX_RED_PIXELS
        )
        else 0
    )


    signals[
        sequence["sequence_id"]
    ].append(
        {
            "frame_number": frame_number,
            "red_pixels": red_pixels,
            "signal": signal,
        }
    )


    if frame_number % 500 == 0:

        print(
            f"Scanned "
            f"{frame_number}/{total_frames}"
        )


cap.release()


# =========================================================
# FIND FIRST SUSTAINED TEXT SIGNAL
# =========================================================

results = []


for sequence in sequences:

    sequence_id = sequence[
        "sequence_id"
    ]

    rows = signals[
        sequence_id
    ]


    first_frame = None

    detection_red_pixels = 0


    for index in range(
        len(rows)
        -
        SUSTAINED_FRAMES
        +
        1
    ):

        window = rows[
            index:
            index + SUSTAINED_FRAMES
        ]


        if all(
            row["signal"] == 1
            for row in window
        ):

            first_frame = (
                window[0][
                    "frame_number"
                ]
            )

            detection_red_pixels = int(
                np.mean(
                    [
                        row["red_pixels"]
                        for row in window
                    ]
                )
            )

            break


    if first_frame is None:

        timestamp = ""

        status = (
            "NO_STABLE_TEXT_SIGNAL"
        )

    else:

        timestamp = (
            first_frame - 1
        ) / fps

        status = (
            "STABLE_RED_TEXT_CANDIDATE"
        )


    results.append(
        {
            "sequence_id":
                sequence_id,

            "scene_name":
                sequence["scene_name"],

            "sequence_start_frame":
                sequence["start_frame"],

            "sequence_end_frame":
                sequence["end_frame"],

            "first_stable_red_text_frame":
                (
                    first_frame
                    if first_frame
                    is not None
                    else ""
                ),

            "timestamp_sec":
                (
                    round(timestamp, 4)
                    if timestamp != ""
                    else ""
                ),

            "mean_red_pixels_at_detection":
                detection_red_pixels,

            "scan_status":
                status,

            "interpretation":
                "EMBEDDED_TEXT_CANDIDATE_ONLY",

            "official_ground_truth_claim":
                "NO",

            "training_allowed":
                "NO",
        }
    )


# =========================================================
# SAVE
# =========================================================

fieldnames = [
    "sequence_id",
    "scene_name",
    "sequence_start_frame",
    "sequence_end_frame",
    "first_stable_red_text_frame",
    "timestamp_sec",
    "mean_red_pixels_at_detection",
    "scan_status",
    "interpretation",
    "official_ground_truth_claim",
    "training_allowed",
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

    writer.writerows(
        results
    )


# =========================================================
# DISPLAY
# =========================================================

print()
print("------------------------------------------")
print("REFINED RESULTS")
print("------------------------------------------")


found_count = 0


for row in results:

    frame = row[
        "first_stable_red_text_frame"
    ]


    if frame == "":

        print(
            f"{row['sequence_id']} | "
            f"{row['scene_name']:<14} | "
            f"NO stable text signal"
        )

    else:

        found_count += 1

        print(
            f"{row['sequence_id']} | "
            f"{row['scene_name']:<14} | "
            f"Frame {frame} | "
            f"Red Pixels "
            f"{row['mean_red_pixels_at_detection']}"
        )


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("REFINED RED TEXT SCAN COMPLETE")
print("==========================================")

print(
    f"Sequences scanned : "
    f"{len(sequences)}"
)

print(
    f"Signals found     : "
    f"{found_count}"
)

print(
    f"No signal         : "
    f"{len(sequences) - found_count}"
)

print(
    f"Output            : "
    f"{OUTPUT_CSV}"
)

print("------------------------------------------")

print(
    "This scan only searches for a stable "
    "embedded red-text-like signal."
)

print(
    "No official ground-truth claim is made."
)

print(
    "Training allowed: NO"
)

print("==========================================")
