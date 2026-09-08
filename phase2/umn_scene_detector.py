import csv
from pathlib import Path

import cv2


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "public_datasets"
    / "umn"
    / "Crowd-Activity-All.avi"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_scene_candidates.csv"
)

# Histogram difference threshold.
# This is only for finding candidate scene cuts,
# NOT for risk prediction or model labels.
SCENE_CHANGE_THRESHOLD = 0.55

# To avoid detecting the same cut multiple times.
MIN_GAP_FRAMES = 30


# =========================================================
# CHECK VIDEO
# =========================================================

if not VIDEO_PATH.exists():

    print("ERROR: UMN video not found.")
    print(VIDEO_PATH)
    raise SystemExit


video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():

    print("ERROR: Could not open UMN video.")
    raise SystemExit


fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("==========================================")
print("UMN SCENE CHANGE DETECTOR")
print("==========================================")
print(f"Video        : {VIDEO_PATH.name}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Threshold    : {SCENE_CHANGE_THRESHOLD}")
print("------------------------------------------")


# =========================================================
# HISTOGRAM FUNCTION
# =========================================================

def get_histogram(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [50, 60],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        hist,
        hist
    )

    return hist


# =========================================================
# READ FIRST FRAME
# =========================================================

ret, previous_frame = video.read()

if not ret:

    print("ERROR: Could not read first frame.")
    video.release()
    raise SystemExit


previous_hist = get_histogram(
    previous_frame
)

frame_number = 1

last_scene_frame = -MIN_GAP_FRAMES

scene_candidates = []


# =========================================================
# PROCESS VIDEO
# =========================================================

while True:

    ret, frame = video.read()

    if not ret:
        break


    frame_number += 1


    current_hist = get_histogram(
        frame
    )


    # Correlation:
    # 1.0 = very similar
    # lower values = more different

    similarity = cv2.compareHist(
        previous_hist,
        current_hist,
        cv2.HISTCMP_CORREL
    )


    difference_score = (
        1.0 - similarity
    )


    # =====================================================
    # SCENE CUT CANDIDATE
    # =====================================================

    if (
        difference_score
        >=
        SCENE_CHANGE_THRESHOLD
        and
        (
            frame_number
            -
            last_scene_frame
        )
        >=
        MIN_GAP_FRAMES
    ):

        timestamp_sec = (
            frame_number / fps
            if fps > 0
            else 0.0
        )


        scene_candidates.append(
            {
                "frame_number": frame_number,
                "timestamp_sec": timestamp_sec,
                "difference_score": difference_score
            }
        )


        last_scene_frame = frame_number


        print(
            f"Candidate -> "
            f"Frame {frame_number}, "
            f"Time {timestamp_sec:.2f}s, "
            f"Score {difference_score:.4f}"
        )


    previous_hist = current_hist


    if frame_number % 1000 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames..."
        )


video.release()


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

    writer = csv.writer(file)

    writer.writerow([
        "candidate_id",
        "frame_number",
        "timestamp_sec",
        "difference_score"
    ])


    for index, candidate in enumerate(
        scene_candidates,
        start=1
    ):

        writer.writerow([
            f"SCENE_CAND_{index:03d}",

            candidate[
                "frame_number"
            ],

            round(
                candidate[
                    "timestamp_sec"
                ],
                4
            ),

            round(
                candidate[
                    "difference_score"
                ],
                6
            )
        ])


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==========================================")
print("SCENE DETECTION COMPLETE")
print("==========================================")
print(
    f"Scene Candidates : "
    f"{len(scene_candidates)}"
)
print(
    f"Output CSV       : "
    f"{OUTPUT_CSV}"
)
print("------------------------------------------")
print(
    "These are candidate scene boundaries only."
)
print(
    "They must be visually verified before "
    "sequence splitting."
)
print("==========================================")