from pathlib import Path
import csv

import cv2
import numpy as np


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
    / "umn_sequence_boundary_candidates.csv"
)

# We only rank candidate cuts.
TOP_CANDIDATES = 40

# Prevent nearby frames from appearing as separate cuts.
MIN_GAP_FRAMES = 20

# Known major scene transition references.
# These are NOT used to force detections.
KNOWN_MAJOR_BOUNDARIES = [
    1454,
    5598
]


# =========================================================
# CHECK VIDEO
# =========================================================

if not VIDEO_PATH.exists():

    print()
    print("ERROR: UMN video not found.")
    print(VIDEO_PATH)

    raise SystemExit


video = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not video.isOpened():

    print()
    print("ERROR: Could not open UMN video.")

    raise SystemExit


fps = float(
    video.get(cv2.CAP_PROP_FPS)
)

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print()
print("==========================================")
print("UMN SEQUENCE BOUNDARY DETECTOR")
print("==========================================")
print(f"Video        : {VIDEO_PATH.name}")
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print("------------------------------------------")
print(
    "Calculating frame-to-frame visual changes..."
)
print("------------------------------------------")


# =========================================================
# PREPROCESS FRAME
# =========================================================

def preprocess_frame(frame):

    # Small grayscale image is sufficient for
    # frame-to-frame cut detection.

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (160, 120)
    )

    return gray


# =========================================================
# HISTOGRAM
# =========================================================

def calculate_histogram(gray):

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [64],
        [0, 256]
    )

    cv2.normalize(
        hist,
        hist
    )

    return hist


# =========================================================
# FIRST FRAME
# =========================================================

ret, previous_frame = video.read()

if not ret:

    print("ERROR: Cannot read first frame.")

    video.release()

    raise SystemExit


previous_gray = preprocess_frame(
    previous_frame
)

previous_hist = calculate_histogram(
    previous_gray
)


frame_number = 1

scores = []


# =========================================================
# PROCESS ALL FRAMES
# =========================================================

while True:

    ret, frame = video.read()

    if not ret:
        break


    frame_number += 1


    current_gray = preprocess_frame(
        frame
    )

    current_hist = calculate_histogram(
        current_gray
    )


    # =====================================================
    # PIXEL DIFFERENCE
    # =====================================================

    pixel_difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    pixel_score = float(
        np.mean(pixel_difference)
        /
        255.0
    )


    # =====================================================
    # HISTOGRAM DIFFERENCE
    # =====================================================

    histogram_similarity = cv2.compareHist(
        previous_hist,
        current_hist,
        cv2.HISTCMP_CORREL
    )

    histogram_difference = float(
        1.0 - histogram_similarity
    )


    # Keep values within a sensible numeric range.
    histogram_difference = max(
        0.0,
        histogram_difference
    )


    # =====================================================
    # COMBINED VISUAL CHANGE SCORE
    # =====================================================

    # Pixel difference detects abrupt image changes.
    # Histogram difference provides global appearance
    # change information.
    #
    # This weighting is only used to rank candidate
    # editing boundaries. It has nothing to do with
    # crowd-risk classification.

    combined_score = (
        0.70 * pixel_score
        +
        0.30 * histogram_difference
    )


    timestamp_sec = (

        (frame_number - 1)
        /
        fps

        if fps > 0

        else 0.0
    )


    scores.append({

        "frame_number": frame_number,

        "timestamp_sec": timestamp_sec,

        "pixel_score": pixel_score,

        "histogram_difference": histogram_difference,

        "combined_score": combined_score
    })


    previous_gray = current_gray
    previous_hist = current_hist


    if frame_number % 1000 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames}"
        )


video.release()


# =========================================================
# RANK BY COMBINED SCORE
# =========================================================

ranked = sorted(
    scores,
    key=lambda item: item["combined_score"],
    reverse=True
)


# =========================================================
# NON-MAXIMUM SUPPRESSION
# =========================================================

selected = []


for candidate in ranked:

    candidate_frame = candidate[
        "frame_number"
    ]


    too_close = False


    for existing in selected:

        existing_frame = existing[
            "frame_number"
        ]


        if abs(
            candidate_frame
            -
            existing_frame
        ) < MIN_GAP_FRAMES:

            too_close = True
            break


    if too_close:
        continue


    selected.append(
        candidate
    )


    if len(selected) >= TOP_CANDIDATES:
        break


# =========================================================
# SORT CHRONOLOGICALLY
# =========================================================

selected = sorted(
    selected,
    key=lambda item: item["frame_number"]
)


# =========================================================
# REFERENCE DISTANCE
# =========================================================

def nearest_major_boundary(frame):

    nearest = min(
        KNOWN_MAJOR_BOUNDARIES,
        key=lambda boundary: abs(
            boundary - frame
        )
    )

    distance = abs(
        nearest - frame
    )

    return nearest, distance


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

        "pixel_difference",

        "histogram_difference",

        "combined_score",

        "nearest_major_scene_boundary",

        "distance_from_major_boundary"
    ])


    for index, candidate in enumerate(
        selected,
        start=1
    ):

        nearest_boundary, distance = (
            nearest_major_boundary(
                candidate["frame_number"]
            )
        )


        writer.writerow([

            f"BOUNDARY_CAND_{index:03d}",

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
                    "pixel_score"
                ],
                6
            ),

            round(
                candidate[
                    "histogram_difference"
                ],
                6
            ),

            round(
                candidate[
                    "combined_score"
                ],
                6
            ),

            nearest_boundary,

            distance
        ])


# =========================================================
# TERMINAL RESULTS
# =========================================================

print()
print("==========================================")
print("TOP SEQUENCE BOUNDARY CANDIDATES")
print("==========================================")

for index, candidate in enumerate(
    selected,
    start=1
):

    frame = candidate[
        "frame_number"
    ]

    time_sec = candidate[
        "timestamp_sec"
    ]

    pixel = candidate[
        "pixel_score"
    ]

    hist = candidate[
        "histogram_difference"
    ]

    combined = candidate[
        "combined_score"
    ]


    print(
        f"{index:02d}. "
        f"Frame {frame:4d} | "
        f"{time_sec:7.2f}s | "
        f"Pixel {pixel:.4f} | "
        f"Hist {hist:.4f} | "
        f"Combined {combined:.4f}"
    )


print()
print("------------------------------------------")
print("KNOWN MAJOR SCENE REFERENCES")
print("------------------------------------------")

for boundary in KNOWN_MAJOR_BOUNDARIES:

    timestamp = (

        (boundary - 1)
        /
        fps

        if fps > 0

        else 0.0
    )

    print(
        f"Frame {boundary} "
        f"~ {timestamp:.2f}s"
    )


print()
print("==========================================")
print("BOUNDARY ANALYSIS COMPLETE")
print("==========================================")
print(
    f"Candidates Saved : {len(selected)}"
)
print(
    f"Output CSV       : {OUTPUT_CSV}"
)
print("------------------------------------------")
print(
    "These are ranked visual-cut candidates."
)
print(
    "They are NOT automatically accepted as "
    "the original UMN sequence boundaries."
)
print("==========================================")