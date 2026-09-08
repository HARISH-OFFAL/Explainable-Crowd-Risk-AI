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

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_refined_cut_candidates.csv"
)


# =========================================================
# APPROXIMATE REGIONS
# =========================================================

APPROXIMATE_CANDIDATES = [
    630,
    2008,
    2663,
    3458,
    3993,
    4867,
    6224,
    6872
]

SEARCH_RADIUS = 150

# Keep several alternatives from each region.
TOP_PER_REGION = 5

# Prevent almost-identical neighbouring frames
# appearing repeatedly in top results.
MIN_LOCAL_GAP = 5


# =========================================================
# CHECK VIDEO
# =========================================================

if not VIDEO_PATH.exists():
    print()
    print("ERROR: UMN video not found.")
    print(VIDEO_PATH)
    raise SystemExit


video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    print()
    print("ERROR: Could not open UMN video.")
    raise SystemExit


fps = float(video.get(cv2.CAP_PROP_FPS))

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)


# =========================================================
# READ FRAME
# =========================================================

def read_frame(frame_number):

    frame_number = max(
        1,
        min(total_frames, frame_number)
    )

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    success, frame = video.read()

    if not success:
        return None

    return frame


# =========================================================
# PREPROCESS
# =========================================================

def preprocess(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Small resolution is enough for cut detection.
    gray = cv2.resize(
        gray,
        (160, 120),
        interpolation=cv2.INTER_AREA
    )

    return gray


# =========================================================
# PIXEL DIFFERENCE
# =========================================================

def pixel_difference(frame_a, frame_b):

    difference = cv2.absdiff(
        frame_a,
        frame_b
    )

    return float(
        np.mean(difference)
    ) / 255.0


# =========================================================
# HISTOGRAM DIFFERENCE
# =========================================================

def histogram_difference(frame_a, frame_b):

    hist_a = cv2.calcHist(
        [frame_a],
        [0],
        None,
        [64],
        [0, 256]
    )

    hist_b = cv2.calcHist(
        [frame_b],
        [0],
        None,
        [64],
        [0, 256]
    )

    cv2.normalize(
        hist_a,
        hist_a
    )

    cv2.normalize(
        hist_b,
        hist_b
    )

    correlation = cv2.compareHist(
        hist_a,
        hist_b,
        cv2.HISTCMP_CORREL
    )

    # High value = stronger histogram change.
    return 1.0 - float(correlation)


# =========================================================
# SCAN ONE LOCAL REGION
# =========================================================

def scan_region(approximate_frame):

    start_frame = max(
        2,
        approximate_frame - SEARCH_RADIUS
    )

    end_frame = min(
        total_frames,
        approximate_frame + SEARCH_RADIUS
    )

    results = []

    previous_raw = read_frame(
        start_frame - 1
    )

    if previous_raw is None:
        return []

    previous = preprocess(
        previous_raw
    )


    for frame_number in range(
        start_frame,
        end_frame + 1
    ):

        current_raw = read_frame(
            frame_number
        )

        if current_raw is None:
            continue

        current = preprocess(
            current_raw
        )


        pixel_score = pixel_difference(
            previous,
            current
        )

        histogram_score = histogram_difference(
            previous,
            current
        )


        # This is only a cut-ranking score.
        # It is NOT a risk or anomaly score.

        combined_score = (
            0.65 * pixel_score
            +
            0.35 * histogram_score
        )


        results.append(
            {
                "approximate_frame":
                    approximate_frame,

                "cut_frame":
                    frame_number,

                "previous_frame":
                    frame_number - 1,

                "pixel_difference":
                    pixel_score,

                "histogram_difference":
                    histogram_score,

                "combined_score":
                    combined_score,

                "distance_from_approx":
                    abs(
                        frame_number
                        -
                        approximate_frame
                    )
            }
        )


        previous = current


    return results


# =========================================================
# SELECT TOP DISTINCT LOCAL CANDIDATES
# =========================================================

def select_top(results):

    ranked = sorted(
        results,
        key=lambda row:
            row["combined_score"],
        reverse=True
    )

    selected = []


    for candidate in ranked:

        frame = candidate[
            "cut_frame"
        ]

        too_close = any(
            abs(
                frame
                -
                existing["cut_frame"]
            )
            < MIN_LOCAL_GAP
            for existing in selected
        )

        if too_close:
            continue


        selected.append(
            candidate
        )


        if len(selected) >= TOP_PER_REGION:
            break


    return selected


# =========================================================
# RUN
# =========================================================

print()
print("==========================================")
print("UMN LOCAL CUT REFINER")
print("==========================================")
print(f"Video         : {VIDEO_PATH.name}")
print(f"FPS           : {fps:.2f}")
print(f"Frames        : {total_frames}")
print(f"Regions       : {len(APPROXIMATE_CANDIDATES)}")
print(f"Search Radius : +/- {SEARCH_RADIUS} frames")
print("------------------------------------------")


all_results = []


for approximate_frame in APPROXIMATE_CANDIDATES:

    local_results = scan_region(
        approximate_frame
    )

    selected = select_top(
        local_results
    )

    print()
    print(
        f"Approximate region: "
        f"{approximate_frame}"
    )

    print(
        "Top local consecutive-frame changes:"
    )


    for rank, candidate in enumerate(
        selected,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{candidate['previous_frame']} -> "
            f"{candidate['cut_frame']} | "
            f"Pixel "
            f"{candidate['pixel_difference']:.4f} | "
            f"Hist "
            f"{candidate['histogram_difference']:.4f} | "
            f"Combined "
            f"{candidate['combined_score']:.4f}"
        )


        candidate[
            "local_rank"
        ] = rank

        all_results.append(
            candidate
        )


# =========================================================
# SAVE CSV
# =========================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


fieldnames = [
    "approximate_frame",
    "local_rank",
    "previous_frame",
    "cut_frame",
    "timestamp_sec",
    "pixel_difference",
    "histogram_difference",
    "combined_score",
    "distance_from_approx",
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


    for row in all_results:

        writer.writerow(
            {
                "approximate_frame":
                    row["approximate_frame"],

                "local_rank":
                    row["local_rank"],

                "previous_frame":
                    row["previous_frame"],

                "cut_frame":
                    row["cut_frame"],

                "timestamp_sec":
                    round(
                        (
                            row["cut_frame"] - 1
                        )
                        / fps,
                        4
                    ),

                "pixel_difference":
                    round(
                        row["pixel_difference"],
                        6
                    ),

                "histogram_difference":
                    round(
                        row["histogram_difference"],
                        6
                    ),

                "combined_score":
                    round(
                        row["combined_score"],
                        6
                    ),

                "distance_from_approx":
                    row["distance_from_approx"],

                "status":
                    "LOCAL_CUT_CANDIDATE",

                "training_allowed":
                    "NO"
            }
        )


video.release()


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("LOCAL CUT REFINEMENT COMPLETE")
print("==========================================")

print(
    f"Regions analyzed : "
    f"{len(APPROXIMATE_CANDIDATES)}"
)

print(
    f"Candidates saved : "
    f"{len(all_results)}"
)

print(
    f"Output           : "
    f"{OUTPUT_CSV}"
)

print("------------------------------------------")

print(
    "These are local cut candidates only."
)

print(
    "No sequence boundary was automatically accepted."
)

print(
    "Training allowed: NO"
)

print("==========================================")