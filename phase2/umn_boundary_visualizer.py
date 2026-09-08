from pathlib import Path
import csv

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

CANDIDATE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_sequence_boundary_candidates.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "boundary_visuals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Candidate frame around which images are saved.
FRAME_OFFSET = 5

# We don't need all 40 initially.
# Strong/important candidates + known major boundaries.
IMPORTANT_FRAMES = [
    505,
    526,
    626,
    1331,
    1454,
    1807,
    2003,
    2606,
    3220,
    3456,
    3939,
    4035,
    4808,
    4930,
    5423,
    5597,
    6170,
    6255,
    6852,
    6932
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
print("UMN BOUNDARY VISUALIZER")
print("==========================================")
print(f"Video        : {VIDEO_PATH.name}")
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Frames       : {total_frames}")
print(f"Candidates   : {len(IMPORTANT_FRAMES)}")
print(f"Output       : {OUTPUT_DIR}")
print("------------------------------------------")


# =========================================================
# READ SPECIFIC FRAME
# =========================================================

def read_frame(frame_number):

    if frame_number < 1:
        return None

    if frame_number > total_frames:
        return None

    # OpenCV uses zero-based indexing internally.
    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    ret, frame = video.read()

    if not ret:
        return None

    return frame


# =========================================================
# ADD TEXT LABEL
# =========================================================

def add_label(
    frame,
    frame_number,
    label
):

    image = frame.copy()

    timestamp_sec = (
        (frame_number - 1) / fps
        if fps > 0
        else 0.0
    )

    text = (
        f"{label} | "
        f"Frame {frame_number} | "
        f"{timestamp_sec:.2f}s"
    )

    cv2.rectangle(
        image,
        (0, 0),
        (width, 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


# =========================================================
# CREATE SIDE-BY-SIDE PANEL
# =========================================================

def create_boundary_panel(
    candidate_frame
):

    before_frame_number = max(
        1,
        candidate_frame - FRAME_OFFSET
    )

    after_frame_number = min(
        total_frames,
        candidate_frame + FRAME_OFFSET
    )


    before = read_frame(
        before_frame_number
    )

    current = read_frame(
        candidate_frame
    )

    after = read_frame(
        after_frame_number
    )


    if (
        before is None
        or
        current is None
        or
        after is None
    ):

        print(
            f"Could not read candidate "
            f"frame {candidate_frame}"
        )

        return None


    before = add_label(
        before,
        before_frame_number,
        "BEFORE"
    )

    current = add_label(
        current,
        candidate_frame,
        "CANDIDATE"
    )

    after = add_label(
        after,
        after_frame_number,
        "AFTER"
    )


    panel = cv2.hconcat([
        before,
        current,
        after
    ])

    return panel


# =========================================================
# GENERATE IMAGES
# =========================================================

saved_count = 0


for index, candidate_frame in enumerate(
    IMPORTANT_FRAMES,
    start=1
):

    panel = create_boundary_panel(
        candidate_frame
    )

    if panel is None:
        continue


    filename = (
        f"{index:02d}_"
        f"frame_{candidate_frame}.jpg"
    )

    output_path = (
        OUTPUT_DIR
        /
        filename
    )


    cv2.imwrite(
        str(output_path),
        panel
    )


    saved_count += 1


    print(
        f"Saved {saved_count:02d} -> "
        f"Frame {candidate_frame}"
    )


video.release()


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==========================================")
print("BOUNDARY VISUALIZATION COMPLETE")
print("==========================================")
print(f"Images Saved : {saved_count}")
print(f"Folder       : {OUTPUT_DIR}")
print("------------------------------------------")
print(
    "Each image contains BEFORE | CANDIDATE | AFTER."
)
print(
    "Use these only to visually verify real video cuts."
)
print("==========================================")