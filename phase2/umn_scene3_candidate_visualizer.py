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

OUTPUT_DIR = (
    BASE_DIR
    / "scene3_candidate_visuals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CANDIDATE_FRAMES = [
    6797,
    7097,
    7367,
    7607
]

# UMN FPS is 30.
# 60 frames = approximately 2 seconds.
FRAME_OFFSET = 60


# =========================================================
# OPEN VIDEO
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
print("UMN SCENE 3 CANDIDATE VISUALIZER")
print("==========================================")
print(f"Video      : {VIDEO_PATH.name}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print(f"Candidates : {len(CANDIDATE_FRAMES)}")
print(f"Offset     : {FRAME_OFFSET} frames")
print(f"Output     : {OUTPUT_DIR}")
print("------------------------------------------")


# =========================================================
# READ SPECIFIC FRAME
# =========================================================

def read_frame(frame_number):

    frame_number = max(
        1,
        min(
            total_frames,
            frame_number
        )
    )

    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    ret, frame = video.read()

    if not ret:
        return None

    return frame


# =========================================================
# ADD LABEL
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
        (width, 32),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


# =========================================================
# CREATE PANEL
# =========================================================

def create_panel(candidate_frame):

    before_frame = max(
        1,
        candidate_frame - FRAME_OFFSET
    )

    after_frame = min(
        total_frames,
        candidate_frame + FRAME_OFFSET
    )

    before = read_frame(
        before_frame
    )

    current = read_frame(
        candidate_frame
    )

    after = read_frame(
        after_frame
    )


    if (
        before is None
        or
        current is None
        or
        after is None
    ):

        return None


    before = add_label(
        before,
        before_frame,
        "2 SEC BEFORE"
    )

    current = add_label(
        current,
        candidate_frame,
        "CANDIDATE"
    )

    after = add_label(
        after,
        after_frame,
        "2 SEC AFTER"
    )


    panel = cv2.hconcat([
        before,
        current,
        after
    ])

    return panel


# =========================================================
# GENERATE VISUALS
# =========================================================

saved_count = 0


for index, candidate_frame in enumerate(
    CANDIDATE_FRAMES,
    start=1
):

    panel = create_panel(
        candidate_frame
    )


    if panel is None:

        print(
            f"Could not create visual for "
            f"frame {candidate_frame}"
        )

        continue


    output_file = (
        OUTPUT_DIR
        /
        f"{index:02d}_candidate_{candidate_frame}.jpg"
    )


    success = cv2.imwrite(
        str(output_file),
        panel
    )


    if success:

        saved_count += 1

        print(
            f"Saved {saved_count:02d} -> "
            f"Frame {candidate_frame}"
        )


video.release()


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("SCENE 3 VISUALIZATION COMPLETE")
print("==========================================")
print(f"Images Saved : {saved_count}")
print(f"Folder       : {OUTPUT_DIR}")
print("------------------------------------------")
print(
    "Each image shows:"
)
print(
    "2 SEC BEFORE | CANDIDATE | 2 SEC AFTER"
)
print("==========================================")