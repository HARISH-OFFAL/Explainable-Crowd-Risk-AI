from pathlib import Path

import cv2


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

OUTPUT_DIR = (
    BASE_DIR
    / "refined_boundary_visuals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# REFINED BOUNDARY CANDIDATES
# =========================================================

CANDIDATES = [
    626,
    2003,
    2688,
    3456,
    4035,
    4930,
    6255,
    6932
]

# Compare wider context around each candidate.
FRAME_OFFSETS = [
    -15,
    -1,
    0,
    1,
    15
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


print()
print("==========================================")
print("UMN REFINED BOUNDARY VISUALIZER")
print("==========================================")
print(f"Video      : {VIDEO_PATH.name}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print(f"Candidates : {len(CANDIDATES)}")
print(f"Output     : {OUTPUT_DIR}")
print("------------------------------------------")


# =========================================================
# READ FRAME
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

    success, frame = video.read()

    if not success:
        return None

    return frame


# =========================================================
# ADD LABEL
# =========================================================

def add_label(
    frame,
    frame_number,
    candidate_frame
):

    image = frame.copy()

    height, width = image.shape[:2]

    difference = (
        frame_number
        -
        candidate_frame
    )


    if difference == 0:

        position = "CANDIDATE"

    elif difference > 0:

        position = f"+{difference}"

    else:

        position = str(difference)


    timestamp = (
        (frame_number - 1)
        /
        fps
    )


    cv2.rectangle(
        image,
        (0, 0),
        (width, 40),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        image,
        f"Frame {frame_number}",
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        image,
        f"{position} | {timestamp:.2f}s",
        (5, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    return image


# =========================================================
# CREATE VISUAL PANEL
# =========================================================

def create_panel(candidate_frame):

    images = []


    for offset in FRAME_OFFSETS:

        frame_number = (
            candidate_frame
            +
            offset
        )


        frame = read_frame(
            frame_number
        )


        if frame is None:

            print(
                f"ERROR reading frame "
                f"{frame_number}"
            )

            return None


        frame = add_label(
            frame,
            frame_number,
            candidate_frame
        )


        # Increase size because UMN is 320x240.

        frame = cv2.resize(
            frame,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )


        images.append(
            frame
        )


    return cv2.hconcat(
        images
    )


# =========================================================
# GENERATE
# =========================================================

saved_count = 0


for index, candidate in enumerate(
    CANDIDATES,
    start=1
):

    panel = create_panel(
        candidate
    )


    if panel is None:
        continue


    output_file = (
        OUTPUT_DIR
        /
        f"{index:02d}_refined_boundary_{candidate}.jpg"
    )


    success = cv2.imwrite(
        str(output_file),
        panel
    )


    if success:

        saved_count += 1

        print(
            f"Saved {saved_count:02d} -> "
            f"Refined candidate {candidate}"
        )


# =========================================================
# CLEANUP
# =========================================================

video.release()


# =========================================================
# SUMMARY
# =========================================================

print()
print("==========================================")
print("REFINED VISUALIZATION COMPLETE")
print("==========================================")

print(
    f"Images Saved : {saved_count}"
)

print(
    f"Expected     : {len(CANDIDATES)}"
)

print(
    f"Folder       : {OUTPUT_DIR}"
)

print("------------------------------------------")

print(
    "Each image contains:"
)

print(
    "-15 | -1 | Candidate | +1 | +15"
)

print("------------------------------------------")

print(
    "These visuals are for sequence-boundary "
    "verification only."
)

print(
    "No automatic boundary acceptance occurred."
)

print(
    "Training labels were not modified."
)

print("==========================================")