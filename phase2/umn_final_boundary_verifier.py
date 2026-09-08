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
    / "final_boundary_visuals"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# BOUNDARY CANDIDATES
# =========================================================
#
# Automatic internal candidates:
# 630, 2008, 2663, 3458, 3993, 4867, 6224, 6872
#
# Visually confirmed major scene changes:
# 1454, 5597
#
# IMPORTANT:
# These are candidate/reference frames.
# This script does NOT declare them official boundaries.
# =========================================================

CANDIDATES = [
    630,
    1454,
    2008,
    2663,
    3458,
    3993,
    4867,
    5597,
    6224,
    6872
]

FRAME_OFFSETS = [
    -2,
    -1,
    0,
    1,
    2
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
print("UMN FINAL BOUNDARY VERIFIER")
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

        position_text = "CANDIDATE"

    elif difference < 0:

        position_text = f"{difference}"

    else:

        position_text = f"+{difference}"


    timestamp = (
        (frame_number - 1)
        /
        fps
    )


    cv2.rectangle(
        image,
        (0, 0),
        (width, 42),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        image,
        f"Frame {frame_number}",
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        image,
        f"{position_text} | {timestamp:.2f}s",
        (5, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    return image


# =========================================================
# CREATE CONTACT SHEET
# =========================================================

def create_contact_sheet(
    candidate_frame
):

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
                f"Could not read frame "
                f"{frame_number}"
            )

            return None


        labelled = add_label(
            frame,
            frame_number,
            candidate_frame
        )


        # Scale image up slightly because
        # UMN source resolution is small.

        labelled = cv2.resize(
            labelled,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )


        images.append(
            labelled
        )


    contact_sheet = cv2.hconcat(
        images
    )

    return contact_sheet


# =========================================================
# GENERATE ALL VISUALS
# =========================================================

saved = 0


for index, candidate in enumerate(
    CANDIDATES,
    start=1
):

    contact_sheet = create_contact_sheet(
        candidate
    )


    if contact_sheet is None:
        continue


    output_file = (
        OUTPUT_DIR
        /
        f"{index:02d}_boundary_{candidate}.jpg"
    )


    success = cv2.imwrite(
        str(output_file),
        contact_sheet
    )


    if success:

        saved += 1

        print(
            f"Saved {saved:02d} -> "
            f"Boundary candidate {candidate}"
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
print("FINAL BOUNDARY VISUALS COMPLETE")
print("==========================================")

print(
    f"Images Saved : {saved}"
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
    "Frame -2 | Frame -1 | Candidate | "
    "Frame +1 | Frame +2"
)

print("------------------------------------------")

print(
    "No boundary was automatically accepted."
)

print(
    "No training labels were changed."
)

print("==========================================")