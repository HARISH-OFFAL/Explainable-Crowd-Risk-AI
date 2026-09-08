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

CANDIDATE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_sequence_abnormal_candidates.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "umn_transition_panels"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIG
# =========================================================

FRAME_OFFSETS = [-30, -15, 0, 15, 30]

PANEL_WIDTH = 320
PANEL_HEIGHT = 240

TEXT_HEIGHT = 55


# =========================================================
# CHECK FILES
# =========================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not CANDIDATE_CSV.exists():
    raise FileNotFoundError(
        f"Candidate CSV not found: {CANDIDATE_CSV}"
    )


# =========================================================
# LOAD RANK-1 CANDIDATES
# =========================================================

candidates = []


with open(
    CANDIDATE_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        rank = int(
            float(
                row["candidate_rank"]
            )
        )

        if rank != 1:
            continue

        candidates.append(
            {
                "sequence_id":
                    row["sequence_id"],

                "scene_name":
                    row["scene_name"],

                "sequence_start_frame":
                    int(
                        float(
                            row[
                                "sequence_start_frame"
                            ]
                        )
                    ),

                "sequence_end_frame":
                    int(
                        float(
                            row[
                                "sequence_end_frame"
                            ]
                        )
                    ),

                "candidate_frame":
                    int(
                        float(
                            row[
                                "candidate_start_frame"
                            ]
                        )
                    ),

                "transition_score":
                    float(
                        row[
                            "transition_score"
                        ]
                    )
            }
        )


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


print()
print("==========================================")
print("UMN TRANSITION VISUALIZER")
print("==========================================")
print(f"Video       : {VIDEO_PATH}")
print(f"FPS         : {fps}")
print(f"Total Frames: {total_frames}")
print(f"Candidates  : {len(candidates)}")
print("------------------------------------------")


# =========================================================
# READ FRAME FUNCTION
# =========================================================

def read_frame(frame_number):

    # OpenCV uses zero-based indexing.
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    success, frame = cap.read()

    if not success:
        return None

    return frame


# =========================================================
# CREATE PANEL
# =========================================================

for candidate in candidates:

    sequence_id = candidate[
        "sequence_id"
    ]

    scene_name = candidate[
        "scene_name"
    ]

    sequence_start = candidate[
        "sequence_start_frame"
    ]

    sequence_end = candidate[
        "sequence_end_frame"
    ]

    candidate_frame = candidate[
        "candidate_frame"
    ]

    transition_score = candidate[
        "transition_score"
    ]


    panel_images = []


    for offset in FRAME_OFFSETS:

        target_frame = (
            candidate_frame
            +
            offset
        )


        # Keep target inside this sequence.
        target_frame = max(
            sequence_start,
            target_frame
        )

        target_frame = min(
            sequence_end,
            target_frame
        )


        frame = read_frame(
            target_frame
        )


        if frame is None:

            frame = np.zeros(
                (
                    PANEL_HEIGHT,
                    PANEL_WIDTH,
                    3
                ),
                dtype=np.uint8
            )


        frame = cv2.resize(
            frame,
            (
                PANEL_WIDTH,
                PANEL_HEIGHT
            )
        )


        # Add white information area.
        canvas = np.full(
            (
                PANEL_HEIGHT + TEXT_HEIGHT,
                PANEL_WIDTH,
                3
            ),
            255,
            dtype=np.uint8
        )


        canvas[
            :PANEL_HEIGHT,
            :
        ] = frame


        if offset == 0:

            label = (
                f"CANDIDATE | Frame {target_frame}"
            )

        elif offset > 0:

            label = (
                f"+{offset} | Frame {target_frame}"
            )

        else:

            label = (
                f"{offset} | Frame {target_frame}"
            )


        cv2.putText(
            canvas,
            label,
            (
                8,
                PANEL_HEIGHT + 23
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


        timestamp = (
            target_frame - 1
        ) / fps


        cv2.putText(
            canvas,
            f"Time: {timestamp:.2f}s",
            (
                8,
                PANEL_HEIGHT + 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )


        panel_images.append(
            canvas
        )


    # =====================================================
    # JOIN PANELS
    # =====================================================

    combined = cv2.hconcat(
        panel_images
    )


    # Add title area.
    title_height = 70

    final_image = np.full(
        (
            combined.shape[0]
            +
            title_height,
            combined.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )


    final_image[
        title_height:,
        :
    ] = combined


    title = (
        f"{sequence_id} | "
        f"{scene_name} | "
        f"Candidate Frame {candidate_frame}"
    )


    cv2.putText(
        final_image,
        title,
        (
            15,
            28
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        final_image,
        (
            f"Transition Score: "
            f"{transition_score:.4f}"
        ),
        (
            15,
            55
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


    output_path = (
        OUTPUT_DIR
        /
        f"{sequence_id}_transition_{candidate_frame}.jpg"
    )


    cv2.imwrite(
        str(output_path),
        final_image
    )


    print(
        f"{sequence_id} -> "
        f"Frame {candidate_frame} -> "
        f"{output_path.name}"
    )


# =========================================================
# CLEANUP
# =========================================================

cap.release()


# =========================================================
# SUMMARY
# =========================================================

print("------------------------------------------")

print(
    f"Panels Created : {len(candidates)}"
)

print(
    f"Output Folder  : {OUTPUT_DIR}"
)

print("------------------------------------------")

print(
    "These panels are for visual verification "
    "of automatic transition candidates."
)

print(
    "No ground-truth labels were created."
)

print(
    "Training allowed: NO"
)

print("==========================================")