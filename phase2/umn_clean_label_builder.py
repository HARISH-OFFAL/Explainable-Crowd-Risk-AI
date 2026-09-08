from pathlib import Path
import csv


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

SEQUENCE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_verified_sequences.csv"
)

RED_SCAN_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_red_annotation_scan.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_clean_labels.csv"
)

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# HELPERS
# =========================================================

def safe_int(value):
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return int(float(value))


# =========================================================
# LOAD SEQUENCES
# =========================================================

sequences = {}

with open(
    SEQUENCE_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        sequences[
            row["sequence_id"]
        ] = {
            "sequence_id":
                row["sequence_id"],

            "scene_id":
                row["scene_id"],

            "scene_name":
                row["scene_name"],

            "start_frame":
                int(row["start_frame"]),

            "end_frame":
                int(row["end_frame"]),

            "video_name":
                row["video_name"],
        }


# =========================================================
# LOAD VERIFIED RED-TEXT ONSETS
# =========================================================

onsets = {}

with open(
    RED_SCAN_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        sequence_id = row["sequence_id"]

        onset_frame = safe_int(
            row[
                "first_stable_red_text_frame"
            ]
        )

        onsets[
            sequence_id
        ] = {
            "onset_frame":
                onset_frame,

            "scan_status":
                row["scan_status"],

            "interpretation":
                row["interpretation"],

            "official_ground_truth_claim":
                row[
                    "official_ground_truth_claim"
                ],
        }


# =========================================================
# BUILD CLEAN LABEL RANGES
# =========================================================

label_rows = []


print()
print("==========================================")
print("UMN CLEAN LABEL BUILDER")
print("==========================================")
print(f"Sequences : {len(sequences)}")
print("------------------------------------------")


for sequence_id in sorted(
    sequences.keys()
):

    sequence = sequences[
        sequence_id
    ]

    if sequence_id not in onsets:

        raise ValueError(
            f"No transition onset found for "
            f"{sequence_id}"
        )


    onset = onsets[
        sequence_id
    ][
        "onset_frame"
    ]


    if onset is None:

        raise ValueError(
            f"{sequence_id} has no stable "
            f"abnormal-text onset."
        )


    start_frame = sequence[
        "start_frame"
    ]

    end_frame = sequence[
        "end_frame"
    ]


    if not (
        start_frame
        <= onset
        <= end_frame
    ):

        raise ValueError(
            f"{sequence_id}: onset frame "
            f"{onset} outside sequence "
            f"{start_frame}-{end_frame}"
        )


    normal_start = start_frame

    normal_end = onset - 1

    abnormal_start = onset

    abnormal_end = end_frame


    # =====================================================
    # NORMAL SEGMENT
    # =====================================================

    if normal_end >= normal_start:

        normal_count = (
            normal_end
            -
            normal_start
            +
            1
        )

        label_rows.append(
            {
                "video_name":
                    sequence["video_name"],

                "sequence_id":
                    sequence_id,

                "scene_id":
                    sequence["scene_id"],

                "scene_name":
                    sequence["scene_name"],

                "label":
                    "NORMAL",

                "label_value":
                    0,

                "start_frame":
                    normal_start,

                "end_frame":
                    normal_end,

                "frame_count":
                    normal_count,

                "transition_onset_frame":
                    onset,

                "label_source":
                    "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

                "ground_truth_status":
                    "RECONSTRUCTED_FROM_EMBEDDED_ANNOTATION",

                "official_umn_frame_claim":
                    "NO",

                "training_allowed":
                    "NO",
            }
        )


    # =====================================================
    # ABNORMAL SEGMENT
    # =====================================================

    abnormal_count = (
        abnormal_end
        -
        abnormal_start
        +
        1
    )

    label_rows.append(
        {
            "video_name":
                sequence["video_name"],

            "sequence_id":
                sequence_id,

            "scene_id":
                sequence["scene_id"],

            "scene_name":
                sequence["scene_name"],

            "label":
                "ABNORMAL",

            "label_value":
                1,

            "start_frame":
                abnormal_start,

            "end_frame":
                abnormal_end,

            "frame_count":
                abnormal_count,

            "transition_onset_frame":
                onset,

            "label_source":
                "LOCAL_VIDEO_EMBEDDED_RED_TEXT_ONSET",

            "ground_truth_status":
                "RECONSTRUCTED_FROM_EMBEDDED_ANNOTATION",

            "official_umn_frame_claim":
                "NO",

            "training_allowed":
                "NO",
        }
    )


    print(
        f"{sequence_id} | "
        f"NORMAL "
        f"{normal_start}-{normal_end} | "
        f"ABNORMAL "
        f"{abnormal_start}-{abnormal_end}"
    )


# =========================================================
# VALIDATE COMPLETE COVERAGE
# =========================================================

expected_frames = sum(
    (
        sequence[
            "end_frame"
        ]
        -
        sequence[
            "start_frame"
        ]
        +
        1
    )
    for sequence in sequences.values()
)


labelled_frames = sum(
    row["frame_count"]
    for row in label_rows
)


if expected_frames != labelled_frames:

    raise ValueError(
        f"Frame coverage mismatch: "
        f"expected {expected_frames}, "
        f"labelled {labelled_frames}"
    )


normal_frames = sum(
    row["frame_count"]
    for row in label_rows
    if row["label"] == "NORMAL"
)

abnormal_frames = sum(
    row["frame_count"]
    for row in label_rows
    if row["label"] == "ABNORMAL"
)


# =========================================================
# SAVE CSV
# =========================================================

fieldnames = [
    "video_name",
    "sequence_id",
    "scene_id",
    "scene_name",
    "label",
    "label_value",
    "start_frame",
    "end_frame",
    "frame_count",
    "transition_onset_frame",
    "label_source",
    "ground_truth_status",
    "official_umn_frame_claim",
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
        label_rows
    )


# =========================================================
# SUMMARY
# =========================================================

print("------------------------------------------")
print(f"Label Rows      : {len(label_rows)}")
print(f"Covered Frames  : {labelled_frames}")
print(f"Expected Frames : {expected_frames}")
print(f"NORMAL Frames   : {normal_frames}")
print(f"ABNORMAL Frames : {abnormal_frames}")
print("------------------------------------------")
print(f"Output : {OUTPUT_CSV}")
print("------------------------------------------")
print(
    "Labels are reconstructed from the "
    "embedded red annotation in the local UMN video."
)
print(
    "They are not claimed as official "
    "UMN per-frame ground truth."
)
print(
    "Training allowed: NO"
)
print("==========================================")