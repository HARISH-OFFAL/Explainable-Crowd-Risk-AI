from pathlib import Path
import csv


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_movement_features.csv"
)

LABEL_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_clean_labels.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "datasets"
    / "umn_labeled_movement_features.csv"
)


# =========================================================
# SEQUENCE-WISE SPLIT
# =========================================================
#
# IMPORTANT:
# Entire sequences stay inside only ONE split.
#
# This prevents adjacent frames from the same sequence
# appearing in both training and evaluation data.
#
# =========================================================

TRAIN_SEQUENCES = {
    "SEQ_01",
    "SEQ_03",
    "SEQ_04",
    "SEQ_05",
    "SEQ_07",
    "SEQ_09",
    "SEQ_10",
}

VALIDATION_SEQUENCES = {
    "SEQ_02",
    "SEQ_06",
}

TEST_SEQUENCES = {
    "SEQ_08",
    "SEQ_11",
}


# =========================================================
# HELPERS
# =========================================================

def safe_int(value, default=0):

    try:
        return int(float(value))

    except (ValueError, TypeError):
        return default


def get_split(sequence_id):

    if sequence_id in TRAIN_SEQUENCES:
        return "TRAIN"

    if sequence_id in VALIDATION_SEQUENCES:
        return "VALIDATION"

    if sequence_id in TEST_SEQUENCES:
        return "TEST"

    raise ValueError(
        f"No training split assigned for {sequence_id}"
    )


# =========================================================
# VALIDATE SPLIT CONFIGURATION
# =========================================================

all_split_sequences = (
    TRAIN_SEQUENCES
    |
    VALIDATION_SEQUENCES
    |
    TEST_SEQUENCES
)


if len(all_split_sequences) != 11:

    raise ValueError(
        "Expected exactly 11 unique sequences "
        "across TRAIN / VALIDATION / TEST."
    )


if (
    TRAIN_SEQUENCES
    &
    VALIDATION_SEQUENCES
):

    raise ValueError(
        "TRAIN and VALIDATION overlap."
    )


if (
    TRAIN_SEQUENCES
    &
    TEST_SEQUENCES
):

    raise ValueError(
        "TRAIN and TEST overlap."
    )


if (
    VALIDATION_SEQUENCES
    &
    TEST_SEQUENCES
):

    raise ValueError(
        "VALIDATION and TEST overlap."
    )


# =========================================================
# LOAD LABEL RANGES
# =========================================================

label_ranges = []


with open(
    LABEL_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        sequence_id = row[
            "sequence_id"
        ]

        label_ranges.append(
            {
                "sequence_id":
                    sequence_id,

                "scene_id":
                    row["scene_id"],

                "scene_name":
                    row["scene_name"],

                "label":
                    row["label"],

                "label_value":
                    safe_int(
                        row["label_value"]
                    ),

                "start_frame":
                    safe_int(
                        row["start_frame"]
                    ),

                "end_frame":
                    safe_int(
                        row["end_frame"]
                    ),

                "transition_onset_frame":
                    safe_int(
                        row[
                            "transition_onset_frame"
                        ]
                    ),

                "training_split":
                    get_split(
                        sequence_id
                    ),
            }
        )


# =========================================================
# FIND LABEL
# =========================================================

def find_label(frame_number):

    for item in label_ranges:

        if (
            item["start_frame"]
            <= frame_number
            <= item["end_frame"]
        ):
            return item

    return None


# =========================================================
# COUNTERS
# =========================================================

stats = {
    "TRAIN": {
        "NORMAL": 0,
        "ABNORMAL": 0,
        "TOTAL": 0,
    },

    "VALIDATION": {
        "NORMAL": 0,
        "ABNORMAL": 0,
        "TOTAL": 0,
    },

    "TEST": {
        "NORMAL": 0,
        "ABNORMAL": 0,
        "TOTAL": 0,
    },
}


rows_read = 0
rows_written = 0
unmatched_rows = 0


# =========================================================
# START
# =========================================================

print()
print("==========================================")
print("UMN SEQUENCE-WISE DATASET BUILDER")
print("==========================================")

print(
    "TRAIN      : "
    +
    ", ".join(
        sorted(TRAIN_SEQUENCES)
    )
)

print(
    "VALIDATION : "
    +
    ", ".join(
        sorted(VALIDATION_SEQUENCES)
    )
)

print(
    "TEST       : "
    +
    ", ".join(
        sorted(TEST_SEQUENCES)
    )
)

print("------------------------------------------")


# =========================================================
# READ FEATURES + WRITE DATASET
# =========================================================

with open(
    FEATURE_CSV,
    "r",
    newline="",
    encoding="utf-8"
) as input_file:

    reader = csv.DictReader(
        input_file
    )

    original_fields = (
        reader.fieldnames
        or []
    )


    extra_fields = [
        "sequence_id",
        "scene_id",
        "scene_name",
        "behavior_label",
        "behavior_label_value",
        "transition_onset_frame",
        "label_source",
        "training_split",
    ]


    fieldnames = (
        original_fields
        +
        extra_fields
    )


    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for row in reader:

            rows_read += 1


            frame_number = safe_int(
                row[
                    "frame_number"
                ]
            )


            label_info = find_label(
                frame_number
            )


            if label_info is None:

                unmatched_rows += 1
                continue


            sequence_id = label_info[
                "sequence_id"
            ]

            label = label_info[
                "label"
            ]

            split = label_info[
                "training_split"
            ]


            # ---------------------------------------------
            # ADD METADATA
            # ---------------------------------------------

            row[
                "sequence_id"
            ] = sequence_id

            row[
                "scene_id"
            ] = label_info[
                "scene_id"
            ]

            row[
                "scene_name"
            ] = label_info[
                "scene_name"
            ]

            row[
                "behavior_label"
            ] = label

            row[
                "behavior_label_value"
            ] = label_info[
                "label_value"
            ]

            row[
                "transition_onset_frame"
            ] = label_info[
                "transition_onset_frame"
            ]

            row[
                "label_source"
            ] = (
                "LOCAL_VIDEO_"
                "EMBEDDED_RED_TEXT_ONSET"
            )

            row[
                "training_split"
            ] = split


            writer.writerow(
                row
            )


            rows_written += 1


            # ---------------------------------------------
            # STATISTICS
            # ---------------------------------------------

            stats[
                split
            ][
                label
            ] += 1

            stats[
                split
            ][
                "TOTAL"
            ] += 1


# =========================================================
# VALIDATION
# =========================================================

if unmatched_rows > 0:

    raise ValueError(
        f"{unmatched_rows} feature rows "
        f"could not be matched."
    )


if rows_read != rows_written:

    raise ValueError(
        f"Row mismatch: "
        f"read {rows_read}, "
        f"wrote {rows_written}"
    )


# =========================================================
# DISPLAY STATISTICS
# =========================================================

print(
    f"Rows Read      : "
    f"{rows_read}"
)

print(
    f"Rows Written   : "
    f"{rows_written}"
)

print(
    f"Unmatched Rows : "
    f"{unmatched_rows}"
)

print("------------------------------------------")


for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    print(split)

    print(
        f"  NORMAL   : "
        f"{stats[split]['NORMAL']}"
    )

    print(
        f"  ABNORMAL : "
        f"{stats[split]['ABNORMAL']}"
    )

    print(
        f"  TOTAL    : "
        f"{stats[split]['TOTAL']}"
    )

    print()


# =========================================================
# FINAL VALIDATION
# =========================================================

for split in [
    "TRAIN",
    "VALIDATION",
    "TEST",
]:

    if stats[split]["NORMAL"] == 0:

        raise ValueError(
            f"{split} contains no NORMAL rows."
        )

    if stats[split]["ABNORMAL"] == 0:

        raise ValueError(
            f"{split} contains no ABNORMAL rows."
        )


total_split_rows = sum(
    stats[split]["TOTAL"]
    for split in stats
)


if total_split_rows != rows_written:

    raise ValueError(
        "Split row total does not "
        "match output row total."
    )


# =========================================================
# SUMMARY
# =========================================================

print("------------------------------------------")

print(
    f"Total Split Rows : "
    f"{total_split_rows}"
)

print(
    f"Output           : "
    f"{OUTPUT_CSV}"
)

print("------------------------------------------")

print(
    "Sequence-wise split completed."
)

print(
    "No sequence appears in more "
    "than one split."
)

print(
    "Random row-level splitting was NOT used."
)

print(
    "Dataset is prepared for the next "
    "model-development stage."
)

print("==========================================")