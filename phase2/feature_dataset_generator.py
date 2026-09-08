from ultralytics import YOLO
import cv2
import torch
import math
import csv
import json
import hashlib
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEOS_FOLDER = BASE_DIR / "videos"
OUTPUT_FOLDER = BASE_DIR / "datasets"

OUTPUT_CSV = OUTPUT_FOLDER / "movement_features.csv"
PROCESSED_FILE = OUTPUT_FOLDER / "processed_videos.json"

MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.15
INFERENCE_SIZE = 960

SUPPORTED_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v"
)


# =========================================================
# CSV COLUMNS
# =========================================================

CSV_HEADER = [

    "video_name",
    "video_id",

    "frame_number",
    "timestamp_sec",

    "track_id",

    "frame_width",
    "frame_height",
    "video_fps",

    "center_x",
    "center_y",

    "normalized_x",
    "normalized_y",

    "bbox_width",
    "bbox_height",
    "bbox_area_ratio",

    "detection_confidence",

    "motion_valid",

    "movement_distance_px",
    "movement_distance_norm",

    "speed_px_s",
    "speed_norm_s",

    "acceleration_px_s2",
    "acceleration_norm_s2",

    "direction_angle_deg",
    "direction_change_deg",

    "direction_sin",
    "direction_cos",

    "consecutive_frames"
]


# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():

    DEVICE = 0

    print(
        "Using GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = "cpu"

    print("Using CPU")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


def calculate_direction_angle(dx, dy):

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    if angle < 0:
        angle += 360

    return angle


def calculate_direction_change(
    previous_angle,
    current_angle
):

    difference = abs(
        current_angle
        -
        previous_angle
    )

    if difference > 180:

        difference = (
            360
            -
            difference
        )

    return difference


# =========================================================
# VIDEO HASH
# =========================================================

def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# =========================================================
# LOAD PROCESSED VIDEO REGISTRY
# =========================================================

def load_processed_registry():

    if not PROCESSED_FILE.exists():

        return {}


    try:

        with open(
            PROCESSED_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        if isinstance(data, dict):

            return data


    except (
        json.JSONDecodeError,
        OSError
    ):

        print()
        print(
            "WARNING: processed_videos.json could not be read."
        )

        print(
            "Starting with an empty processing registry."
        )


    return {}


# =========================================================
# SAVE PROCESSED VIDEO REGISTRY
# =========================================================

def save_processed_registry(registry):

    temp_file = PROCESSED_FILE.with_suffix(
        ".tmp"
    )


    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


    temp_file.replace(
        PROCESSED_FILE
    )


# =========================================================
# FIND ALL VIDEOS
# =========================================================

if not VIDEOS_FOLDER.exists():

    print(
        f"ERROR: Folder not found: {VIDEOS_FOLDER}"
    )

    raise SystemExit


video_files = sorted([

    file_path

    for file_path in VIDEOS_FOLDER.iterdir()

    if (
        file_path.is_file()
        and
        file_path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )
])


if not video_files:

    print(
        f"ERROR: No supported videos found in {VIDEOS_FOLDER}"
    )

    raise SystemExit


# =========================================================
# LOAD REGISTRY
# =========================================================

processed_registry = (
    load_processed_registry()
)


# =========================================================
# CHECK EXISTING CSV
# =========================================================

csv_exists = (
    OUTPUT_CSV.exists()
    and
    OUTPUT_CSV.stat().st_size > 0
)


# =========================================================
# IMPORTANT FIRST-TIME MIGRATION
# =========================================================
#
# We already generated movement_features.csv before adding
# incremental processing.
#
# Therefore, if:
#
#   movement_features.csv exists
#   BUT
#   processed_videos.json does not yet contain records
#
# we read video names already present in the CSV and register
# their current hashes.
#
# This prevents all old videos from being processed again
# unnecessarily on the first incremental run.
# =========================================================

if csv_exists and not processed_registry:

    print()
    print("==========================================")
    print("INITIALIZING INCREMENTAL VIDEO REGISTRY")
    print("==========================================")


    existing_video_names = set()


    try:

        with open(
            OUTPUT_CSV,
            "r",
            newline="",
            encoding="utf-8"
        ) as existing_csv:

            reader = csv.DictReader(
                existing_csv
            )


            for row in reader:

                video_name = row.get(
                    "video_name"
                )

                if video_name:

                    existing_video_names.add(
                        video_name
                    )


    except OSError:

        existing_video_names = set()


    for video_path in video_files:

        if (
            video_path.name
            in existing_video_names
        ):

            print(
                f"Registering existing video: {video_path.name}"
            )


            video_hash = (
                calculate_file_hash(
                    video_path
                )
            )


            processed_registry[
                video_path.name
            ] = {

                "hash":
                    video_hash,

                "video_id":
                    None,

                "status":
                    "completed"
            }


    if processed_registry:

        save_processed_registry(
            processed_registry
        )


    print(
        "Existing dataset registered successfully."
    )

    print("==========================================")


# =========================================================
# DETERMINE NEXT VIDEO ID
# =========================================================

existing_video_ids = []


if csv_exists:

    try:

        with open(
            OUTPUT_CSV,
            "r",
            newline="",
            encoding="utf-8"
        ) as existing_csv:

            reader = csv.DictReader(
                existing_csv
            )


            for row in reader:

                video_id = row.get(
                    "video_id"
                )


                if (
                    video_id
                    and
                    video_id.startswith(
                        "VIDEO_"
                    )
                ):

                    try:

                        numeric_id = int(
                            video_id.split(
                                "_"
                            )[1]
                        )

                        existing_video_ids.append(
                            numeric_id
                        )

                    except (
                        ValueError,
                        IndexError
                    ):

                        pass


    except OSError:

        pass


if existing_video_ids:

    next_video_number = (
        max(existing_video_ids)
        +
        1
    )

else:

    next_video_number = 1


# =========================================================
# CHECK VIDEO STATUS
# =========================================================

videos_to_process = []

skipped_videos = []


print()
print("==========================================")
print("CHECKING VIDEO DATASET")
print("==========================================")


for video_path in video_files:

    video_name = (
        video_path.name
    )


    print(
        f"Checking: {video_name}"
    )


    current_hash = (
        calculate_file_hash(
            video_path
        )
    )


    previous_record = (
        processed_registry.get(
            video_name
        )
    )


    if (
        previous_record
        and
        previous_record.get(
            "status"
        ) == "completed"
        and
        previous_record.get(
            "hash"
        ) == current_hash
    ):

        skipped_videos.append(
            video_name
        )

        print(
            "  -> Already processed. SKIPPED"
        )


    else:

        videos_to_process.append(
            (
                video_path,
                current_hash
            )
        )


        if previous_record:

            print(
                "  -> Video changed. Will process."
            )

        else:

            print(
                "  -> New video. Will process."
            )


print("------------------------------------------")

print(
    f"Total Videos    : {len(video_files)}"
)

print(
    f"Already Done    : {len(skipped_videos)}"
)

print(
    f"Need Processing : {len(videos_to_process)}"
)

print("==========================================")


# =========================================================
# NOTHING NEW
# =========================================================

if not videos_to_process:

    print()
    print("==========================================")
    print("DATASET ALREADY UP TO DATE")
    print("==========================================")

    print(
        "No new or changed videos found."
    )

    print(
        "YOLO + ByteTrack processing was NOT started."
    )

    print(
        f"Dataset: {OUTPUT_CSV}"
    )

    print("==========================================")

    raise SystemExit


# =========================================================
# OPEN CSV
# =========================================================

csv_mode = (
    "a"
    if csv_exists
    else
    "w"
)


csv_file = open(
    OUTPUT_CSV,
    mode=csv_mode,
    newline="",
    encoding="utf-8"
)


csv_writer = csv.writer(
    csv_file
)


if not csv_exists:

    csv_writer.writerow(
        CSV_HEADER
    )


# =========================================================
# GLOBAL STATISTICS
# =========================================================

total_new_rows = 0

successfully_processed = 0

failed_videos = []


# =========================================================
# PROCESS ONLY NEW / CHANGED VIDEOS
# =========================================================

for processing_index, (
    video_path,
    video_hash
) in enumerate(
    videos_to_process,
    start=1
):

    video_name = (
        video_path.name
    )


    # =====================================================
    # IMPORTANT:
    # CHANGED VIDEO SAFETY
    # =====================================================
    #
    # If a video with the SAME filename has changed,
    # appending its new rows would create duplicate source
    # versions in the CSV.
    #
    # For now we stop safely and tell the user.
    #
    # New videos can be appended normally.
    # =====================================================

    previous_record = (
        processed_registry.get(
            video_name
        )
    )


    if (
        previous_record
        and
        previous_record.get(
            "hash"
        ) != video_hash
    ):

        print()
        print(
            f"WARNING: {video_name} has changed."
        )

        print(
            "The same filename already exists in the dataset."
        )

        print(
            "Changed-video replacement must be handled separately"
        )

        print(
            "to avoid duplicate old/new rows."
        )


        failed_videos.append(
            video_name
        )

        continue


    # =====================================================
    # ASSIGN UNIQUE VIDEO ID
    # =====================================================

    video_id = (
        f"VIDEO_{next_video_number:03d}"
    )

    next_video_number += 1


    print()
    print("==========================================")

    print(
        f"PROCESSING NEW VIDEO {processing_index}/{len(videos_to_process)}"
    )

    print("==========================================")

    print(
        f"Video ID   : {video_id}"
    )

    print(
        f"Video Name : {video_name}"
    )

    print(
        f"Video Path : {video_path}"
    )


    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        print(
            f"ERROR: Could not open {video_name}"
        )

        failed_videos.append(
            video_name
        )

        continue


    # =====================================================
    # VIDEO INFORMATION
    # =====================================================

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if video_fps <= 0:

        print(
            "WARNING: Invalid FPS metadata."
        )

        print(
            "Using fallback FPS = 30"
        )

        video_fps = 30.0


    frame_diagonal = math.sqrt(
        width ** 2
        +
        height ** 2
    )


    frame_area = (
        width
        *
        height
    )


    print(
        f"Resolution : {width} x {height}"
    )

    print(
        f"FPS        : {video_fps:.2f}"
    )

    print(
        f"Frames     : {total_frames}"
    )

    print("------------------------------------------")


    # =====================================================
    # FRESH YOLO + BYTETRACK
    # =====================================================

    video_model = YOLO(
        MODEL_PATH
    )


    # =====================================================
    # TRACK MEMORY
    # =====================================================

    track_memory = {}


    # =====================================================
    # VIDEO STATISTICS
    # =====================================================

    frame_number = 0

    video_rows_generated = 0

    manually_stopped = False


    # =====================================================
    # TEMPORARY ROW STORAGE
    # =====================================================
    #
    # Rows for the current video are stored temporarily.
    #
    # Only after the COMPLETE video finishes successfully
    # are they written to movement_features.csv.
    #
    # Therefore pressing Q / failure will not leave a
    # half-processed video inside the main dataset.
    # =====================================================

    pending_rows = []


    # =====================================================
    # FRAME LOOP
    # =====================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_number += 1


        timestamp_sec = (
            (frame_number - 1)
            /
            video_fps
        )


        # =================================================
        # YOLO + BYTETRACK
        # =================================================

        results = video_model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            classes=[0],

            conf=CONFIDENCE,

            imgsz=INFERENCE_SIZE,

            device=DEVICE,

            verbose=False
        )


        current_tracked_persons = 0


        # =================================================
        # PROCESS TRACKED PEOPLE
        # =================================================

        for result in results:

            boxes = (
                result.boxes
            )


            if boxes is None:

                continue


            if boxes.id is None:

                continue


            for box in boxes:

                if box.id is None:

                    continue


                current_tracked_persons += 1


                # =========================================
                # TRACK ID
                # =========================================

                track_id = int(
                    box.id[0].item()
                )


                # =========================================
                # DETECTION CONFIDENCE
                # =========================================

                if box.conf is not None:

                    detection_confidence = float(
                        box.conf[0].item()
                    )

                else:

                    detection_confidence = 0.0


                # =========================================
                # BOUNDING BOX
                # =========================================

                x1, y1, x2, y2 = map(
                    float,
                    box.xyxy[0].tolist()
                )


                bbox_width = max(
                    0.0,
                    x2 - x1
                )

                bbox_height = max(
                    0.0,
                    y2 - y1
                )


                bbox_area = (
                    bbox_width
                    *
                    bbox_height
                )


                bbox_area_ratio = (
                    bbox_area
                    /
                    frame_area

                    if frame_area > 0

                    else 0.0
                )


                # =========================================
                # CENTER
                # =========================================

                center_x = (
                    x1 + x2
                ) / 2.0

                center_y = (
                    y1 + y2
                ) / 2.0


                current_position = (
                    center_x,
                    center_y
                )


                normalized_x = (
                    center_x / width
                    if width > 0
                    else 0.0
                )

                normalized_y = (
                    center_y / height
                    if height > 0
                    else 0.0
                )


                # =========================================
                # DEFAULT TEMPORAL FEATURES
                # =========================================

                motion_valid = 0

                movement_distance_px = 0.0
                movement_distance_norm = 0.0

                speed_px_s = 0.0
                speed_norm_s = 0.0

                acceleration_px_s2 = 0.0
                acceleration_norm_s2 = 0.0

                direction_angle_deg = 0.0
                direction_change_deg = 0.0

                direction_sin = 0.0
                direction_cos = 0.0

                consecutive_frames = 1


                # =========================================
                # PREVIOUS TRACK STATE
                # =========================================

                previous_state = (
                    track_memory.get(
                        track_id
                    )
                )


                if previous_state is not None:

                    last_seen_frame = (
                        previous_state[
                            "last_seen_frame"
                        ]
                    )


                    if (
                        last_seen_frame
                        ==
                        frame_number - 1
                    ):

                        motion_valid = 1


                        consecutive_frames = (
                            previous_state[
                                "consecutive_frames"
                            ]
                            +
                            1
                        )


                        previous_x, previous_y = (
                            previous_state[
                                "position"
                            ]
                        )


                        # ---------------------------------
                        # MOVEMENT VECTOR
                        # ---------------------------------

                        dx = (
                            center_x
                            -
                            previous_x
                        )

                        dy = (
                            center_y
                            -
                            previous_y
                        )


                        # ---------------------------------
                        # MOVEMENT
                        # ---------------------------------

                        movement_distance_px = (
                            calculate_distance(

                                (
                                    previous_x,
                                    previous_y
                                ),

                                (
                                    center_x,
                                    center_y
                                )
                            )
                        )


                        movement_distance_norm = (
                            movement_distance_px
                            /
                            frame_diagonal

                            if frame_diagonal > 0

                            else 0.0
                        )


                        # ---------------------------------
                        # SPEED
                        # ---------------------------------

                        speed_px_s = (
                            movement_distance_px
                            *
                            video_fps
                        )


                        speed_norm_s = (
                            movement_distance_norm
                            *
                            video_fps
                        )


                        # ---------------------------------
                        # ACCELERATION
                        # ---------------------------------

                        if (
                            previous_state[
                                "motion_valid"
                            ]
                            ==
                            1
                        ):

                            acceleration_px_s2 = (
                                speed_px_s
                                -
                                previous_state[
                                    "speed_px_s"
                                ]
                            ) * video_fps


                            acceleration_norm_s2 = (
                                speed_norm_s
                                -
                                previous_state[
                                    "speed_norm_s"
                                ]
                            ) * video_fps


                        # ---------------------------------
                        # DIRECTION
                        # ---------------------------------

                        if movement_distance_px > 0:

                            direction_angle_deg = (
                                calculate_direction_angle(
                                    dx,
                                    dy
                                )
                            )


                            direction_radians = (
                                math.radians(
                                    direction_angle_deg
                                )
                            )


                            direction_sin = (
                                math.sin(
                                    direction_radians
                                )
                            )


                            direction_cos = (
                                math.cos(
                                    direction_radians
                                )
                            )


                            previous_direction_valid = (
                                previous_state[
                                    "direction_valid"
                                ]
                            )


                            if previous_direction_valid:

                                direction_change_deg = (
                                    calculate_direction_change(

                                        previous_state[
                                            "direction_angle"
                                        ],

                                        direction_angle_deg
                                    )
                                )


                # =========================================
                # DIRECTION VALID
                # =========================================

                direction_valid = (
                    motion_valid == 1
                    and
                    movement_distance_px > 0
                )


                # =========================================
                # UPDATE TRACK MEMORY
                # =========================================

                track_memory[
                    track_id
                ] = {

                    "position":
                        current_position,

                    "speed_px_s":
                        speed_px_s,

                    "speed_norm_s":
                        speed_norm_s,

                    "direction_angle":
                        direction_angle_deg,

                    "direction_valid":
                        direction_valid,

                    "motion_valid":
                        motion_valid,

                    "last_seen_frame":
                        frame_number,

                    "consecutive_frames":
                        consecutive_frames
                }


                # =========================================
                # TEMPORARY DATASET ROW
                # =========================================

                pending_rows.append([

                    video_name,
                    video_id,

                    frame_number,

                    round(
                        timestamp_sec,
                        4
                    ),

                    track_id,

                    width,
                    height,

                    round(
                        video_fps,
                        4
                    ),

                    round(
                        center_x,
                        2
                    ),

                    round(
                        center_y,
                        2
                    ),

                    round(
                        normalized_x,
                        6
                    ),

                    round(
                        normalized_y,
                        6
                    ),

                    round(
                        bbox_width,
                        2
                    ),

                    round(
                        bbox_height,
                        2
                    ),

                    round(
                        bbox_area_ratio,
                        8
                    ),

                    round(
                        detection_confidence,
                        6
                    ),

                    motion_valid,

                    round(
                        movement_distance_px,
                        4
                    ),

                    round(
                        movement_distance_norm,
                        8
                    ),

                    round(
                        speed_px_s,
                        4
                    ),

                    round(
                        speed_norm_s,
                        8
                    ),

                    round(
                        acceleration_px_s2,
                        4
                    ),

                    round(
                        acceleration_norm_s2,
                        8
                    ),

                    round(
                        direction_angle_deg,
                        4
                    ),

                    round(
                        direction_change_deg,
                        4
                    ),

                    round(
                        direction_sin,
                        8
                    ),

                    round(
                        direction_cos,
                        8
                    ),

                    consecutive_frames
                ])


                video_rows_generated += 1


                # =========================================
                # VISUALIZATION
                # =========================================

                x1_int = int(x1)
                y1_int = int(y1)
                x2_int = int(x2)
                y2_int = int(y2)


                cv2.rectangle(

                    frame,

                    (
                        x1_int,
                        y1_int
                    ),

                    (
                        x2_int,
                        y2_int
                    ),

                    (
                        0,
                        255,
                        0
                    ),

                    2
                )


                cv2.circle(

                    frame,

                    (
                        int(center_x),
                        int(center_y)
                    ),

                    3,

                    (
                        0,
                        0,
                        255
                    ),

                    -1
                )


                cv2.putText(

                    frame,

                    f"ID {track_id}",

                    (
                        x1_int,

                        max(
                            y1_int - 7,
                            18
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.42,

                    (
                        0,
                        255,
                        0
                    ),

                    1,

                    cv2.LINE_AA
                )


        # =================================================
        # REMOVE STALE TRACK MEMORY
        # =================================================

        stale_track_ids = [

            track_id

            for track_id, state
            in track_memory.items()

            if (
                frame_number
                -
                state[
                    "last_seen_frame"
                ]
            ) > 60
        ]


        for track_id in stale_track_ids:

            del track_memory[
                track_id
            ]


        # =================================================
        # INFORMATION PANEL
        # =================================================

        cv2.rectangle(
            frame,
            (20, 20),
            (530, 225),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "INCREMENTAL FEATURE EXTRACTION",
            (40, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Video : {video_name}",
            (40, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Frame : {frame_number}/{total_frames}",
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"Tracked Persons : {current_tracked_persons}",
            (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            f"New Rows : {video_rows_generated}",
            (40, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Only new videos are processed",
            (40, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # =================================================
        # DISPLAY RESIZE
        # =================================================

        if width > 1280:

            scale = (
                1280
                /
                width
            )


            display_frame = cv2.resize(

                frame,

                (
                    int(
                        width * scale
                    ),

                    int(
                        height * scale
                    )
                ),

                interpolation=cv2.INTER_AREA
            )

        else:

            display_frame = (
                frame
            )


        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(

            "Phase 2 - Incremental Dataset Generator",

            display_frame
        )


        # =================================================
        # PRESS Q
        # =================================================

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            print(
                "Current video stopped by user."
            )

            manually_stopped = True

            break


    # =====================================================
    # CLOSE CURRENT VIDEO
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()


    # =====================================================
    # IF USER STOPPED:
    # DO NOT SAVE PARTIAL VIDEO
    # =====================================================

    if manually_stopped:

        print()
        print(
            f"NOT SAVED: {video_name}"
        )

        print(
            "Partial rows were discarded."
        )

        print(
            "This video will be retried next run."
        )

        continue


    # =====================================================
    # SAVE COMPLETE VIDEO ROWS
    # =====================================================

    csv_writer.writerows(
        pending_rows
    )

    csv_file.flush()


    total_new_rows += (
        len(pending_rows)
    )


    successfully_processed += 1


    # =====================================================
    # REGISTER VIDEO ONLY AFTER SUCCESS
    # =====================================================

    processed_registry[
        video_name
    ] = {

        "hash":
            video_hash,

        "video_id":
            video_id,

        "status":
            "completed",

        "rows":
            len(pending_rows),

        "frames":
            frame_number,

        "fps":
            round(
                video_fps,
                4
            ),

        "width":
            width,

        "height":
            height
    }


    save_processed_registry(
        processed_registry
    )


    print()
    print(
        f"Completed : {video_name}"
    )

    print(
        f"Frames    : {frame_number}"
    )

    print(
        f"Rows Added: {len(pending_rows)}"
    )

    print(
        "Status    : SAVED"
    )

    print("==========================================")


# =========================================================
# CLOSE CSV
# =========================================================

csv_file.close()


# =========================================================
# FINAL SUMMARY
# =========================================================

print()
print("==========================================")
print("INCREMENTAL FEATURE EXTRACTION COMPLETE")
print("==========================================")

print(
    f"Total Videos       : {len(video_files)}"
)

print(
    f"Previously Processed: {len(skipped_videos)}"
)

print(
    f"Newly Processed    : {successfully_processed}"
)

print(
    f"New Rows Added     : {total_new_rows}"
)

print(
    f"Dataset            : {OUTPUT_CSV}"
)

print(
    f"Registry           : {PROCESSED_FILE}"
)


if failed_videos:

    print()
    print("Needs Attention:")

    for failed_video in failed_videos:

        print(
            f" - {failed_video}"
        )


print("==========================================")