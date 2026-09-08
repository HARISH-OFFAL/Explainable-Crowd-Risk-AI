import csv
import math
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


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

OUTPUT_DIR = BASE_DIR / "datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "umn_movement_features.csv"
TEMP_CSV = OUTPUT_DIR / "umn_movement_features.tmp.csv"

MODEL_NAME = "yolo11s.pt"

# UMN video is only 320 x 240.
# 960 would unnecessarily enlarge every frame.
IMAGE_SIZE = 320

CONFIDENCE_THRESHOLD = 0.15
TRACKER_CONFIG = "bytetrack.yaml"

VIDEO_ID = "UMN_001"

# Print progress frequently so we know processing is alive.
PROGRESS_INTERVAL = 100

# Remove tracks not seen for this many frames.
STALE_TRACK_FRAMES = 60


# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = "GPU"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"


# =========================================================
# OUTPUT COLUMNS
# =========================================================

OUTPUT_HEADER = [
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
# CHECK INPUT VIDEO
# =========================================================

if not VIDEO_PATH.exists():

    print()
    print("==========================================")
    print("ERROR")
    print("==========================================")
    print("UMN video was not found.")
    print(f"Expected: {VIDEO_PATH}")
    print("==========================================")

    raise SystemExit


# =========================================================
# REMOVE OLD TEMP FILE
# =========================================================

if TEMP_CSV.exists():

    TEMP_CSV.unlink()


# =========================================================
# OPEN VIDEO
# =========================================================

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():

    print()
    print("ERROR: Could not open UMN video.")
    print(f"Video: {VIDEO_PATH}")

    raise SystemExit


fps = float(
    video.get(cv2.CAP_PROP_FPS)
)

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

frame_width = int(
    video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

frame_diagonal = math.sqrt(
    frame_width ** 2
    +
    frame_height ** 2
)

frame_area = (
    frame_width
    *
    frame_height
)


# =========================================================
# INFORMATION
# =========================================================

print()
print(f"Using {DEVICE_NAME}")

print()
print("==========================================")
print("UMN SPATIO-TEMPORAL FEATURE EXTRACTION")
print("==========================================")
print(f"Video      : {VIDEO_PATH.name}")
print(f"Video ID   : {VIDEO_ID}")
print(f"Resolution : {frame_width} x {frame_height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print(f"YOLO Size  : {IMAGE_SIZE}")
print("------------------------------------------")


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading YOLO11s model...")

model_load_start = time.perf_counter()

model = YOLO(MODEL_NAME)

model_load_time = (
    time.perf_counter()
    -
    model_load_start
)

print(
    f"YOLO model loaded in "
    f"{model_load_time:.2f} seconds."
)

print("ByteTrack will start with first inference.")
print("------------------------------------------")
print("Starting frame processing...")
print("------------------------------------------")


# =========================================================
# TRACK MEMORY
# =========================================================

track_memory = {}


# =========================================================
# COUNTERS
# =========================================================

frame_number = 0
rows_written = 0
frames_with_tracks = 0

processing_start = time.perf_counter()

completed_successfully = False


# =========================================================
# PROCESS VIDEO
# =========================================================

try:

    with open(
        TEMP_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(OUTPUT_HEADER)


        while True:

            ret, frame = video.read()

            if not ret:
                break


            frame_number += 1


            # =============================================
            # TIMESTAMP
            # =============================================

            timestamp_sec = (

                (frame_number - 1)
                /
                fps

                if fps > 0

                else 0.0
            )


            # =============================================
            # SHOW FIRST FRAME STATUS
            # =============================================

            if frame_number == 1:

                print(
                    "Frame 1 loaded. "
                    "Running first YOLO + ByteTrack inference..."
                )


            # =============================================
            # YOLO + BYTETRACK
            # =============================================

            results = model.track(
                frame,

                persist=True,

                tracker=TRACKER_CONFIG,

                classes=[0],

                conf=CONFIDENCE_THRESHOLD,

                imgsz=IMAGE_SIZE,

                device=DEVICE,

                verbose=False
            )


            # =============================================
            # PROCESS TRACKED PERSONS
            # =============================================

            if results:

                result = results[0]

                if (
                    result.boxes is not None
                    and
                    len(result.boxes) > 0
                ):

                    boxes = result.boxes

                    tracked_in_frame = 0


                    for box in boxes:

                        if box.id is None:
                            continue


                        tracked_in_frame += 1


                        track_id = int(
                            box.id.item()
                        )


                        confidence = float(
                            box.conf.item()
                        )


                        x1, y1, x2, y2 = map(
                            float,
                            box.xyxy[0].tolist()
                        )


                        # =================================
                        # BOUNDING BOX
                        # =================================

                        bbox_width = max(
                            x2 - x1,
                            0.0
                        )

                        bbox_height = max(
                            y2 - y1,
                            0.0
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


                        # =================================
                        # CENTER POSITION
                        # =================================

                        center_x = (
                            x1 + x2
                        ) / 2.0

                        center_y = (
                            y1 + y2
                        ) / 2.0


                        normalized_x = (

                            center_x
                            /
                            frame_width

                            if frame_width > 0

                            else 0.0
                        )


                        normalized_y = (

                            center_y
                            /
                            frame_height

                            if frame_height > 0

                            else 0.0
                        )


                        # =================================
                        # DEFAULT TEMPORAL VALUES
                        # =================================

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


                        previous = track_memory.get(
                            track_id
                        )


                        # =================================
                        # CONSECUTIVE FRAME MOTION
                        # =================================

                        if (
                            previous is not None
                            and
                            previous["last_seen_frame"]
                            ==
                            frame_number - 1
                        ):

                            motion_valid = 1


                            dx = (
                                center_x
                                -
                                previous["x"]
                            )

                            dy = (
                                center_y
                                -
                                previous["y"]
                            )


                            movement_distance_px = math.hypot(
                                dx,
                                dy
                            )


                            movement_distance_norm = (

                                movement_distance_px
                                /
                                frame_diagonal

                                if frame_diagonal > 0

                                else 0.0
                            )


                            speed_px_s = (
                                movement_distance_px
                                *
                                fps
                            )


                            speed_norm_s = (
                                movement_distance_norm
                                *
                                fps
                            )


                            # =============================
                            # DIRECTION
                            # =============================

                            current_direction = None


                            if movement_distance_px > 0:

                                direction_radians = math.atan2(
                                    dy,
                                    dx
                                )


                                direction_angle_deg = math.degrees(
                                    direction_radians
                                )


                                if direction_angle_deg < 0:

                                    direction_angle_deg += 360.0


                                direction_sin = math.sin(
                                    direction_radians
                                )

                                direction_cos = math.cos(
                                    direction_radians
                                )


                                current_direction = (
                                    direction_angle_deg
                                )


                            # =============================
                            # ACCELERATION
                            # =============================

                            if (
                                previous["motion_valid"]
                                ==
                                1
                            ):

                                acceleration_px_s2 = (

                                    speed_px_s
                                    -
                                    previous["speed_px"]

                                ) * fps


                                acceleration_norm_s2 = (

                                    speed_norm_s
                                    -
                                    previous["speed_norm"]

                                ) * fps


                                previous_direction = (
                                    previous["direction"]
                                )


                                if (
                                    previous_direction
                                    is not None
                                    and
                                    current_direction
                                    is not None
                                ):

                                    difference = abs(

                                        current_direction
                                        -
                                        previous_direction
                                    )


                                    direction_change_deg = min(
                                        difference,
                                        360.0 - difference
                                    )


                            consecutive_frames = (

                                previous[
                                    "consecutive_frames"
                                ]
                                +
                                1
                            )


                        else:

                            current_direction = None


                        # =================================
                        # UPDATE TRACK MEMORY
                        # =================================

                        track_memory[track_id] = {

                            "x": center_x,

                            "y": center_y,

                            "speed_px": speed_px_s,

                            "speed_norm": speed_norm_s,

                            "direction": current_direction,

                            "last_seen_frame": frame_number,

                            "consecutive_frames": consecutive_frames,

                            "motion_valid": motion_valid
                        }


                        # =================================
                        # WRITE FEATURE ROW
                        # =================================

                        writer.writerow([

                            VIDEO_PATH.name,
                            VIDEO_ID,

                            frame_number,

                            round(
                                timestamp_sec,
                                6
                            ),

                            track_id,

                            frame_width,
                            frame_height,

                            round(
                                fps,
                                4
                            ),

                            round(
                                center_x,
                                4
                            ),

                            round(
                                center_y,
                                4
                            ),

                            round(
                                normalized_x,
                                8
                            ),

                            round(
                                normalized_y,
                                8
                            ),

                            round(
                                bbox_width,
                                4
                            ),

                            round(
                                bbox_height,
                                4
                            ),

                            round(
                                bbox_area_ratio,
                                8
                            ),

                            round(
                                confidence,
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


                        rows_written += 1


                    if tracked_in_frame > 0:

                        frames_with_tracks += 1


            # =============================================
            # REMOVE STALE TRACKS
            # =============================================

            stale_ids = [

                track_id

                for track_id, state
                in track_memory.items()

                if (
                    frame_number
                    -
                    state["last_seen_frame"]
                )
                >
                STALE_TRACK_FRAMES
            ]


            for track_id in stale_ids:

                del track_memory[track_id]


            # =============================================
            # PROGRESS
            # =============================================

            if (
                frame_number == 1
                or
                frame_number % PROGRESS_INTERVAL == 0
                or
                frame_number == total_frames
            ):

                elapsed = (
                    time.perf_counter()
                    -
                    processing_start
                )


                processing_fps = (

                    frame_number
                    /
                    elapsed

                    if elapsed > 0

                    else 0.0
                )


                progress = (

                    frame_number
                    /
                    total_frames
                    *
                    100.0

                    if total_frames > 0

                    else 0.0
                )


                remaining_frames = max(
                    total_frames - frame_number,
                    0
                )


                eta_seconds = (

                    remaining_frames
                    /
                    processing_fps

                    if processing_fps > 0

                    else 0.0
                )


                print(
                    f"Frame {frame_number}/{total_frames} "
                    f"({progress:.1f}%) | "
                    f"Rows {rows_written} | "
                    f"Speed {processing_fps:.2f} FPS | "
                    f"ETA {eta_seconds / 60:.1f} min"
                )


        completed_successfully = True


except KeyboardInterrupt:

    print()
    print("==========================================")
    print("PROCESSING STOPPED BY USER")
    print("==========================================")
    print(f"Frames Processed : {frame_number}")
    print(f"Temporary File   : {TEMP_CSV}")
    print("------------------------------------------")
    print(
        "The incomplete temporary dataset will NOT "
        "be used as the final UMN dataset."
    )
    print("==========================================")


finally:

    video.release()


# =========================================================
# SUCCESSFUL COMPLETION
# =========================================================

if completed_successfully:

    # Replace old final output only after the entire
    # source video has been processed successfully.

    os.replace(
        TEMP_CSV,
        OUTPUT_CSV
    )


    total_time = (
        time.perf_counter()
        -
        processing_start
    )


    average_processing_fps = (

        frame_number
        /
        total_time

        if total_time > 0

        else 0.0
    )


    print()
    print("==========================================")
    print("UMN FEATURE EXTRACTION COMPLETE")
    print("==========================================")
    print(f"Frames Processed   : {frame_number}")
    print(f"Frames With Tracks : {frames_with_tracks}")
    print(f"Rows Generated     : {rows_written}")
    print(
        f"Processing Speed   : "
        f"{average_processing_fps:.2f} FPS"
    )
    print(
        f"Total Time         : "
        f"{total_time / 60:.2f} minutes"
    )
    print(f"Output Dataset     : {OUTPUT_CSV}")
    print("------------------------------------------")
    print(
        "Speed values are image/video-space measurements."
    )
    print(
        "Normalized movement features are also preserved."
    )
    print(
        "No LOW/MEDIUM/HIGH risk labels were generated."
    )
    print("==========================================")


else:

    # Remove incomplete temporary dataset.
    if TEMP_CSV.exists():

        TEMP_CSV.unlink()

        print()
        print(
            "Incomplete temporary CSV deleted safely."
        )