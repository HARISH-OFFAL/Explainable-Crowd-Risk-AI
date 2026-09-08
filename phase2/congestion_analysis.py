from ultralytics import YOLO
import cv2
import torch
import time
import math


# =========================================================
# CONFIGURATION
# =========================================================

VIDEO_PATH = "phase2/videos/mall.mp4"
MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.15
INFERENCE_SIZE = 960

# Congestion parameters
NEARBY_DISTANCE = 100       # pixels
MIN_NEIGHBORS = 3           # nearby persons needed
LOW_SPEED_THRESHOLD = 40    # pixels / second


# =========================================================
# DEVICE
# =========================================================

if torch.cuda.is_available():
    DEVICE = 0
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("Using CPU")


# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    raise SystemExit


video_fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


if video_fps <= 0:
    video_fps = 30


print("Video opened successfully")
print(f"Resolution: {width} x {height}")
print(f"Video FPS: {video_fps:.2f}")


# =========================================================
# TRACKING MEMORY
# =========================================================

# Previous center position of each ByteTrack ID
previous_positions = {}


# =========================================================
# DISTANCE FUNCTION
# =========================================================

def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# =========================================================
# CONGESTION LEVEL
# =========================================================

def get_congestion_level(score):

    if score < 25:
        return "LOW"

    elif score < 60:
        return "MEDIUM"

    else:
        return "HIGH"


# =========================================================
# FRAME LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break


    start_time = time.perf_counter()


    # =====================================================
    # YOLO + BYTETRACK
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE,
        imgsz=INFERENCE_SIZE,
        device=DEVICE,
        verbose=False
    )


    # =====================================================
    # CURRENT FRAME DATA
    # =====================================================

    people = []

    current_positions = {}


    # =====================================================
    # GET PERSON POSITIONS
    # =====================================================

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        if boxes.id is None:
            continue


        for box in boxes:

            if box.id is None:
                continue


            track_id = int(box.id[0])


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Person center
            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            current_position = (
                center_x,
                center_y
            )


            current_positions[
                track_id
            ] = current_position


            # =================================================
            # SPEED CALCULATION
            # =================================================

            speed = 0.0


            if track_id in previous_positions:

                previous_position = (
                    previous_positions[track_id]
                )


                movement_distance = (
                    calculate_distance(
                        previous_position,
                        current_position
                    )
                )


                # Image-space speed
                speed = (
                    movement_distance *
                    video_fps
                )


            people.append({
                "id": track_id,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "center": current_position,
                "speed": speed
            })


    # =====================================================
    # CONGESTION ANALYSIS
    # =====================================================

    congested_ids = set()


    for person in people:

        nearby_count = 0

        person_center = person["center"]


        # -------------------------------------------------
        # FIND NEARBY PEOPLE
        # -------------------------------------------------

        for other_person in people:

            if (
                person["id"]
                ==
                other_person["id"]
            ):
                continue


            distance = calculate_distance(
                person_center,
                other_person["center"]
            )


            if distance <= NEARBY_DISTANCE:

                nearby_count += 1


        # -------------------------------------------------
        # SPATIAL + TEMPORAL CONDITION
        # -------------------------------------------------

        is_crowded = (
            nearby_count >= MIN_NEIGHBORS
        )

        is_slow = (
            person["speed"]
            <= LOW_SPEED_THRESHOLD
        )


        # Congested only when:
        # 1. Person has several nearby people
        # 2. Movement is slow
        if is_crowded and is_slow:

            congested_ids.add(
                person["id"]
            )


        person[
            "nearby_count"
        ] = nearby_count


    # =====================================================
    # COUNTS
    # =====================================================

    tracked_person_count = len(
        people
    )

    congested_person_count = len(
        congested_ids
    )


    # =====================================================
    # CONGESTION SCORE
    # =====================================================

    if tracked_person_count > 0:

        congestion_score = (
            congested_person_count
            /
            tracked_person_count
        ) * 100

    else:

        congestion_score = 0.0


    congestion_level = (
        get_congestion_level(
            congestion_score
        )
    )


    # =====================================================
    # AVERAGE SPEED
    # =====================================================

    valid_speeds = [
        person["speed"]
        for person in people
        if person["id"] in previous_positions
    ]


    if valid_speeds:

        average_speed = (
            sum(valid_speeds)
            /
            len(valid_speeds)
        )

    else:

        average_speed = 0.0


    # =====================================================
    # DRAW PEOPLE
    # =====================================================

    for person in people:

        track_id = person["id"]

        x1, y1, x2, y2 = (
            person["box"]
        )

        center_x, center_y = (
            person["center"]
        )


        # -------------------------------------------------
        # CONGESTED PERSON
        # -------------------------------------------------

        if track_id in congested_ids:

            box_color = (
                0,
                0,
                255
            )

            status = "CONGESTED"


        # -------------------------------------------------
        # NORMAL PERSON
        # -------------------------------------------------

        else:

            box_color = (
                0,
                255,
                0
            )

            status = "NORMAL"


        # -------------------------------------------------
        # BOX
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # -------------------------------------------------
        # CENTER
        # -------------------------------------------------

        cv2.circle(
            frame,
            (
                center_x,
                center_y
            ),
            3,
            box_color,
            -1
        )


        # -------------------------------------------------
        # ID + STATUS
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"ID {track_id} {status}",
            (
                x1,
                max(
                    y1 - 7,
                    18
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            box_color,
            1,
            cv2.LINE_AA
        )


    # =====================================================
    # UPDATE PREVIOUS POSITIONS
    # =====================================================

    previous_positions = (
        current_positions.copy()
    )


    # =====================================================
    # PROCESSING FPS
    # =====================================================

    processing_time = (
        time.perf_counter()
        -
        start_time
    )


    if processing_time > 0:

        processing_fps = (
            1 / processing_time
        )

    else:

        processing_fps = 0


    # =====================================================
    # CONGESTION COLOR
    # =====================================================

    if congestion_level == "LOW":

        level_color = (
            0,
            255,
            0
        )

    elif congestion_level == "MEDIUM":

        level_color = (
            0,
            200,
            255
        )

    else:

        level_color = (
            0,
            0,
            255
        )


    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (440, 245),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "CONGESTION ANALYSIS",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Tracked Persons  : {tracked_person_count}",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Congested Persons: {congested_person_count}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Average Speed    : {average_speed:.1f} px/s",
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Congestion Score : {congestion_score:.1f}%",
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Congestion Level : {congestion_level}",
        (40, 212),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        level_color,
        2,
        cv2.LINE_AA
    )


    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {processing_fps:.1f}",
        (
            max(
                width - 150,
                10
            ),
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # =====================================================
    # LEGEND
    # =====================================================

    legend_y = height - 60


    cv2.rectangle(
        frame,
        (20, legend_y),
        (350, height - 15),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "GREEN: Normal   RED: Congested",
        (35, height - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # =====================================================
    # DISPLAY RESIZE
    # =====================================================

    if width > 1280:

        scale = (
            1280 / width
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

        display_frame = frame


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Phase 2 - Congestion Analysis",
        display_frame
    )


    # =====================================================
    # PRESS Q TO STOP
    # =====================================================

    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == ord("q"):

        print("Stopped by user")

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("Congestion Analysis finished")