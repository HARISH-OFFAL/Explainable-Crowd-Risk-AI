from ultralytics import YOLO
import cv2
import torch
import time
import math
from collections import defaultdict, deque


# =========================================================
# CONFIGURATION
# =========================================================

VIDEO_PATH = "phase2/videos/mall.mp4"
MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.15
INFERENCE_SIZE = 960

# Number of recent movement values stored for each person
HISTORY_SIZE = 8

# ---------------------------------------------------------
# PROTOTYPE IMAGE-SPACE THRESHOLDS
# ---------------------------------------------------------

# Very high movement speed
FAST_SPEED_THRESHOLD = 180.0       # pixels / second

# Current speed compared with person's recent average
SPEED_INCREASE_RATIO = 2.5

# Avoid detecting tiny movements as direction changes
MIN_DIRECTION_SPEED = 25.0         # pixels / second

# Large direction change
DIRECTION_CHANGE_THRESHOLD = 120   # degrees


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

previous_positions = {}

previous_angles = {}

speed_history = defaultdict(
    lambda: deque(
        maxlen=HISTORY_SIZE
    )
)


# =========================================================
# DISTANCE
# =========================================================

def calculate_distance(
    point1,
    point2
):

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# =========================================================
# MOVEMENT ANGLE
# =========================================================

def calculate_angle(dx, dy):

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    return angle


# =========================================================
# ANGLE DIFFERENCE
# =========================================================

def angle_difference(
    angle1,
    angle2
):

    difference = abs(
        angle1 - angle2
    )

    if difference > 180:

        difference = (
            360 - difference
        )

    return difference


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


    current_positions = {}

    current_angles = {}

    people = []


    # =====================================================
    # PROCESS PEOPLE
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


            # -------------------------------------------------
            # TRACK ID
            # -------------------------------------------------

            track_id = int(
                box.id[0]
            )


            # -------------------------------------------------
            # BOX
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # -------------------------------------------------
            # CENTER
            # -------------------------------------------------

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
            # DEFAULT VALUES
            # =================================================

            speed = 0.0

            movement_angle = None

            abnormal_reasons = []


            # =================================================
            # MOVEMENT
            # =================================================

            if track_id in previous_positions:

                previous_x, previous_y = (
                    previous_positions[
                        track_id
                    ]
                )


                dx = (
                    center_x -
                    previous_x
                )

                dy = (
                    center_y -
                    previous_y
                )


                movement_distance = (
                    calculate_distance(
                        (
                            previous_x,
                            previous_y
                        ),
                        current_position
                    )
                )


                speed = (
                    movement_distance *
                    video_fps
                )


                # =============================================
                # RECENT NORMAL SPEED
                # =============================================

                history = speed_history[
                    track_id
                ]


                if len(history) >= 3:

                    average_previous_speed = (
                        sum(history)
                        /
                        len(history)
                    )

                else:

                    average_previous_speed = 0.0


                # =============================================
                # SIGNAL 1:
                # UNUSUALLY FAST MOVEMENT
                # =============================================

                if speed >= FAST_SPEED_THRESHOLD:

                    abnormal_reasons.append(
                        "FAST"
                    )


                # =============================================
                # SIGNAL 2:
                # SUDDEN SPEED INCREASE
                # =============================================

                if (
                    average_previous_speed
                    >= 10
                    and
                    speed
                    >
                    average_previous_speed
                    *
                    SPEED_INCREASE_RATIO
                ):

                    abnormal_reasons.append(
                        "SPEED SURGE"
                    )


                # =============================================
                # DIRECTION
                # =============================================

                if speed >= MIN_DIRECTION_SPEED:

                    movement_angle = (
                        calculate_angle(
                            dx,
                            dy
                        )
                    )


                    current_angles[
                        track_id
                    ] = movement_angle


                    # =========================================
                    # SIGNAL 3:
                    # SUDDEN DIRECTION CHANGE
                    # =========================================

                    if (
                        track_id
                        in previous_angles
                    ):

                        direction_change = (
                            angle_difference(
                                previous_angles[
                                    track_id
                                ],
                                movement_angle
                            )
                        )


                        if (
                            direction_change
                            >=
                            DIRECTION_CHANGE_THRESHOLD
                        ):

                            abnormal_reasons.append(
                                "DIRECTION CHANGE"
                            )


                # =============================================
                # UPDATE SPEED HISTORY
                # =============================================

                history.append(
                    speed
                )


            # =================================================
            # PERSON DATA
            # =================================================

            people.append({

                "id": track_id,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "center": (
                    center_x,
                    center_y
                ),

                "speed": speed,

                "reasons": abnormal_reasons
            })


    # =====================================================
    # ABNORMAL PERSON COUNT
    # =====================================================

    abnormal_people = [

        person

        for person in people

        if len(
            person["reasons"]
        ) > 0
    ]


    tracked_person_count = len(
        people
    )

    abnormal_person_count = len(
        abnormal_people
    )


    # =====================================================
    # ABNORMAL MOVEMENT RATIO
    # =====================================================

    if tracked_person_count > 0:

        abnormal_ratio = (
            abnormal_person_count
            /
            tracked_person_count
        ) * 100

    else:

        abnormal_ratio = 0.0


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

        reasons = person[
            "reasons"
        ]


        # -------------------------------------------------
        # ABNORMAL
        # -------------------------------------------------

        if reasons:

            box_color = (
                0,
                0,
                255
            )

            status_text = (
                "ABNORMAL"
            )


        # -------------------------------------------------
        # NORMAL
        # -------------------------------------------------

        else:

            box_color = (
                0,
                255,
                0
            )

            status_text = (
                "NORMAL"
            )


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
        # ID
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"ID {track_id}",
            (
                x1,
                max(
                    y1 - 20,
                    18
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            box_color,
            1,
            cv2.LINE_AA
        )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        cv2.putText(
            frame,
            status_text,
            (
                x1,
                max(
                    y1 - 6,
                    30
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            box_color,
            1,
            cv2.LINE_AA
        )


    # =====================================================
    # UPDATE POSITION MEMORY
    # =====================================================

    previous_positions = (
        current_positions.copy()
    )


    # =====================================================
    # UPDATE DIRECTION MEMORY
    # =====================================================

    for (
        track_id,
        angle
    ) in current_angles.items():

        previous_angles[
            track_id
        ] = angle


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
            1 /
            processing_time
        )

    else:

        processing_fps = 0


    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (440, 205),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "ABNORMAL CROWD BEHAVIOUR",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Tracked Persons : {tracked_person_count}",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Abnormal Persons: {abnormal_person_count}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Abnormal Ratio  : {abnormal_ratio:.1f}%",
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # CROWD STATUS
    # -----------------------------------------------------

    if abnormal_person_count == 0:

        crowd_status = "NORMAL"

        status_color = (
            0,
            255,
            0
        )

    else:

        crowd_status = (
            "ABNORMAL MOVEMENT DETECTED"
        )

        status_color = (
            0,
            0,
            255
        )


    cv2.putText(
        frame,
        f"Status: {crowd_status}",
        (40, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        status_color,
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

    cv2.rectangle(
        frame,
        (
            20,
            height - 65
        ),
        (
            440,
            height - 15
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "GREEN: Normal   RED: Abnormal Movement",
        (
            35,
            height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # =====================================================
    # DISPLAY RESIZE
    # =====================================================

    if width > 1280:

        scale = (
            1280 /
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

        display_frame = frame


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Phase 2 - Abnormal Behaviour Analysis",
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

print("Abnormal Behaviour Analysis finished")