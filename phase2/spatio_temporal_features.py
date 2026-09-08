from ultralytics import YOLO
import cv2
import math
import time
from collections import Counter


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------

model = YOLO("yolo11s.pt")


# ---------------------------------------------------------
# INPUT VIDEO
# ---------------------------------------------------------

video_path = "phase2/videos/mall.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    raise SystemExit


video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully")
print(f"Resolution: {width} x {height}")
print(f"Video FPS: {video_fps:.2f}")


# ---------------------------------------------------------
# TRACK HISTORY
# ---------------------------------------------------------

previous_positions = {}


# ---------------------------------------------------------
# DIRECTION
# ---------------------------------------------------------

def get_direction(dx, dy):

    minimum_movement = 2

    if abs(dx) < minimum_movement and abs(dy) < minimum_movement:
        return "STATIONARY"

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"

    return "DOWN" if dy > 0 else "UP"


# ---------------------------------------------------------
# DRAW MOVEMENT ARROW
# ---------------------------------------------------------

def draw_direction_arrow(frame, center_x, center_y, dx, dy):

    distance = math.sqrt(dx ** 2 + dy ** 2)

    if distance < 2:
        return

    # Fixed arrow length keeps the display clean
    arrow_length = 30

    end_x = int(
        center_x + (dx / distance) * arrow_length
    )

    end_y = int(
        center_y + (dy / distance) * arrow_length
    )

    cv2.arrowedLine(
        frame,
        (center_x, center_y),
        (end_x, end_y),
        (0, 255, 255),
        2,
        tipLength=0.35
    )


# ---------------------------------------------------------
# FRAME PROCESSING
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    start_time = time.time()

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.15,
        imgsz=1280,
        verbose=False
    )

    tracked_person_count = 0

    speeds = []
    directions = []

    moving_count = 0
    stationary_count = 0


    # -----------------------------------------------------
    # PROCESS TRACKED PEOPLE
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        for box in result.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            tracked_person_count += 1

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (
                center_x,
                center_y
            )

            speed = 0.0
            direction = "NEW"

            dx = 0
            dy = 0


            # -------------------------------------------------
            # TEMPORAL MOVEMENT
            # -------------------------------------------------

            if track_id in previous_positions:

                previous_x, previous_y = previous_positions[
                    track_id
                ]

                dx = center_x - previous_x
                dy = center_y - previous_y

                movement_distance = math.sqrt(
                    dx ** 2 + dy ** 2
                )

                speed = movement_distance * video_fps

                direction = get_direction(dx, dy)

                speeds.append(speed)
                directions.append(direction)

                if direction == "STATIONARY":
                    stationary_count += 1
                else:
                    moving_count += 1


            previous_positions[track_id] = current_position


            # -------------------------------------------------
            # CLEAN PERSON BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # ONLY ID ABOVE PERSON
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


            # -------------------------------------------------
            # CENTER POINT
            # -------------------------------------------------

            cv2.circle(
                frame,
                current_position,
                4,
                (0, 0, 255),
                -1
            )


            # -------------------------------------------------
            # DIRECTION ARROW
            # -------------------------------------------------

            if direction != "NEW" and direction != "STATIONARY":

                draw_direction_arrow(
                    frame,
                    center_x,
                    center_y,
                    dx,
                    dy
                )


    # -----------------------------------------------------
    # CROWD SUMMARY
    # -----------------------------------------------------

    if speeds:
        average_speed = sum(speeds) / len(speeds)
    else:
        average_speed = 0.0


    valid_directions = [
        direction
        for direction in directions
        if direction != "STATIONARY"
    ]


    if valid_directions:

        dominant_direction = Counter(
            valid_directions
        ).most_common(1)[0][0]

    else:
        dominant_direction = "NONE"


    # -----------------------------------------------------
    # PROCESSING FPS
    # -----------------------------------------------------

    processing_time = time.time() - start_time

    processing_fps = (
        1 / processing_time
        if processing_time > 0
        else 0
    )


    # -----------------------------------------------------
    # CLEAN INFORMATION PANEL
    # -----------------------------------------------------

    panel_x1 = 20
    panel_y1 = 20
    panel_x2 = 390
    panel_y2 = 230

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "SPATIO-TEMPORAL MONITOR",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Tracked Persons : {tracked_person_count}",
        (40, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Moving          : {moving_count}",
        (40, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Stationary      : {stationary_count}",
        (40, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Avg Speed       : {average_speed:.1f} px/s",
        (40, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Main Direction  : {dominant_direction}",
        (40, 208),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # SMALL LEGEND
    # -----------------------------------------------------

    legend_x = max(width - 330, 20)

    cv2.rectangle(
        frame,
        (legend_x, 20),
        (width - 20, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Green Box : Tracked Person",
        (legend_x + 15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Red Dot   : Person Position",
        (legend_x + 15, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Yellow Arrow : Movement",
        (legend_x + 15, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    display_frame = frame.copy()

    if width > 1280:

        scale = 1280 / width

        display_frame = cv2.resize(
            frame,
            (
                int(width * scale),
                int(height * scale)
            )
        )


    cv2.imshow(
        "Phase 2 - Spatio-Temporal Feature Extraction",
        display_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Spatio-Temporal Feature Extraction finished")