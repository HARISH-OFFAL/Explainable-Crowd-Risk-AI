from ultralytics import YOLO
import cv2
import torch
import time


# =========================================================
# CONFIGURATION
# =========================================================

VIDEO_PATH = "phase2/videos/mall.mp4"

MODEL_PATH = "yolo11s.pt"

CONFIDENCE = 0.15

INFERENCE_SIZE = 960


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
# LOAD YOLO MODEL
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


print("Video opened successfully")

print(
    f"Resolution: {width} x {height}"
)

print(
    f"Video FPS: {video_fps:.2f}"
)


# =========================================================
# READ FIRST FRAME
# =========================================================

ret, first_frame = cap.read()

if not ret:

    print("ERROR: Could not read first frame")

    cap.release()

    raise SystemExit


# =========================================================
# DYNAMIC ZONE SELECTION
# =========================================================

print()
print("====================================")
print("ENTRY / EXIT ZONE CONFIGURATION")
print("====================================")
print()
print("1. Select ENTRY zone")
print("2. Press ENTER")
print("3. Select EXIT zone")
print("4. Press ENTER")
print()
print("Press C while selecting to cancel")
print()


# ---------------------------------------------------------
# SELECT ENTRY ZONE
# ---------------------------------------------------------

entry_roi = cv2.selectROI(
    "Select ENTRY Zone - Press ENTER",
    first_frame,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyWindow(
    "Select ENTRY Zone - Press ENTER"
)


# Check selection
if entry_roi[2] == 0 or entry_roi[3] == 0:

    print("ERROR: ENTRY zone was not selected")

    cap.release()

    cv2.destroyAllWindows()

    raise SystemExit


# ---------------------------------------------------------
# SELECT EXIT ZONE
# ---------------------------------------------------------

exit_roi = cv2.selectROI(
    "Select EXIT Zone - Press ENTER",
    first_frame,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyWindow(
    "Select EXIT Zone - Press ENTER"
)


# Check selection
if exit_roi[2] == 0 or exit_roi[3] == 0:

    print("ERROR: EXIT zone was not selected")

    cap.release()

    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# CONVERT ROI FORMAT
# =========================================================

def convert_roi(roi):

    x, y, w, h = roi

    return (
        int(x),
        int(y),
        int(x + w),
        int(y + h)
    )


entry_zone = convert_roi(entry_roi)

exit_zone = convert_roi(exit_roi)


print()
print("ENTRY Zone:", entry_zone)
print("EXIT Zone :", exit_zone)
print()
print("Monitoring started...")
print("Press Q to stop")


# =========================================================
# RESET VIDEO TO BEGINNING
# =========================================================

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)


# =========================================================
# TRACKING MEMORY
# =========================================================

# IDs currently inside entry zone
inside_entry_zone = set()

# IDs currently inside exit zone
inside_exit_zone = set()


# IDs already counted as entered
counted_entry_ids = set()

# IDs already counted as exited
counted_exit_ids = set()


# Total counters
total_entered = 0

total_exited = 0


# =========================================================
# CHECK POINT INSIDE RECTANGLE
# =========================================================

def point_inside_zone(
    center_x,
    center_y,
    zone
):

    x1, y1, x2, y2 = zone

    return (
        x1 <= center_x <= x2
        and
        y1 <= center_y <= y2
    )


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


    current_tracked_people = 0


    # IDs detected in current frame
    current_entry_ids = set()

    current_exit_ids = set()


    # =====================================================
    # PROCESS TRACKED PEOPLE
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
            # BOUNDING BOX
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            current_tracked_people += 1


            # -------------------------------------------------
            # PERSON CENTER
            # -------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # =================================================
            # ENTRY ZONE CHECK
            # =================================================

            is_inside_entry = point_inside_zone(
                center_x,
                center_y,
                entry_zone
            )


            if is_inside_entry:

                current_entry_ids.add(
                    track_id
                )


                # Count only first entry
                if (
                    track_id
                    not in counted_entry_ids
                ):

                    total_entered += 1

                    counted_entry_ids.add(
                        track_id
                    )

                    print(
                        f"ID {track_id} ENTERED"
                    )


            # =================================================
            # EXIT ZONE CHECK
            # =================================================

            is_inside_exit = point_inside_zone(
                center_x,
                center_y,
                exit_zone
            )


            if is_inside_exit:

                current_exit_ids.add(
                    track_id
                )


                # Count only first exit
                if (
                    track_id
                    not in counted_exit_ids
                ):

                    total_exited += 1

                    counted_exit_ids.add(
                        track_id
                    )

                    print(
                        f"ID {track_id} EXITED"
                    )


            # =================================================
            # PERSON BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # CENTER POINT
            # =================================================

            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )


            # =================================================
            # TRACK ID
            # =================================================

            cv2.putText(
                frame,
                f"ID {track_id}",
                (
                    x1,
                    max(y1 - 7, 18)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


    # =====================================================
    # UPDATE CURRENT ZONE STATE
    # =====================================================

    inside_entry_zone = current_entry_ids

    inside_exit_zone = current_exit_ids


    # =====================================================
    # DRAW ENTRY ZONE
    # =====================================================

    ex1, ey1, ex2, ey2 = entry_zone


    cv2.rectangle(
        frame,
        (ex1, ey1),
        (ex2, ey2),
        (255, 255, 0),
        3
    )


    cv2.putText(
        frame,
        "ENTRY ZONE",
        (
            ex1,
            max(ey1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )


    # =====================================================
    # DRAW EXIT ZONE
    # =====================================================

    xx1, xy1, xx2, xy2 = exit_zone


    cv2.rectangle(
        frame,
        (xx1, xy1),
        (xx2, xy2),
        (0, 165, 255),
        3
    )


    cv2.putText(
        frame,
        "EXIT ZONE",
        (
            xx1,
            max(xy1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2,
        cv2.LINE_AA
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
    # NET FLOW
    # =====================================================

    net_flow = (
        total_entered
        -
        total_exited
    )


    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (420, 255),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "ENTRY / EXIT FLOW ANALYSIS",
        (40, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # TRACKED PERSONS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Tracked Persons : {current_tracked_people}",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # ENTERED
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Total Entered   : {total_entered}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 0),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # EXITED
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Total Exited    : {total_exited}",
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 165, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # PEOPLE IN ENTRY ZONE
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"In Entry Zone   : {len(inside_entry_zone)}",
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # PEOPLE IN EXIT ZONE
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"In Exit Zone    : {len(inside_exit_zone)}",
        (40, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # NET FLOW
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Net Flow        : {net_flow:+d}",
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
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
            max(width - 150, 10),
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
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
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        display_frame = frame


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Phase 2 - Dynamic Entry Exit Flow",
        display_frame
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        print("Stopped by user")

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()


# =========================================================
# FINAL RESULT
# =========================================================

print()
print("====================================")
print("ENTRY / EXIT FLOW RESULT")
print("====================================")

print(
    f"Total Entered : {total_entered}"
)

print(
    f"Total Exited  : {total_exited}"
)

print(
    f"Net Flow      : {total_entered - total_exited}"
)

print("====================================")