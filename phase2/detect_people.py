from ultralytics import YOLO
import cv2
import time


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------

# YOLO11 Small model
# Better detection accuracy than YOLO11 Nano
model = YOLO("yolo11s.pt")


# ---------------------------------------------------------
# INPUT VIDEO
# ---------------------------------------------------------

video_path = "phase2/videos/mall.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    raise SystemExit


# ---------------------------------------------------------
# VIDEO INFORMATION
# ---------------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")


# ---------------------------------------------------------
# FRAME PROCESSING LOOP
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    start_time = time.time()


    # -----------------------------------------------------
    # YOLO PERSON DETECTION
    # -----------------------------------------------------

    results = model(
        frame,
        classes=[0],      # COCO class 0 = Person
        conf=0.15,        # Lower threshold for distant people
        imgsz=1280,       # Higher resolution for small people
        verbose=False
    )

    person_count = 0


    # -----------------------------------------------------
    # DRAW PERSON DETECTIONS
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            person_count += 1


            # -------------------------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # PERSON CONFIDENCE LABEL
            # -------------------------------------------------

            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


    # -----------------------------------------------------
    # CALCULATE PROCESSING FPS
    # -----------------------------------------------------

    processing_time = time.time() - start_time

    if processing_time > 0:
        processing_fps = 1 / processing_time
    else:
        processing_fps = 0


    # -----------------------------------------------------
    # INFORMATION PANEL
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (340, 110),
        (0, 0, 0),
        -1
    )


    # -----------------------------------------------------
    # DISPLAY PERSON COUNT
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Detected Persons: {person_count}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # DISPLAY PROCESSING FPS
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Processing FPS: {processing_fps:.1f}",
        (35, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # DISPLAY FRAME
    # -----------------------------------------------------

    display_frame = frame.copy()

    # Resize only for screen display.
    # YOLO still processes the original frame.
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
        "Phase 2 - YOLO Crowd Person Detection",
        display_frame
    )


    # -----------------------------------------------------
    # PRESS Q TO EXIT
    # -----------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Detection stopped by user")
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("YOLO person detection finished")