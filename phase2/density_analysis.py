from ultralytics import YOLO
import cv2
import time
import torch


# ---------------------------------------------------------
# DEVICE CONFIGURATION
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {device_name}")
else:
    device = "cpu"
    print("Using CPU")


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


# ---------------------------------------------------------
# VIDEO INFORMATION
# ---------------------------------------------------------

video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_area = width * height

print("Video opened successfully")
print(f"Resolution: {width} x {height}")
print(f"Video FPS: {video_fps:.2f}")
print("YOLO Inference Size: 960")


# ---------------------------------------------------------
# DENSITY FUNCTION
# ---------------------------------------------------------

def calculate_density(person_count, frame_area):

    if frame_area <= 0:
        return 0.0, "UNKNOWN"

    # Frame-space density score
    density_score = (
        person_count / frame_area
    ) * 100000

    # -----------------------------------------------------
    # CROWD COUNT BASED LEVEL
    # -----------------------------------------------------

    if person_count <= 60:
        density_level = "LOW"

    elif person_count <= 200:
        density_level = "MEDIUM"

    else:
        density_level = "HIGH"

    return density_score, density_level


# ---------------------------------------------------------
# FRAME PROCESSING
# ---------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    start_time = time.perf_counter()


    # -----------------------------------------------------
    # YOLO + BYTETRACK
    # -----------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",

        # Person class only
        classes=[0],

        # Detection confidence
        conf=0.15,

        # Optimized from 1280 -> 960
        imgsz=960,

        # Automatically CPU / GPU
        device=device,

        verbose=False
    )


    tracked_person_count = 0


    # -----------------------------------------------------
    # PROCESS TRACKED PEOPLE
    # -----------------------------------------------------

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

            tracked_person_count += 1


            # -------------------------------------------------
            # DRAW PERSON BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # DRAW TRACK ID
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


    # -----------------------------------------------------
    # CROWD DENSITY
    # -----------------------------------------------------

    density_score, density_level = calculate_density(
        tracked_person_count,
        frame_area
    )


    # -----------------------------------------------------
    # PROCESSING FPS
    # -----------------------------------------------------

    processing_time = (
        time.perf_counter() - start_time
    )

    if processing_time > 0:
        processing_fps = 1 / processing_time
    else:
        processing_fps = 0


    # -----------------------------------------------------
    # DENSITY COLOR
    # -----------------------------------------------------

    if density_level == "LOW":
        density_color = (0, 255, 0)

    elif density_level == "MEDIUM":
        density_color = (0, 200, 255)

    elif density_level == "HIGH":
        density_color = (0, 0, 255)

    else:
        density_color = (255, 255, 255)


    # -----------------------------------------------------
    # INFORMATION PANEL
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (460, 225),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "CROWD DENSITY ANALYSIS",
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
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Density Score   : {density_score:.2f}",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Density Level   : {density_level}",
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        density_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "LOW: 0-60 | MEDIUM: 61-200 | HIGH: 201+",
        (40, 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Processing FPS  : {processing_fps:.1f}",
        (40, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    # Resize only for display.
    # Detection still uses original frame.
    if width > 1280:

        display_scale = 1280 / width

        display_frame = cv2.resize(
            frame,
            (
                int(width * display_scale),
                int(height * display_scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    else:
        display_frame = frame


    cv2.imshow(
        "Phase 2 - Crowd Density Analysis",
        display_frame
    )


    # -----------------------------------------------------
    # PRESS Q TO EXIT
    # -----------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Crowd Density Analysis finished")