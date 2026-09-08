from ultralytics import YOLO
import cv2
import time


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
    # YOLO + BYTETRACK
    # -----------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],      # Person only
        conf=0.15,
        imgsz=1280,
        verbose=False
    )


    tracked_person_count = 0


    # -----------------------------------------------------
    # READ TRACKING RESULTS
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        if boxes.id is None:
            continue


        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            tracked_person_count += 1


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
            # TRACK ID + CONFIDENCE
            # -------------------------------------------------

            label = f"ID {track_id} | {confidence:.2f}"

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
    # PROCESSING FPS
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
        (350, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Tracked Persons: {tracked_person_count}",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

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
        "Phase 2 - YOLO + ByteTrack",
        display_frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Tracking stopped by user")
        break


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("ByteTrack person tracking finished")