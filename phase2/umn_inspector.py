import cv2
from pathlib import Path


# =========================================================
# PATH CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    BASE_DIR
    / "public_datasets"
    / "umn"
    / "Crowd-Activity-All.avi"
)


# =========================================================
# OPEN VIDEO
# =========================================================

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    print("ERROR: Could not open UMN video.")
    print(f"Path: {VIDEO_PATH}")
    raise SystemExit


fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

duration_sec = total_frames / fps if fps > 0 else 0


print()
print("==========================================")
print("UMN DATASET INSPECTOR")
print("==========================================")
print(f"Video        : {VIDEO_PATH.name}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Duration     : {duration_sec:.2f} seconds")
print("------------------------------------------")
print("Controls:")
print("SPACE  -> Pause / Resume")
print("RIGHT  -> Move forward 30 frames")
print("LEFT   -> Move backward 30 frames")
print("N      -> Print current frame number")
print("Q      -> Quit")
print("==========================================")
print()


paused = False


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    if not paused:

        ret, frame = video.read()

        if not ret:
            print("End of video reached.")
            break

    else:

        current_frame = int(
            video.get(cv2.CAP_PROP_POS_FRAMES)
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            max(current_frame - 1, 0)
        )

        ret, frame = video.read()

        if not ret:
            break


    current_frame = int(
        video.get(cv2.CAP_PROP_POS_FRAMES)
    )

    current_time = (
        current_frame / fps
        if fps > 0
        else 0
    )


    # =====================================================
    # DISPLAY INFO
    # =====================================================

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        f"Frame: {current_frame}/{total_frames}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    cv2.putText(
        display_frame,
        f"Time: {current_time:.2f} sec",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    status_text = (
        "PAUSED"
        if paused
        else "PLAYING"
    )

    cv2.putText(
        display_frame,
        status_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )


    cv2.imshow(
        "UMN Dataset Inspector",
        display_frame
    )


    # =====================================================
    # KEYBOARD CONTROL
    # =====================================================

    delay = (
        0
        if paused
        else max(
            int(1000 / fps),
            1
        )
    )

    key = cv2.waitKey(delay) & 0xFF


    # Q = quit
    if key == ord("q"):

        break


    # SPACE = pause/resume
    elif key == 32:

        paused = not paused


    # N = print current frame
    elif key == ord("n"):

        print(
            f"MARKED FRAME -> "
            f"Frame: {current_frame}, "
            f"Time: {current_time:.2f} sec"
        )


    # RIGHT ARROW / D = forward 30 frames
    elif key in [83, ord("d")]:

        target_frame = min(
            current_frame + 30,
            total_frames - 1
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame
        )


    # LEFT ARROW / A = backward 30 frames
    elif key in [81, ord("a")]:

        target_frame = max(
            current_frame - 30,
            0
        )

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame
        )


# =========================================================
# CLEANUP
# =========================================================

video.release()

cv2.destroyAllWindows()

print()
print("UMN inspection finished.")