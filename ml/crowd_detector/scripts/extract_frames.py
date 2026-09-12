"""Sample frames from source videos for manual YOLO person annotation.

Frames are grouped by source video so train/val/test can be split by video,
avoiding temporal leakage. This script never creates labels or detections.
"""
from argparse import ArgumentParser
from pathlib import Path
import cv2


def extract(video, output, interval):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video}")
    fps = max(capture.get(cv2.CAP_PROP_FPS), 1.0)
    frame_step = max(1, round(fps * interval))
    index = saved = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_step == 0:
            name = f"{video.stem}_frame_{index:08d}.jpg"
            if cv2.imwrite(str(output / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                saved += 1
        index += 1
    capture.release()
    return saved


def main():
    parser = ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/crowd_detector/dataset/raw_frames"))
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        print(f"{video}: {extract(video, args.output, args.interval)} frames")


if __name__ == "__main__":
    main()
