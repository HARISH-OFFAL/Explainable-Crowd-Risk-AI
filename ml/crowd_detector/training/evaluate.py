"""Evaluate precision/recall/mAP and per-image count MAE/RMSE on test data."""
from argparse import ArgumentParser
from pathlib import Path
import math
from ultralytics import YOLO


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="ml/crowd_detector/configs/crowd_person.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--test-images", type=Path, default=Path("ml/crowd_detector/dataset/images/test"))
    parser.add_argument("--test-labels", type=Path, default=Path("ml/crowd_detector/dataset/labels/test"))
    args = parser.parse_args()
    model = YOLO(args.model)
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, conf=args.conf, max_det=1000, plots=True)
    print(f"Precision: {metrics.box.mp:.6f}")
    print(f"Recall: {metrics.box.mr:.6f}")
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    errors = []
    for image in sorted(args.test_images.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = args.test_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing test label for {image}")
        ground_truth = len([line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()])
        prediction = model.predict(str(image), imgsz=args.imgsz, conf=args.conf, max_det=1000, verbose=False)[0]
        predicted = len(prediction.boxes) if prediction.boxes is not None else 0
        errors.append(predicted - ground_truth)
    if errors:
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
        print(f"Count test images: {len(errors)}")
        print(f"Count MAE: {mae:.6f}")
        print(f"Count RMSE: {rmse:.6f}")
    else:
        print("Count MAE/RMSE unavailable: labelled test images were not supplied.")


if __name__ == "__main__":
    main()
