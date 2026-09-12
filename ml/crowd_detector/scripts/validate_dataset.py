"""Validate a YOLO person dataset before training."""
from argparse import ArgumentParser
from pathlib import Path


def validate_split(root, split):
    images = root / "images" / split
    labels = root / "labels" / split
    if not images.is_dir() or not labels.is_dir():
        raise ValueError(f"Missing images/labels directory for {split}")
    errors = []
    image_paths = [p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    for image in image_paths:
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            errors.append(f"missing label: {image.name}")
            continue
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) != 5:
                errors.append(f"{label}:{line_no}: expected 5 values")
                continue
            try:
                class_id, *coords = map(float, values)
            except ValueError:
                errors.append(f"{label}:{line_no}: non-numeric label")
                continue
            if class_id != 0 or any(value < 0 or value > 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                errors.append(f"{label}:{line_no}: invalid person box")
    return len(image_paths), errors


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("ml/crowd_detector/dataset"))
    args = parser.parse_args()
    total = 0
    errors = []
    for split in ("train", "val", "test"):
        count, split_errors = validate_split(args.root, split)
        total += count
        errors.extend(split_errors)
        print(f"{split}: {count} images")
    if errors:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(errors))
    print(f"Dataset validation passed: {total} images, class 0=person")


if __name__ == "__main__":
    main()
