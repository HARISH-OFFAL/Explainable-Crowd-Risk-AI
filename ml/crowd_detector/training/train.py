"""Fine-tune YOLO from a pretrained checkpoint; never overwrite project weights."""
from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--data", default="ml/crowd_detector/configs/crowd_person.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    model = YOLO(args.model)
    result = model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                         patience=12, device=args.device, workers=args.workers, seed=42,
                         project="ml/crowd_detector/runs", name="crowd_person")
    print(f"Training output: {result.save_dir}")
    print("Copy the evaluated best.pt to ml/crowd_detector/models/custom/crowd_person_best.pt")


if __name__ == "__main__":
    main()
