# Crowd-person detector workflow

This workflow is for a labelled, person-only YOLO dataset. It does not invent
labels or counts. Keep train/val/test split by source video to avoid temporal
leakage.

1. Sample source videos:

```powershell
python ml/crowd_detector/scripts/extract_frames.py phase2/videos/gem.mp4 phase2/videos/mall.mp4 --interval 1
```

2. Manually annotate sampled images in YOLO format (`0 = person`) and place
images/labels in `dataset/images/{train,val,test}` and
`dataset/labels/{train,val,test}`.

3. Validate before training:

```powershell
python ml/crowd_detector/scripts/validate_dataset.py
```

4. Fine-tune from the existing checkpoint:

```powershell
python ml/crowd_detector/training/train.py --model yolo11s.pt --device cpu
```

5. Evaluate the custom checkpoint on the held-out test split:

```powershell
python ml/crowd_detector/training/evaluate.py --model path/to/best.pt
```

Copy only an evaluated checkpoint to
`ml/crowd_detector/models/custom/crowd_person_best.pt`. The application keeps
`yolo11s.pt` as the fallback and selects the custom model only when
`CROWD_MODEL=custom` is set before starting the backend. If loading fails, the
pretrained model is used and a warning is logged.

No custom metrics are reported by this repository yet because no labelled
crowd-person train/validation/test split has been supplied. Do not claim an
accuracy improvement until the evaluation script has been run.
