#!/usr/bin/env python3
"""Fine-tune the completed 768 checkpoint for 100 additional epochs.

The final best.pt has no optimizer state, so this loads its weights and starts
a new optimizer/schedule. It deliberately does not use Ultralytics resume=True.
"""

from pathlib import Path
import sys

# Resolve the project even when this file is invoked through a symlink.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from ultralytics import YOLO

from train import ScratchTrainer


WEIGHTS = PROJECT_DIR / "runs/detect/yolo26l_scratch_768/weights/best.pt"
DATA = PROJECT_DIR / "robin_extra_seahorse.yaml"
RUN_NAME = "yolo26l_finetune_768_texture_100e"
EPOCHS = 100
LR_START = 1e-4
LR_END = 1e-5


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {WEIGHTS}")
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA}")

    model = YOLO(str(WEIGHTS))
    print(f"Fine-tuning weights: {WEIGHTS}", flush=True)
    print(
        f"Training {EPOCHS} new epochs at imgsz=768; AdamW learning rate "
        f"{LR_START:g} -> {LR_END:g} with cosine decay after 1 warmup epoch. "
        "Screen augmentation is supplied by ScratchTrainer.",
        flush=True,
    )
    return model.train(
        trainer=ScratchTrainer,
        data=str(DATA),
        project=str(PROJECT_DIR / "runs/detect"),
        name=RUN_NAME,
        exist_ok=False,
        resume=False,
        pretrained=True,
        epochs=EPOCHS,
        patience=0,  # Disable early stopping: run the requested 100 epochs.
        imgsz=768,
        batch=8,
        device=0,
        workers=8,
        cache=False,
        amp=True,
        seed=0,
        deterministic=True,
        save_period=5,
        optimizer="AdamW",
        lr0=LR_START,
        lrf=LR_END / LR_START,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_bias_lr=0.0,
        cos_lr=True,
        cls_pw=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.05,
        cutmix=0.0,
        multi_scale=0.0,
    )


if __name__ == "__main__":
    main()
