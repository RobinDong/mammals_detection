import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

try:
    import albumentations as A
except ImportError:
    A = None
import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from training_augmentations import ScreenCaptureAugmentation
from texture_validation import TextureValidationProbe


RARE_CLASS_ID = 7  # others
HARD_NEGATIVE_CLASS_ID = 6  # shrimp: the dominant source of others false positives
RARE_IMAGE_FRACTION = 0.002
HARD_NEGATIVE_IMAGE_FRACTION = 0.08
SEAHORSE_IMAGE_FRACTION = 0.01
SEAHORSE_TRAIN_ROOT = "/data/seahorse/images/train/"
SCREEN_AUGMENTATION_P = 0.35
TRAIN_IMAGE_SIZE = 896
TRAIN_MULTI_SCALE = 1 / 7  # Stride-aligned batch sizes from 768 through 1024.


def check_training_runtime():
    """Reject an incompatible interpreter before loading weights or making a run."""
    if "cls_pw" not in DEFAULT_CFG_DICT:
        raise RuntimeError(
            f"Incompatible training environment: Ultralytics {ULTRALYTICS_VERSION} "
            f"does not support cls_pw. Python: {sys.executable}\n"
            "Activate the project environment with `conda activate yolo`, then "
            "run the same python3 train.py command again. Do not remove cls_pw "
            "to bypass this check: it controls the intended class weighting."
        )


def class_group_sample_weights(
    labels,
    image_files,
    rare_class_id,
    hard_negative_class_id,
    rare_fraction,
    hard_negative_fraction,
    seahorse_fraction,
):
    """Assign probability mass to rare, shrimp, seahorse, and other images."""
    common_fraction = 1.0 - rare_fraction - hard_negative_fraction - seahorse_fraction
    if min(rare_fraction, hard_negative_fraction, seahorse_fraction, common_fraction) <= 0.0:
        raise ValueError("All sampling fractions must be positive")

    classes = [np.asarray(label["cls"], dtype=np.int64).reshape(-1) for label in labels]
    if len(classes) != len(image_files):
        raise ValueError("Labels and image paths have different lengths")
    rare_mask = np.asarray([rare_class_id in cls for cls in classes], dtype=bool)
    # Mixed rare/shrimp images belong to the rare group so every rare image is retained.
    hard_negative_mask = np.asarray(
        [hard_negative_class_id in cls for cls in classes], dtype=bool
    ) & ~rare_mask
    seahorse_mask = np.asarray(
        [os.path.abspath(path).startswith(SEAHORSE_TRAIN_ROOT) for path in image_files],
        dtype=bool,
    ) & ~(rare_mask | hard_negative_mask)
    common_mask = ~(rare_mask | hard_negative_mask | seahorse_mask)
    group_masks = (rare_mask, hard_negative_mask, seahorse_mask, common_mask)
    group_counts = tuple(int(mask.sum()) for mask in group_masks)
    if any(count == 0 for count in group_counts):
        raise RuntimeError(f"Empty sampling group: counts={group_counts}")

    weights = np.zeros(len(labels), dtype=np.float64)
    for mask, count, fraction in zip(
        group_masks,
        group_counts,
        (rare_fraction, hard_negative_fraction, seahorse_fraction, common_fraction),
        strict=True,
    ):
        weights[mask] = fraction / count
    return weights, group_counts


class ScratchTrainer(DetectionTrainer):
    """Train all parameters from scratch with mild class-aware sampling."""

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            augmentation = ScreenCaptureAugmentation(p=SCREEN_AUGMENTATION_P)
            dataset.transforms.insert(-1, augmentation)  # Before image-to-tensor Format.
            LOGGER.info("Training image augmentation: %s", augmentation)
        return dataset

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        # close_mosaic rebuilds the pipeline, so restore this image-only stage.
        self.train_loader.dataset.transforms.insert(
            -1, ScreenCaptureAugmentation(p=SCREEN_AUGMENTATION_P)
        )

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if "epochs" in overrides:
            self.args.epochs = int(overrides["epochs"])
        if "multi_scale" in overrides:
            self.args.multi_scale = float(overrides["multi_scale"])

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1:
            raise RuntimeError("ScratchTrainer currently supports one GPU; use device=0")

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        weights, group_counts = class_group_sample_weights(
            dataset.labels,
            dataset.im_files,
            rare_class_id=RARE_CLASS_ID,
            hard_negative_class_id=HARD_NEGATIVE_CLASS_ID,
            rare_fraction=RARE_IMAGE_FRACTION,
            hard_negative_fraction=HARD_NEGATIVE_IMAGE_FRACTION,
            seahorse_fraction=SEAHORSE_IMAGE_FRACTION,
        )
        samples = len(dataset)
        sampler_generator = torch.Generator().manual_seed(int(self.args.seed))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=samples,
            replacement=True,
            generator=sampler_generator,
        )

        batch_size = min(batch_size, len(dataset))
        batches = math.ceil(samples / batch_size)
        device_count = torch.cuda.device_count()
        cpu_count = os.cpu_count() or 1
        workers = min(cpu_count // max(device_count, 1), self.args.workers, batches)
        loader_generator = torch.Generator().manual_seed(6148914691236517205)
        LOGGER.info(
            "Class-group sampling: others=%d (%.2f%%), shrimp hard negatives=%d (%.1f%%), "
            "seahorse=%d (%.1f%%), other=%d (%.1f%%), samples/epoch=%d; "
            "expected others/shrimp/seahorse=%d/%d/%d",
            group_counts[0],
            100 * RARE_IMAGE_FRACTION,
            group_counts[1],
            100 * HARD_NEGATIVE_IMAGE_FRACTION,
            group_counts[2],
            100 * SEAHORSE_IMAGE_FRACTION,
            group_counts[3],
            100 * (1.0 - RARE_IMAGE_FRACTION - HARD_NEGATIVE_IMAGE_FRACTION - SEAHORSE_IMAGE_FRACTION),
            samples,
            round(samples * RARE_IMAGE_FRACTION),
            round(samples * HARD_NEGATIVE_IMAGE_FRACTION),
            round(samples * SEAHORSE_IMAGE_FRACTION),
        )
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=device_count > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

    def validate(self):
        metrics, fitness = super().validate()
        class_maps = self.validator.metrics.box.maps
        for class_id, name in self.data["names"].items():
            if class_id < len(class_maps):
                metrics[f"metrics/mAP50-95({name})"] = float(class_maps[class_id])
        if len(class_maps) > RARE_CLASS_ID:
            rare_map = float(class_maps[RARE_CLASS_ID])
            metrics["metrics/mAP50-95(others)"] = rare_map
            LOGGER.info("others mAP50-95: %.5f", rare_map)
        if not hasattr(self, "texture_probe"):
            self.texture_probe = TextureValidationProbe(
                self.test_loader.dataset.labels, self.data["names"], self.save_dir
            )
        probe_metrics = self.texture_probe(self.ema.ema, self.device, self.epoch + 1)
        metrics.update(probe_metrics)
        LOGGER.info("Texture probe (localized top-class recall, not mAP): %s", probe_metrics)
        return metrics, fitness

    def save_model(self):
        saved = super().save_model()
        if saved is False:
            return False
        # Keep standard best.pt selection intact. This second checkpoint uses a
        # fixed validation subset at BOTH deployment sizes, clean and corrupted.
        score = self.metrics.get("probe/score")
        metadata = self.wdir / "best_texture.json"
        if not hasattr(self, "_best_texture_score"):
            self._best_texture_score = json.loads(metadata.read_text())["score"] if metadata.exists() else -1.0
        if score is not None and score > self._best_texture_score:
            temporary = self.wdir / "best_texture.pt.tmp"
            shutil.copyfile(self.last, temporary)
            temporary.replace(self.wdir / "best_texture.pt")
            metadata.write_text(json.dumps({"score": score, "epoch": self.epoch + 1}))
            self._best_texture_score = score
        return saved


# Target the failure mode in bad2.jpg: a large waterbird near the lower image
# boundary whose body is partly hidden, leaving the head and neck dominant.
targeted_augmentations = [
    A.OneOf(
        [
            A.MotionBlur(
                blur_limit=(3, 9),
                angle_range=(0, 360),
                direction_range=(-0.5, 0.5),
                allow_shifted=True,
                p=0.75,
            ),
            A.MotionBlur(
                blur_limit=(11, 21),
                angle_range=(0, 360),
                direction_range=(-1.0, 1.0),
                allow_shifted=True,
                p=0.25,
            ),
        ],
        p=0.25,
    ),
    # Bias some samples downward so the labelled animal is clipped by the
    # lower boundary, matching nest/shore/foreground occlusion compositions.
    A.Affine(
        scale=(0.85, 1.10),
        translate_percent={"x": (-0.10, 0.10), "y": (-0.03, 0.20)},
        rotate=(-4.0, 4.0),
        shear=(-2.0, 2.0),
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=(114, 114, 114),
        keep_ratio=True,
        balanced_scale=True,
        p=0.25,
    ),
    # Simulate branches, nest material, and foreground objects hiding parts
    # of the body without erasing the whole animal.
    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(0.04, 0.14),
        hole_width_range=(0.04, 0.14),
        fill="random_uniform",
        p=0.20,
    ),
    # Waterfowl often have a dark head/neck against bright water or sky.
    A.OneOf(
        [
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                shadow_intensity_range=(0.25, 0.50),
                p=0.50,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.20, 0.15),
                contrast_limit=(-0.15, 0.25),
                p=0.50,
            ),
        ],
        p=0.20,
    ),
] if A is not None else []


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train with screen-texture augmentation and validation")
    parser.add_argument("--weights", type=Path, help="Fine-tune these .pt weights with a fresh optimizer")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)
    check_training_runtime()
    if args.weights is not None and not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    finetune = args.weights is not None
    # An explicit checkpoint avoids spending another run learning from scratch.
    model = YOLO(str(args.weights) if finetune else "yolo26l.yaml")
    return model.train(
        trainer=ScratchTrainer,
        data=str(Path(__file__).resolve().with_name("robin_extra_seahorse.yaml")),
        name="yolo26l_finetune_texture_v2" if finetune else "yolo26l_scratch_texture_v2",
        epochs=args.epochs if args.epochs is not None else (30 if finetune else 200),
        patience=40,
        batch=8,
        imgsz=TRAIN_IMAGE_SIZE,
        save_period=5,
        resume=False,
        pretrained=finetune,
        device=0,
        cache=False,
        workers=8,
        amp=True,
        seed=0,
        deterministic=True,
        # Keep the bird-specific Albumentations pipeline disabled for the
        # heterogeneous "others" class. The standard transforms below provide
        # diversity without imposing a waterbird-only prior.
        # augmentations=targeted_augmentations,

        optimizer="AdamW" if finetune else "MuSGD",
        lr0=0.0001 if finetune else 0.01,
        lrf=0.1 if finetune else 0.01,
        momentum=0.9 if finetune else 0.937,
        weight_decay=0.0005,
        warmup_epochs=1.0 if finetune else 3.0,
        warmup_bias_lr=0.0 if finetune else 0.1,
        cos_lr=True,

        # Mild inverse-frequency weighting complements the conservative sampler.
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
        multi_scale=TRAIN_MULTI_SCALE,
    )


if __name__ == "__main__":
    results = main()

# model.export(format="onnx", int8=True)
# model.export(format="openvino", imgsz=640, **{"int8": True, "data": "coco.yaml"})
