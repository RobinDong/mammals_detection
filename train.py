import math
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first


RARE_CLASS_ID = 7  # others
HARD_NEGATIVE_CLASS_ID = 6  # shrimp: the dominant source of others false positives
RARE_IMAGE_FRACTION = 0.002
HARD_NEGATIVE_IMAGE_FRACTION = 0.08


def class_group_sample_weights(
    labels,
    rare_class_id,
    hard_negative_class_id,
    rare_fraction,
    hard_negative_fraction,
):
    """Assign exact probability mass to rare, hard-negative, and other images."""
    common_fraction = 1.0 - rare_fraction - hard_negative_fraction
    if min(rare_fraction, hard_negative_fraction, common_fraction) <= 0.0:
        raise ValueError("rare, hard-negative, and common sampling fractions must all be positive")

    classes = [np.asarray(label["cls"], dtype=np.int64).reshape(-1) for label in labels]
    rare_mask = np.asarray([rare_class_id in cls for cls in classes], dtype=bool)
    # Mixed rare/shrimp images belong to the rare group so every rare image is retained.
    hard_negative_mask = np.asarray(
        [hard_negative_class_id in cls for cls in classes], dtype=bool
    ) & ~rare_mask
    common_mask = ~(rare_mask | hard_negative_mask)
    group_masks = (rare_mask, hard_negative_mask, common_mask)
    group_counts = tuple(int(mask.sum()) for mask in group_masks)
    if any(count == 0 for count in group_counts):
        raise RuntimeError(f"Empty sampling group: counts={group_counts}")

    weights = np.zeros(len(labels), dtype=np.float64)
    for mask, count, fraction in zip(
        group_masks,
        group_counts,
        (rare_fraction, hard_negative_fraction, common_fraction),
        strict=True,
    ):
        weights[mask] = fraction / count
    return weights, group_counts


class ScratchTrainer(DetectionTrainer):
    """Train all parameters from scratch with mild class-aware sampling."""

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
            rare_class_id=RARE_CLASS_ID,
            hard_negative_class_id=HARD_NEGATIVE_CLASS_ID,
            rare_fraction=RARE_IMAGE_FRACTION,
            hard_negative_fraction=HARD_NEGATIVE_IMAGE_FRACTION,
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
            "other=%d (%.1f%%), samples/epoch=%d; expected others/shrimp=%d/%d",
            group_counts[0],
            100 * RARE_IMAGE_FRACTION,
            group_counts[1],
            100 * HARD_NEGATIVE_IMAGE_FRACTION,
            group_counts[2],
            100 * (1.0 - RARE_IMAGE_FRACTION - HARD_NEGATIVE_IMAGE_FRACTION),
            samples,
            round(samples * RARE_IMAGE_FRACTION),
            round(samples * HARD_NEGATIVE_IMAGE_FRACTION),
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
        if len(class_maps) > RARE_CLASS_ID:
            rare_map = float(class_maps[RARE_CLASS_ID])
            metrics["metrics/mAP50-95(others)"] = rare_map
            LOGGER.info("others mAP50-95: %.5f", rare_map)
        return metrics, fitness


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
]


def main():
    # Loading the architecture YAML (not yolo26l.pt) creates random weights.
    model = YOLO("yolo26l.yaml")
    return model.train(
        trainer=ScratchTrainer,
        data="robin_extra.yaml",
        name="yolo26l_scratch_768",
        epochs=200,
        patience=40,
        batch=8,
        imgsz=768,
        save_period=5,
        resume=False,
        pretrained=False,
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

        optimizer="MuSGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
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
        multi_scale=0.0,
    )


if __name__ == "__main__":
    results = main()

# model.export(format="onnx", int8=True)
# model.export(format="openvino", imgsz=640, **{"int8": True, "data": "coco.yaml"})
