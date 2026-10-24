"""Small, deterministic validation probe for classification under screen texture.

This supplements full clean mAP. It reports localized top-class recall, NOT mAP:
for each GT, take the highest-confidence detection with IoU >= .5 and conf >= .25.
Count it only if its class matches. Duplicate classes on the same box therefore
cannot hide a higher-scoring wrong class. Samples come only from the val loader.
"""

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou

from training_augmentations import ScreenCaptureAugmentation


def select_probe_labels(labels, nc, per_class=16):
    """Stable, class-balanced image selection, independent of model predictions."""
    selected = {}
    ordered = sorted(labels, key=lambda x: hashlib.sha256(x["im_file"].encode()).digest())
    for cls in range(nc):
        candidates = [x for x in ordered if cls in np.asarray(x["cls"]).reshape(-1)]
        for label in candidates[:per_class]:
            selected[label["im_file"]] = label
    return list(selected.values())


def localized_class_counts(predictions, gt_boxes, gt_classes, nc):
    """Return per-class correct/total GT counts; the highest score wins overlaps."""
    total = np.bincount(gt_classes.cpu().numpy().astype(int), minlength=nc)
    correct = np.zeros(nc, dtype=np.int64)
    predictions = predictions[predictions[:, 4] >= 0.25]
    if len(predictions) and len(gt_boxes):
        ious = box_iou(gt_boxes, predictions[:, :4])
        scores = predictions[:, 4].expand(len(gt_boxes), -1).masked_fill(ious < 0.5, -1)
        best_scores, indices = scores.max(dim=1)
        matched = (best_scores >= 0.25) & (predictions[indices, 5].long() == gt_classes)
        correct = np.bincount(gt_classes[matched].cpu().numpy().astype(int), minlength=nc)
    return correct, total


class TextureValidationProbe:
    def __init__(self, labels, names, save_dir, per_class=16, sizes=(768, 1024)):
        self.labels = select_probe_labels(labels, len(names), per_class)
        self.names = names
        self.sizes = sizes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "texture_probe_manifest.json").write_text(json.dumps({
            "source": "validation loader only", "sizes": sizes,
            "metric": "macro localized top-class recall at conf=.25 and IoU=.5 (not mAP)",
            "images": [x["im_file"] for x in self.labels],
        }, indent=2))

    @torch.inference_mode()
    def __call__(self, model, device, epoch):
        nc = len(self.names)
        counts = {(size, mode): [np.zeros(nc, np.int64), np.zeros(nc, np.int64)]
                  for size in self.sizes for mode in ("clean", "texture")}
        was_training = model.training
        model.eval()
        try:
            for label in self.labels:
                image = cv2.imread(label["im_file"])
                if image is None:
                    raise RuntimeError(f"Unreadable validation probe image: {label['im_file']}")
                if not label.get("normalized", True) or label.get("bbox_format", "xywh") != "xywh":
                    raise ValueError("Texture probe expects normalized xywh validation labels")
                h, w = image.shape[:2]
                seed = int.from_bytes(hashlib.sha256(label["im_file"].encode()).digest()[:8], "little")
                for size in self.sizes:
                    # Fixed square inputs also test the fixed-shape ONNX deployment layout.
                    im = LetterBox((size, size), auto=False, scaleup=True)(image=image)
                    gain = size / max(h, w)
                    pad = (round((size - round(w * gain)) / 2 - 0.1),
                           round((size - round(h * gain)) / 2 - 0.1))
                    gt = xywh2xyxy(np.asarray(label["bboxes"], dtype=np.float32).copy())
                    gt *= np.array([w, h, w, h]) * gain
                    gt += np.array([pad[0], pad[1], pad[0], pad[1]])
                    gt = torch.as_tensor(gt, device=device)
                    classes = torch.as_tensor(label["cls"].reshape(-1), dtype=torch.long, device=device)
                    for mode in ("clean", "texture"):
                        variant = im if mode == "clean" else ScreenCaptureAugmentation.apply_image(
                            im, random.Random(seed + size), mode="moire")
                        x = torch.from_numpy(np.ascontiguousarray(variant[:, :, ::-1].transpose(2, 0, 1)))
                        x = x.to(device=device, dtype=torch.float32).unsqueeze(0) / 255.0
                        # Float32 keeps the probe identical between standalone and trainer calls.
                        result = model(x)
                        pred = result[0] if isinstance(result, tuple) else result
                        if pred.ndim != 3 or pred.shape[-1] != 6:
                            raise ValueError("Texture probe requires YOLO26 end-to-end [B,N,6] output")
                        correct, total = localized_class_counts(pred[0], gt, classes, nc)
                        counts[size, mode][0] += correct
                        counts[size, mode][1] += total
        finally:
            model.train(was_training)
        metrics, details = {}, {}
        for (size, mode), (correct, total) in counts.items():
            present = total > 0
            recall = np.divide(correct, total, out=np.zeros(nc, float), where=present)
            metrics[f"probe/{mode}_{size}"] = float(recall[present].mean()) if present.any() else 0.0
            metrics[f"probe/bird_{mode}_{size}"] = float(recall[1]) if nc > 1 else 0.0
            details[f"{mode}_{size}"] = {str(self.names[i]): {"correct": int(correct[i]), "total": int(total[i])}
                                          for i in range(nc)}
        metrics["probe/score"] = float(np.mean([metrics[f"probe/{m}_{s}"]
                                               for s in self.sizes for m in ("clean", "texture")]))
        with (self.save_dir / "texture_probe.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "metrics": metrics, "classes": details}) + "\n")
        return metrics
