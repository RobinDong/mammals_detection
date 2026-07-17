#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


DEFAULT_MODEL = "runs/detect/train-15/weights/best.pt"
DEFAULT_DATA = "robin_extra.yaml"
DEFAULT_IMGSZ = 640
DEFAULT_CALIBRATION_IMAGES = 1000
DEFAULT_VERIFY_IMAGES = 100
DEFAULT_VERIFY_CONF = 0.05
DEFAULT_MIN_MATCH_RECALL = 0.90
MODEL_NODE_PATTERN = re.compile(r"^/model\.(\d+)(?:/|$)")


def resolve_split_paths(data_yaml: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    dataset_root = Path(cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    split_spec = cfg[split]
    split_path = Path(split_spec)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path

    if split_path.is_file():
        paths = []
        for line in split_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = dataset_root / image_path
            paths.append(image_path.resolve())
        return [p for p in paths if p.exists()]

    if split_path.is_dir():
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(p.resolve() for p in split_path.rglob("*") if p.suffix.lower() in suffixes)

    raise FileNotFoundError(f"Could not resolve {split!r} split from {data_yaml}: {split_path}")


def preprocess_image(image_path: Path, imgsz: int) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC -> RGB CHW
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return image[None]


class YoloCalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths: list[Path], input_name: str, imgsz: int):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self.rewind()

    def rewind(self) -> None:
        self._iterator = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for image_path in self._iterator:
            image = preprocess_image(image_path, self.imgsz)
            if image is not None:
                return {self.input_name: image}
        return None


def choose_calibration_images(data_yaml: Path, split: str, count: int, seed: int) -> list[Path]:
    image_paths = resolve_split_paths(data_yaml, split)
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {data_yaml} split={split!r}")

    rng = random.Random(seed)
    rng.shuffle(image_paths)
    return image_paths[: min(count, len(image_paths))]


def choose_verification_images(
    data_yaml: Path,
    split: str,
    calibration_images: list[Path],
    count: int,
    seed: int,
) -> list[Path]:
    """Choose verification images that were not used to calibrate INT8 ranges."""
    calibration_set = set(calibration_images)
    image_paths = [p for p in resolve_split_paths(data_yaml, split) if p not in calibration_set]
    if not image_paths:
        raise RuntimeError("No independent images remain for post-quantization verification.")

    rng = random.Random(seed + 1)
    rng.shuffle(image_paths)
    return image_paths[: min(count, len(image_paths))]


def export_fp32_onnx(model_path: Path, imgsz: int, simplify: bool, force: bool) -> Path:
    fp32_path = model_path.with_suffix(".onnx")
    if fp32_path.exists() and not force:
        print(f"Using existing FP32 ONNX: {fp32_path}")
        return fp32_path

    print(f"Exporting FP32 ONNX from {model_path}")
    exported = YOLO(str(model_path)).export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        dynamic=False,
    )
    return Path(exported)


def find_detect_head_conv_nodes(fp32_path: Path) -> tuple[list[str], int, int]:
    """Return Conv nodes in the final YOLO module, which is the Detect head."""
    model = onnx.load(str(fp32_path), load_external_data=False)
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    module_indices = []
    for node in conv_nodes:
        match = MODEL_NODE_PATTERN.match(node.name)
        if match:
            module_indices.append(int(match.group(1)))

    if not module_indices:
        raise RuntimeError(
            "Could not identify YOLO module names in the ONNX graph. Refusing to quantize the Detect head silently."
        )

    detect_index = max(module_indices)
    detect_prefix = f"/model.{detect_index}/"
    detect_nodes = [node.name for node in conv_nodes if node.name.startswith(detect_prefix)]
    if not detect_nodes:
        raise RuntimeError(f"No Conv nodes found under inferred Detect head {detect_prefix!r}.")
    return detect_nodes, len(conv_nodes), detect_index


def add_quantization_metadata(
    onnx_path: Path,
    excluded_detect_nodes: list[str],
    activation_type: str,
) -> None:
    model = onnx.load(str(onnx_path))
    metadata = {item.key: item for item in model.metadata_props}
    values = {
        "quantization.strategy": "mixed-precision-conv",
        "quantization.activation_type": activation_type,
        "quantization.detect_head": "fp32" if excluded_detect_nodes else "int8",
        "quantization.excluded_detect_conv_count": str(len(excluded_detect_nodes)),
    }
    for key, value in values.items():
        if key in metadata:
            metadata[key].value = value
        else:
            item = model.metadata_props.add()
            item.key, item.value = key, value
    onnx.save(model, str(onnx_path))


def quantize_conv_only(
    fp32_path: Path,
    int8_path: Path,
    calibration_images: list[Path],
    imgsz: int,
    quant_format: str,
    activation_type: str,
    quantize_detect_head: bool,
) -> None:
    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = YoloCalibrationReader(calibration_images, input_name, imgsz)
    qformat = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator
    detect_nodes, conv_count, detect_index = find_detect_head_conv_nodes(fp32_path)
    excluded_nodes = [] if quantize_detect_head else detect_nodes
    activation_qtype = QuantType.QUInt8 if activation_type == "u8" else QuantType.QInt8

    head_strategy = (
        f"keeping model.{detect_index} Detect head ({len(excluded_nodes)} Conv nodes) in FP32"
        if excluded_nodes
        else f"quantizing model.{detect_index} Detect head (unsafe for YOLO26)"
    )
    print(f"Quantizing {conv_count - len(excluded_nodes)}/{conv_count} Conv nodes -> {int8_path}; {head_strategy}")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=qformat,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=excluded_nodes,
        per_channel=True,
        activation_type=activation_qtype,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
    )
    add_quantization_metadata(int8_path, excluded_nodes, activation_type)


def extract_scores(output: np.ndarray) -> np.ndarray:
    """Extract confidence scores from YOLO26 end-to-end or traditional YOLO output."""
    if output.ndim != 3:
        raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")
    if output.shape[-1] == 6:  # YOLO26 end-to-end: [x1, y1, x2, y2, confidence, class]
        return output[..., 4]
    if output.shape[1] >= 5:  # Traditional YOLO: [x, y, w, h, class scores...] x anchors
        return output[:, 4:, :]
    raise RuntimeError(f"Could not identify confidence scores in output shape {output.shape}")


def smoke_test(onnx_path: Path, image_path: Path, imgsz: int) -> tuple[float, int, tuple[int, ...]]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    image = preprocess_image(image_path, imgsz)
    if image is None:
        raise RuntimeError(f"Could not read smoke-test image: {image_path}")

    output = session.run(None, {input_name: image})[0]
    output = np.asarray(output)
    scores = extract_scores(output)

    score_max = float(np.max(scores))
    nonzero_scores = int(np.count_nonzero(np.abs(scores) > 1e-8))
    return score_max, nonzero_scores, tuple(output.shape)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-9)


def verify_end2end_quality(
    fp32_path: Path,
    int8_path: Path,
    image_paths: list[Path],
    imgsz: int,
    conf: float,
    min_match_recall: float,
) -> None:
    """Reject YOLO26 INT8 models whose final detections diverge sharply from FP32."""
    sessions = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (fp32_path, int8_path)
    ]
    input_names = [session.get_inputs()[0].name for session in sessions]
    fp32_count = 0
    int8_count = 0
    matched_count = 0
    top_score_diffs = []
    top_class_matches = 0
    checked = 0

    for image_path in image_paths:
        image = preprocess_image(image_path, imgsz)
        if image is None:
            continue
        outputs = [
            np.asarray(session.run(None, {input_name: image})[0])[0]
            for session, input_name in zip(sessions, input_names)
        ]
        fp32_output, int8_output = outputs
        if fp32_output.shape[-1] != 6 or int8_output.shape[-1] != 6:
            print(f"Skipping YOLO26 quality gate for non-end-to-end output shape {fp32_output.shape}.")
            return

        checked += 1
        top_score_diffs.append(abs(float(fp32_output[0, 4] - int8_output[0, 4])))
        top_class_matches += int(fp32_output[0, 5] == int8_output[0, 5])
        fp32_detections = fp32_output[fp32_output[:, 4] >= conf]
        int8_detections = int8_output[int8_output[:, 4] >= conf]
        fp32_count += len(fp32_detections)
        int8_count += len(int8_detections)
        if len(fp32_detections) and len(int8_detections):
            ious = box_iou(fp32_detections[:, :4], int8_detections[:, :4])
            same_class = fp32_detections[:, 5, None] == int8_detections[None, :, 5]
            matched_count += int(((ious >= 0.5) & same_class).any(axis=1).sum())

    if checked == 0:
        raise RuntimeError("Could not read any images for post-quantization verification.")
    if fp32_count == 0:
        raise RuntimeError(f"FP32 model produced no detections at conf={conf}; quality gate cannot be evaluated.")

    match_recall = matched_count / fp32_count
    detection_ratio = int8_count / fp32_count
    mean_top_score_diff = float(np.mean(top_score_diffs))
    top_class_agreement = top_class_matches / checked
    print(
        "INT8 quality gate: "
        f"images={checked} conf={conf:g} match_recall={match_recall:.4f} "
        f"detection_ratio={detection_ratio:.4f} top_class_agreement={top_class_agreement:.4f} "
        f"mean_top_score_abs_diff={mean_top_score_diff:.6f}"
    )

    failures = []
    if match_recall < min_match_recall:
        failures.append(f"match recall {match_recall:.4f} < {min_match_recall:.4f}")
    if not 0.75 <= detection_ratio <= 1.25:
        failures.append(f"detection ratio {detection_ratio:.4f} is outside [0.75, 1.25]")
    if top_class_agreement < 0.90:
        failures.append(f"top-class agreement {top_class_agreement:.4f} < 0.9000")
    if mean_top_score_diff > 0.10:
        failures.append(f"mean top-score error {mean_top_score_diff:.4f} > 0.1000")
    if failures:
        raise RuntimeError("INT8 quality gate failed: " + "; ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a mixed-precision YOLO ONNX model with an FP32 Detect head and INT8 backbone/neck."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Input .pt model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Ultralytics data yaml used for calibration images.")
    parser.add_argument("--split", default="val", help="Dataset split used for calibration.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--calib-images", type=int, default=DEFAULT_CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output", default=None, help="Output INT8 ONNX path.")
    parser.add_argument("--quant-format", choices=("qdq", "qoperator"), default="qdq")
    parser.add_argument(
        "--activation-type",
        choices=("u8", "s8"),
        default="u8",
        help="INT8 activation type. u8 gave the lowest YOLO26l score error in local validation.",
    )
    parser.add_argument(
        "--quantize-detect-head",
        action="store_true",
        help="Also quantize the Detect head. Not recommended for YOLO26 end-to-end models.",
    )
    parser.add_argument("--verify-images", type=int, default=DEFAULT_VERIFY_IMAGES)
    parser.add_argument("--verify-conf", type=float, default=DEFAULT_VERIFY_CONF)
    parser.add_argument("--min-match-recall", type=float, default=DEFAULT_MIN_MATCH_RECALL)
    parser.add_argument("--force-fp32-export", action="store_true", help="Re-export the FP32 ONNX even if it exists.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable FP32 ONNX simplification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calib_images < 1:
        raise ValueError("--calib-images must be at least 1")
    if args.verify_images < 1:
        raise ValueError("--verify-images must be at least 1")
    if not 0.0 < args.verify_conf < 1.0:
        raise ValueError("--verify-conf must be between 0 and 1")
    if not 0.0 <= args.min_match_recall <= 1.0:
        raise ValueError("--min-match-recall must be between 0 and 1")
    model_path = Path(args.model)
    data_yaml = Path(args.data)
    fp32_path = export_fp32_onnx(
        model_path,
        imgsz=args.imgsz,
        simplify=not args.no_simplify,
        force=args.force_fp32_export,
    )
    int8_path = Path(args.output) if args.output else fp32_path.with_name(f"{fp32_path.stem}_int8.onnx")

    calibration_images = choose_calibration_images(
        data_yaml=data_yaml,
        split=args.split,
        count=args.calib_images,
        seed=args.seed,
    )
    verification_images = choose_verification_images(
        data_yaml=data_yaml,
        split=args.split,
        calibration_images=calibration_images,
        count=args.verify_images,
        seed=args.seed,
    )
    print(f"Using {len(calibration_images)} calibration images from {data_yaml} split={args.split!r}")
    print(f"Using {len(verification_images)} independent images for the INT8 quality gate")

    candidate_path = int8_path.with_name(f".{int8_path.stem}.candidate{int8_path.suffix}")
    candidate_path.unlink(missing_ok=True)
    try:
        quantize_conv_only(
            fp32_path=fp32_path,
            int8_path=candidate_path,
            calibration_images=calibration_images,
            imgsz=args.imgsz,
            quant_format=args.quant_format,
            activation_type=args.activation_type,
            quantize_detect_head=args.quantize_detect_head,
        )

        score_max, nonzero_scores, output_shape = smoke_test(candidate_path, verification_images[0], args.imgsz)
        print(f"Smoke test output_shape={output_shape} score_max={score_max:.6g} nonzero_scores={nonzero_scores}")
        if score_max <= 0.0 or nonzero_scores == 0:
            raise RuntimeError("INT8 export produced zero class scores. Do not use this ONNX file.")
        verify_end2end_quality(
            fp32_path=fp32_path,
            int8_path=candidate_path,
            image_paths=verification_images,
            imgsz=args.imgsz,
            conf=args.verify_conf,
            min_match_recall=args.min_match_recall,
        )
        candidate_path.replace(int8_path)
    except Exception:
        candidate_path.unlink(missing_ok=True)
        raise

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.1f} MiB)")
    print(f"INT8 ONNX: {int8_path} ({int8_size:.1f} MiB)")


if __name__ == "__main__":
    main()
