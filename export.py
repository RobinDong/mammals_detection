#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
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


DEFAULT_MODEL = "yolov8m_800_20240202.pt"
DEFAULT_DATA = "robin_old.yaml"
DEFAULT_IMGSZ = 640
DEFAULT_CALIBRATION_IMAGES = 1000


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


def quantize_conv_only(
    fp32_path: Path,
    int8_path: Path,
    calibration_images: list[Path],
    imgsz: int,
    quant_format: str,
) -> None:
    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    reader = YoloCalibrationReader(calibration_images, input_name, imgsz)
    qformat = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator

    print(f"Quantizing Conv nodes only -> {int8_path}")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=qformat,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
    )


def smoke_test(onnx_path: Path, image_path: Path, imgsz: int) -> tuple[float, int, tuple[int, ...]]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    image = preprocess_image(image_path, imgsz)
    if image is None:
        raise RuntimeError(f"Could not read smoke-test image: {image_path}")

    output = session.run(None, {input_name: image})[0]
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[1] >= 5:
        scores = output[:, 4:, :]
    elif output.ndim == 3 and output.shape[-1] >= 6:
        scores = output[..., 4:]
    else:
        scores = output

    score_max = float(np.max(scores))
    nonzero_scores = int(np.count_nonzero(np.abs(scores) > 1e-8))
    return score_max, nonzero_scores, tuple(output.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 ONNX INT8 model that keeps the Detect head stable.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Input .pt model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Ultralytics data yaml used for calibration images.")
    parser.add_argument("--split", default="val", help="Dataset split used for calibration.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--calib-images", type=int, default=DEFAULT_CALIBRATION_IMAGES)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output", default=None, help="Output INT8 ONNX path.")
    parser.add_argument("--quant-format", choices=("qdq", "qoperator"), default="qdq")
    parser.add_argument("--force-fp32-export", action="store_true", help="Re-export the FP32 ONNX even if it exists.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable FP32 ONNX simplification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    print(f"Using {len(calibration_images)} calibration images from {data_yaml} split={args.split!r}")

    quantize_conv_only(
        fp32_path=fp32_path,
        int8_path=int8_path,
        calibration_images=calibration_images,
        imgsz=args.imgsz,
        quant_format=args.quant_format,
    )

    score_max, nonzero_scores, output_shape = smoke_test(int8_path, calibration_images[-1], args.imgsz)
    print(f"Smoke test output_shape={output_shape} score_max={score_max:.6g} nonzero_scores={nonzero_scores}")
    if score_max <= 0.0 or nonzero_scores == 0:
        raise RuntimeError("INT8 export produced zero class scores. Do not use this ONNX file.")

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.1f} MiB)")
    print(f"INT8 ONNX: {int8_path} ({int8_size:.1f} MiB)")


if __name__ == "__main__":
    main()
