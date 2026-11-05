#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

# onnxruntime-gpu already provides the onnxruntime module.  Prevent Ultralytics'
# requirement checker from replacing it with the CPU-only onnxruntime wheel.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.data.augment import LetterBox


DEFAULT_MODEL = "runs/detect/yolo26l_finetune_texture_v2-11/weights/best.pt"
DEFAULT_DATA = "robin_extra_seahorse.yaml"
DEFAULT_IMGSZ = None
DEFAULT_CALIBRATION_IMAGES = 512
DEFAULT_VERIFY_IMAGES = 100
DEFAULT_VERIFY_CONF = 0.05
DEFAULT_MIN_MATCH_RECALL = 0.90
COMPACT_MAX_BYTES = 30_000_000
COMPACT_WEIGHT_GROUP_SIZE = 32
VALIDATION_SETTINGS = dict(rect=False, batch=1, half=False, conf=0.001, iou=0.7, max_det=300)
MODEL_NODE_PATTERN = re.compile(r"^/model\.(\d+)(?:/|$)")
# Keep the early backbone and YOLO26l neck in FP32. Quantize backbone modules 7–10.
DEFAULT_FP32_MODULES = (0, 1, 2, 3, 4, 5, 6, 13, 16, 17, 19, 20, 22)
INTEGER_FP32_MODULES = (0, 1, 2, 3, 4, 13, 16, 17, 19, 20, 22)


def resolve_split_paths(data_yaml: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    dataset_root = Path(cfg.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    split_specs = cfg[split]
    if not isinstance(split_specs, list):
        split_specs = [split_specs]

    paths = []
    for split_spec in split_specs:
        split_path = Path(split_spec)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        if split_path.is_file():
            for line in split_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = split_path.parent / image_path
                if image_path.exists():
                    paths.append(image_path.resolve())
        elif split_path.is_dir():
            suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            paths.extend(sorted(
                p.resolve() for p in split_path.rglob("*") if p.suffix.lower() in suffixes
            ))
        else:
            raise FileNotFoundError(f"Could not resolve {split!r} split from {data_yaml}: {split_path}")
    return list(dict.fromkeys(paths))


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


def stratified_sample(image_paths: list[Path], count: int, seed: int, nc: int) -> list[Path]:
    """Sample each label class, then fill unused slots; never duplicate an image."""
    paths = list(dict.fromkeys(image_paths))
    random.Random(seed).shuffle(paths)
    quota = max(1, count // nc)
    groups = [[] for _ in range(nc)]
    for path in paths:
        label = Path(str(path).replace('/images/', '/labels/')).with_suffix('.txt')
        classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()} if label.exists() else set()
        for cls in classes:
            if not 0 <= cls < nc:
                raise ValueError(f"Invalid class {cls} in {label}")
            if len(groups[cls]) < quota:
                groups[cls].append(path)
        if all(len(group) >= quota for group in groups):
            break
    selected = list(dict.fromkeys(path for group in groups for path in group))
    seen = set(selected)
    for path in paths:
        if len(selected) >= count:
            break
        if path not in seen:
            selected.append(path)
            seen.add(path)
    random.Random(seed + 1).shuffle(selected)
    return selected[:count]


def choose_calibration_images(data_yaml: Path, split: str, count: int, seed: int) -> list[Path]:
    image_paths = resolve_split_paths(data_yaml, split)
    if not image_paths:
        raise RuntimeError(f"No calibration images found in {data_yaml} split={split!r}")

    cfg = yaml.safe_load(data_yaml.read_text())
    return stratified_sample(image_paths, count, seed, len(cfg['names']))


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


def resolve_export_imgsz(model_path: Path, requested_imgsz: int | None) -> int:
    """Use an explicit size, otherwise preserve the training size stored in a PT checkpoint."""
    if requested_imgsz is not None:
        return requested_imgsz
    if model_path.suffix.lower() != ".pt":
        raise ValueError("--imgsz is required when --model is not a .pt checkpoint")

    checkpoint_imgsz = YOLO(str(model_path)).overrides.get("imgsz")
    if not isinstance(checkpoint_imgsz, int) or checkpoint_imgsz < 1:
        raise RuntimeError(f"Could not infer a valid training imgsz from {model_path}; pass --imgsz explicitly")
    print(f"Using checkpoint training image size: {checkpoint_imgsz}")
    return checkpoint_imgsz


def get_static_onnx_input_hw(onnx_path: Path) -> tuple[int, int] | None:
    """Return the fixed ONNX input height/width, or None for a dynamic/unknown input."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    if not model.graph.input:
        return None
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) < 4 or not dims[-2].dim_value or not dims[-1].dim_value:
        return None
    return int(dims[-2].dim_value), int(dims[-1].dim_value)


def get_static_onnx_output_shape(onnx_path: Path) -> tuple[int, ...] | None:
    """Return the first fixed ONNX output shape, or None when any dimension is dynamic."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    if not model.graph.output:
        return None
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if not dims or any(not dim.dim_value for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def get_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Return ONNX metadata without loading external tensor data."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def restore_checkpoint_end2end_mode(model: YOLO) -> bool:
    """Restore YOLO26's class-level end2end flag, which is not serialized as an instance attribute."""
    expected_end2end = bool(model.model.yaml.get("end2end", False))
    if not expected_end2end:
        return False

    detect_head = model.model.model[-1]
    if not hasattr(detect_head, "one2one_cv2") or not hasattr(detect_head, "one2one_cv3"):
        raise RuntimeError("Checkpoint declares end2end=True but its Detect head has no one-to-one branches")
    if not detect_head.end2end:
        print("Restoring end-to-end Detect mode from checkpoint YAML")
    detect_head.end2end = True
    return True


def export_fp32_onnx(model_path: Path, imgsz: int, simplify: bool, force: bool) -> Path:
    fp32_path = model_path.with_suffix(".onnx")
    export_model = YOLO(str(model_path))
    checkpoint_version = str(getattr(export_model, "ckpt", {}).get("version", "")).strip()
    if checkpoint_version and checkpoint_version != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Checkpoint was trained with ultralytics {checkpoint_version}, but the active environment has "
            f"{ULTRALYTICS_VERSION}. Install the checkpoint version before exporting: "
            f"python3 -m pip install --no-deps ultralytics=={checkpoint_version}"
        )
    expected_end2end = restore_checkpoint_end2end_mode(export_model)
    if fp32_path.exists() and not force:
        input_hw = get_static_onnx_input_hw(fp32_path)
        output_shape = get_static_onnx_output_shape(fp32_path)
        onnx_version = get_onnx_metadata(fp32_path).get("version")
        source_is_current = fp32_path.stat().st_mtime >= model_path.stat().st_mtime
        output_mode_is_current = not expected_end2end or (output_shape is not None and output_shape[-1] == 6)
        exporter_version_is_current = not checkpoint_version or onnx_version == checkpoint_version
        if input_hw == (imgsz, imgsz) and source_is_current and output_mode_is_current and exporter_version_is_current:
            print(f"Using existing FP32 ONNX: {fp32_path} (input {imgsz}x{imgsz})")
            return fp32_path

        reasons = []
        if input_hw != (imgsz, imgsz):
            reasons.append(f"input is {input_hw}, expected {(imgsz, imgsz)}")
        if not source_is_current:
            reasons.append("checkpoint is newer than ONNX")
        if not output_mode_is_current:
            reasons.append(f"output is {output_shape}, expected end-to-end (..., 6)")
        if not exporter_version_is_current:
            reasons.append(f"exporter version is {onnx_version!r}, expected {checkpoint_version!r}")
        print(f"Re-exporting stale/incompatible FP32 ONNX: {fp32_path} ({'; '.join(reasons)})")

    print(f"Exporting FP32 ONNX from {model_path}")
    # Ultralytics hides CUDA when exporting on CPU. Restore visibility for calibration and validation.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        exported = export_model.export(format="onnx", imgsz=imgsz, simplify=simplify, dynamic=False)
    finally:
        if visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
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
    fp32_modules: tuple[int, ...] = DEFAULT_FP32_MODULES,
    calibration_providers: list[str] | None = None,
) -> None:
    graph = onnx.load(str(fp32_path), load_external_data=False).graph
    input_name = graph.input[0].name
    reader = YoloCalibrationReader(calibration_images, input_name, imgsz)
    qformat = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator
    detect_nodes, conv_count, detect_index = find_detect_head_conv_nodes(fp32_path)
    available_modules = {
        int(match.group(1)) for node in graph.node if node.op_type == 'Conv'
        and (match := MODEL_NODE_PATTERN.match(node.name))
    }
    if missing := set(fp32_modules) - available_modules:
        raise ValueError(f'FP32 modules not found in this model: {sorted(missing)}')
    excluded_nodes = [] if quantize_detect_head else list(detect_nodes)
    excluded_nodes += [
        node.name for node in graph.node if node.op_type == 'Conv'
        and (match := MODEL_NODE_PATTERN.match(node.name))
        and int(match.group(1)) in fp32_modules
        and node.name not in excluded_nodes
    ]
    activation_qtype = QuantType.QUInt8 if activation_type == "u8" else QuantType.QInt8

    head_strategy = (
        f"keeping model.{detect_index} Detect head and modules {fp32_modules} in FP32"
        if not quantize_detect_head
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
        calibration_providers=calibration_providers or ["CPUExecutionProvider"],
    )
    add_quantization_metadata(int8_path, [] if quantize_detect_head else detect_nodes, activation_type)
    model = onnx.load(str(int8_path))
    for key, value in {
        'quantization.fp32_modules': json.dumps(fp32_modules),
        'quantization.quantized_conv_count': str(conv_count - len(excluded_nodes)),
        'quantization.calibration_method': 'MinMax',
        'quantization.calibration_images': str(len(calibration_images)),
    }.items():
        item = model.metadata_props.add()
        item.key, item.value = key, value
    onnx.save(model, str(int8_path))


def quantize_compact_weights(fp32_path: Path, int8_path: Path, *, fold_on_load: bool = False) -> int:
    """Store Conv weights as grouped INT8, with FP32 dequantization before Conv."""
    model = onnx.load(str(fp32_path))
    graph = model.graph
    input_names = {value.name for value in graph.input}
    weight_names = {node.input[1] for node in graph.node if node.op_type == 'Conv'} - input_names
    initializers = [tensor for tensor in graph.initializer if tensor.name in weight_names
                    and tensor.data_type == onnx.TensorProto.FLOAT]
    if not initializers and not fold_on_load:
        raise RuntimeError('No FP32 Conv weights found for compact quantization')

    tensor_names = {tensor.name for tensor in graph.initializer} | input_names
    tensor_names.update(name for node in graph.node for name in node.output)
    dequant_nodes = []
    for index, tensor in enumerate(initializers):
        weight = onnx.numpy_helper.to_array(tensor)
        channels = weight.reshape(weight.shape[0], -1)
        groups = channels
        if channels.shape[1] % COMPACT_WEIGHT_GROUP_SIZE == 0:
            groups = channels.reshape(-1, COMPACT_WEIGHT_GROUP_SIZE)
        max_abs = np.max(np.abs(groups), axis=1)
        scale = np.where(max_abs == 0, np.float32(1), max_abs / np.float32(127))
        quantized = np.clip(np.rint(groups / scale[:, None]), -127, 127).astype(np.int8)

        prefix = f'compact_weight_{index}'
        while any(f'{prefix}_{suffix}' in tensor_names for suffix in ('int8', 'scale', 'flat', 'shape', 'float')):
            prefix += '_'
        quant_name, scale_name, flat_name, shape_name = (
            f'{prefix}_{suffix}' for suffix in ('int8', 'scale', 'flat', 'shape')
        )
        graph.initializer.extend([
            onnx.numpy_helper.from_array(quantized, quant_name),
            onnx.numpy_helper.from_array(scale[:, None] if fold_on_load else scale, scale_name),
            onnx.numpy_helper.from_array(np.asarray(weight.shape, dtype=np.int64), shape_name),
        ])
        if fold_on_load:
            # Constant Cast/Mul can fold while activation Q/DQ remains available for integer kernels.
            float_name = f'{prefix}_float'
            dequant_nodes.extend([
                onnx.helper.make_node('Cast', [quant_name], [float_name], to=onnx.TensorProto.FLOAT),
                onnx.helper.make_node('Mul', [float_name, scale_name], [flat_name]),
            ])
        else:
            dequant_nodes.append(onnx.helper.make_node('DequantizeLinear', [quant_name, scale_name], [flat_name], axis=0))
        dequant_nodes.append(onnx.helper.make_node('Reshape', [flat_name, shape_name], [tensor.name]))

    quantized_names = {tensor.name for tensor in initializers}
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name in quantized_names:
            del graph.initializer[index]
    dequant_nodes.extend(graph.node)
    del graph.node[:]
    graph.node.extend(dequant_nodes)
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update({
        'quantization.strategy': 'compact-integer-conv' if fold_on_load else 'compact-weight-only',
        'quantization.weight_type': 's8',
        'quantization.weight_group_size': str(COMPACT_WEIGHT_GROUP_SIZE),
        'quantization.activation_type': metadata.get('quantization.activation_type', 'fp32') if fold_on_load else 'fp32',
        'quantization.quantized_weight_count': str(len(initializers)),
    })
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, str(int8_path))
    return len(initializers)


def extract_scores(output: np.ndarray) -> np.ndarray:
    """Extract confidence scores from YOLO26 end-to-end or traditional YOLO output."""
    if output.ndim != 3:
        raise RuntimeError(f"Unexpected YOLO output shape: {output.shape}")
    if output.shape[-1] == 6:  # YOLO26 end-to-end: [x1, y1, x2, y2, confidence, class]
        return output[..., 4]
    if output.shape[1] >= 5:  # Traditional YOLO: [x, y, w, h, class scores...] x anchors
        return output[:, 4:, :]
    raise RuntimeError(f"Could not identify confidence scores in output shape {output.shape}")


def inference_session(path: Path, providers: list[str] | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=providers or ['CPUExecutionProvider'])


def smoke_test(onnx_path: Path, image_path: Path, imgsz: int) -> tuple[float, int, tuple[int, ...]]:
    session = inference_session(onnx_path)
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
    providers: list[str] | None = None,
) -> None:
    """Reject YOLO26 INT8 models whose final detections diverge sharply from FP32."""
    sessions = [
        inference_session(path, providers) for path in (fp32_path, int8_path)
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


def check_map_drop(
    reference: dict, candidate: dict, max_drop: float, max_class_drop: float, *, loss_only: bool = False,
) -> dict:
    """Check mAP50-95, with optional symmetric parity and AP50 limits."""
    if not reference['classes']:
        raise RuntimeError('No labeled classes found for accuracy validation')
    if reference['classes'].keys() != candidate['classes'].keys():
        missing = sorted(reference['classes'].keys() - candidate['classes'].keys())
        extra = sorted(candidate['classes'].keys() - reference['classes'].keys())
        raise RuntimeError(f'Validation class mismatch (missing={missing}, extra={extra})')
    for result in (reference, candidate):
        values = [result['map50_95']]
        values.extend(value for metrics in result['classes'].values() for value in metrics.values())
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError('Non-finite mAP in accuracy validation')
    drops = {'overall': reference['map50_95'] - candidate['map50_95'], 'classes': {}}
    failures = []
    gap_name = 'loss' if loss_only else 'difference'
    overall_gap = drops['overall'] if loss_only else abs(drops['overall'])
    if overall_gap > max_drop:
        failures.append(f"overall mAP50-95 {gap_name} {overall_gap:.5f} > {max_drop:.5f}")
    for name, metrics in reference['classes'].items():
        other = candidate['classes'][name]
        delta = {key: metrics[key] - other[key] for key in ('map50', 'map50_95')}
        drops['classes'][name] = delta
        class_gap = delta['map50_95'] if loss_only else abs(delta['map50_95'])
        if class_gap > max_class_drop:
            failures.append(f"{name} mAP50-95 {gap_name} {class_gap:.5f} > {max_class_drop:.5f}")
        if not loss_only and delta['map50'] > max_class_drop:
            failures.append(f"{name} AP50 drop {delta['map50']:.5f} > {max_class_drop:.5f}")
    if failures:
        raise RuntimeError('Labeled INT8 accuracy gate failed: ' + '; '.join(failures))
    return drops


def validate_map(path: Path, data_yaml: Path, split: str, imgsz: int, device: str, project: Path) -> dict:
    model = YOLO(str(path), task='detect')
    metrics = model.val(
        data=str(data_yaml.resolve()), split=split, imgsz=imgsz,
        device=device, workers=4, plots=False, **VALIDATION_SETTINGS,
        project=str(project), name=path.stem, exist_ok=True,
    )
    return {
        'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
        'classes': {
            metrics.names[int(cls)]: {
                'map50': float(metrics.box.ap50[index]),
                'map50_95': float(metrics.box.ap[index]),
            }
            for index, cls in enumerate(metrics.box.ap_class_index)
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def profile_integer_convolutions(model: Path, image_path: Path, imgsz: int, providers: list) -> dict:
    """Check that ONNX Runtime actually runs the model's integer convolution kernels."""
    graph = onnx.load(str(model), load_external_data=False).graph
    qlinear_convs = sum(node.op_type == 'QLinearConv' for node in graph.node)
    if not qlinear_convs:
        raise RuntimeError('Integer ONNX contains no QLinearConv nodes')
    image = preprocess_image(image_path, imgsz)
    if image is None:
        raise RuntimeError(f'Could not read integer profile image: {image_path}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_profiling = True
    options.profile_file_prefix = str(model.parent / 'integer_conv_profile')
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    session.run(None, {session.get_inputs()[0].name: image})
    events = json.loads(Path(session.end_profiling()).read_text())
    counts = Counter(event['args']['provider'] for event in events if
                     event.get('cat') == 'Node' and event.get('name', '').endswith('_kernel_time') and
                     event.get('args', {}).get('op_name') == 'QLinearConv')
    if sum(counts.values()) != qlinear_convs:
        raise RuntimeError(f"ORT executed {sum(counts.values())}/{qlinear_convs} QLinearConv kernels")
    print(f"Profile confirmed {qlinear_convs} integer convolution kernel calls: {dict(counts)}")
    return {'providers': session.get_providers(), 'ort_version': ort.__version__,
            'qlinear_conv_calls': qlinear_convs, 'qlinear_conv_calls_by_provider': dict(counts),
            'fp32_convs_in_graph': sum(node.op_type == 'Conv' for node in graph.node)}


def load_pt_reference(path: Path, report: dict) -> dict:
    reference = json.loads(path.read_text())
    keys = ('checkpoint_sha256', 'data_yaml_sha256', 'imgsz', 'validation_split',
            'validation_images', 'validation_manifest_sha256', 'accuracy_reference', *VALIDATION_SETTINGS)
    mismatches = [key for key in keys if reference.get(key) != report[key]]
    if reference.get('accuracy_verified') is not True or mismatches:
        raise ValueError(f'PT reference report is unverified or incompatible: {mismatches}')
    print(f'Using previously validated PT metrics from {path}')
    return reference['pt_metrics']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact INT8-weight ONNX under 30 MB with labeled accuracy validation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Input .pt model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Ultralytics dataset YAML for validation and calibration.")
    parser.add_argument("--verify-split", default="val", help="Independent split for quality checks.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Export/calibration image size. Defaults to the training size stored in a .pt checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", default=None, help="Output INT8 ONNX path.")
    calibrated = parser.add_argument_group("Calibrated weight and activation quantization")
    modes = calibrated.add_mutually_exclusive_group()
    modes.add_argument("--weight-activation", action="store_true",
                       help="Use the previous mixed-precision W8A8 export instead of the compact profile.")
    modes.add_argument('--integer-conv', action='store_true',
                       help='Export QLinearConv with compact weights and verify actual ORT integer execution.')
    calibrated.add_argument("--split", help="Calibration split. Defaults to train.")
    calibrated.add_argument("--calib-images", type=int, help="Calibration sample count. Defaults to 512.")
    calibrated.add_argument("--quant-format", choices=("qdq", "qoperator"), help="Defaults to qdq.")
    calibrated.add_argument(
        "--activation-type",
        choices=("u8", "s8"),
        help="Activation quantization type. Defaults to u8.",
    )
    calibrated.add_argument(
        "--quantize-detect-head",
        action="store_true",
        help="Also quantize the Detect head. Not recommended for YOLO26 end-to-end models.",
    )
    calibrated.add_argument('--fp32-modules', type=int, nargs='*',
                            help='Sensitive module indices kept in FP32, in addition to the Detect head.')
    parser.add_argument("--verify-images", type=int, default=DEFAULT_VERIFY_IMAGES)
    parser.add_argument("--verify-conf", type=float, default=DEFAULT_VERIFY_CONF)
    parser.add_argument("--min-match-recall", type=float, default=DEFAULT_MIN_MATCH_RECALL)
    parser.add_argument('--device', default='auto', help='Accuracy validation device: auto, cpu, or GPU index.')
    parser.add_argument('--max-map-drop', type=float,
                        help='Overall mAP50-95 limit (default 0.003 absolute, or 0.01 loss in integer mode).')
    parser.add_argument('--max-class-map-drop', type=float,
                        help='Per-class limit (default 0.005 absolute/AP50 loss, or 0.01 mAP50-95 loss in integer mode).')
    parser.add_argument('--reference-report', type=Path,
                        help='Reuse PT metrics from a verified report with matching checkpoint, data and validation settings.')
    parser.add_argument('--skip-map-check', action='store_true',
                        help='Skip full labeled validation; report marks the export as accuracy-unverified.')
    parser.add_argument("--force-fp32-export", action="store_true", help="Re-export the FP32 ONNX even if it exists.")
    parser.add_argument("--no-simplify", action="store_true", help="Disable FP32 ONNX simplification.")
    args = parser.parse_args()
    # Existing calibration options continue to select the calibrated export profile.
    if args.integer_conv and args.quant_format not in {None, 'qoperator'}:
        parser.error('--integer-conv requires --quant-format qoperator')
    calibration_defaults = {
        'split': 'train', 'calib_images': DEFAULT_CALIBRATION_IMAGES,
        'quant_format': 'qoperator' if args.integer_conv else 'qdq', 'activation_type': 'u8',
        'fp32_modules': list(INTEGER_FP32_MODULES if args.integer_conv else DEFAULT_FP32_MODULES),
    }
    if not args.integer_conv:
        args.weight_activation |= args.quantize_detect_head or any(
            getattr(args, name) is not None for name in calibration_defaults
        )
    for name, default in calibration_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.max_map_drop is None:
        args.max_map_drop = .01 if args.integer_conv else .003
    if args.max_class_map_drop is None:
        args.max_class_map_drop = .01 if args.integer_conv else .005
    return args


def main() -> None:
    args = parse_args()
    calibrate = args.weight_activation or args.integer_conv
    if args.calib_images < 1:
        raise ValueError("--calib-images must be at least 1")
    if args.verify_images < 1:
        raise ValueError("--verify-images must be at least 1")
    if not 0.0 < args.verify_conf < 1.0:
        raise ValueError("--verify-conf must be between 0 and 1")
    if not 0.0 <= args.min_match_recall <= 1.0:
        raise ValueError("--min-match-recall must be between 0 and 1")
    if not 0 <= args.max_map_drop <= 1 or not 0 <= args.max_class_map_drop <= 1:
        raise ValueError('mAP drop limits must be between 0 and 1')
    if args.integer_conv and (args.skip_map_check or args.max_class_map_drop > .01):
        raise ValueError('--integer-conv requires full validation and a per-class loss limit of at most 0.01')
    if calibrate and args.split == args.verify_split:
        raise ValueError('Calibration and verification splits must differ; use --split train --verify-split val')
    model_path = Path(args.model)
    if model_path.suffix.lower() != '.pt':
        raise ValueError('--model must be a .pt checkpoint')
    checkpoint_sha256 = sha256(model_path)
    data_yaml = Path(args.data)
    cuda_available = torch.cuda.is_available()
    device = ('0' if cuda_available and not args.integer_conv else 'cpu') if args.device == 'auto' else args.device
    providers = ['CPUExecutionProvider']
    if device != 'cpu' and cuda_available and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    imgsz = resolve_export_imgsz(model_path, args.imgsz)
    fp32_path = export_fp32_onnx(
        model_path,
        imgsz=imgsz,
        simplify=not args.no_simplify,
        force=args.force_fp32_export,
    )
    output_suffix = '_int8.onnx' if args.weight_activation else '_int8_compact.onnx'
    if args.integer_conv:
        output_suffix = '_int8_conv.onnx'
    int8_path = Path(args.output) if args.output else fp32_path.with_name(fp32_path.stem + output_suffix)
    if int8_path.resolve() in {model_path.resolve(), fp32_path.resolve()}:
        raise ValueError('INT8 output must differ from the source checkpoint and FP32 ONNX')
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_images = []
    if calibrate:
        calibration_images = choose_calibration_images(data_yaml, args.split, args.calib_images, args.seed)
        print(f"Using {len(calibration_images)} calibration images from {data_yaml} split={args.split!r}")
    verification_images = choose_verification_images(
        data_yaml=data_yaml,
        split=args.verify_split,
        calibration_images=calibration_images,
        count=args.verify_images,
        seed=args.seed,
    )
    print(f"Using {len(verification_images)} independent images for the INT8 quality gate")
    validation_paths = resolve_split_paths(data_yaml, args.verify_split)
    if set(calibration_images).intersection(validation_paths):
        raise ValueError('Calibration and validation contain overlapping image paths')
    calibration_providers = ['CPUExecutionProvider'] if args.integer_conv else providers
    inference_providers = providers
    if args.integer_conv and 'CUDAExecutionProvider' in providers:
        inference_providers = [('CUDAExecutionProvider', {'device_id': int(device.removeprefix('cuda:'))}),
                               'CPUExecutionProvider']
    print(f"Validation device: {device}; ONNX providers: {', '.join(providers)}")
    report = {
        'checkpoint': str(model_path.resolve()), 'checkpoint_sha256': checkpoint_sha256,
        'fp32_sha256': sha256(fp32_path), 'imgsz': imgsz, **VALIDATION_SETTINGS,
        'data_yaml': str(data_yaml.resolve()), 'data_yaml_sha256': sha256(data_yaml),
        'validation_split': args.verify_split, 'validation_images': len(validation_paths),
        'validation_manifest_sha256': hashlib.sha256('\n'.join(map(str, validation_paths)).encode()).hexdigest(),
        'weight_type': 's8', 'accuracy_verified': False, 'device': device, 'onnx_providers': providers,
        'accuracy_reference': 'pt',
        'max_map_drop': args.max_map_drop, 'max_class_map_drop': args.max_class_map_drop,
    }

    try:
        with tempfile.TemporaryDirectory(prefix='.int8_export_', dir=int8_path.parent) as workdir:
            project = Path(workdir)
            candidate_path = project / int8_path.name
            if calibrate:
                report.update(
                    quantization_strategy='weight-activation', activation_type=args.activation_type,
                    quant_format=args.quant_format, quantize_detect_head=args.quantize_detect_head,
                    fp32_modules=args.fp32_modules, calibration_method='MinMax', per_channel=True,
                    calibration_split=args.split, calibration_providers=calibration_providers,
                    calibration_images=[str(p) for p in calibration_images],
                )
                calibrated_path = project / 'calibrated.onnx' if args.integer_conv else candidate_path
                quantize_conv_only(
                    fp32_path, calibrated_path, calibration_images, imgsz,
                    args.quant_format, args.activation_type, args.quantize_detect_head,
                    fp32_modules=tuple(args.fp32_modules), calibration_providers=calibration_providers,
                )
                if args.integer_conv:
                    report.update(quantization_strategy='compact-integer-conv',
                                  weight_group_size=COMPACT_WEIGHT_GROUP_SIZE, max_model_bytes=COMPACT_MAX_BYTES)
                    report['compact_fp32_conv_weight_count'] = quantize_compact_weights(
                        calibrated_path, candidate_path, fold_on_load=True,
                    )
            else:
                report.update(
                    quantization_strategy='compact-weight-only', activation_type='fp32',
                    weight_group_size=COMPACT_WEIGHT_GROUP_SIZE, max_model_bytes=COMPACT_MAX_BYTES,
                )
                report['quantized_weight_count'] = quantize_compact_weights(fp32_path, candidate_path)
            report['model_bytes'] = candidate_path.stat().st_size
            if not args.weight_activation:
                print(f"Compact ONNX size: {report['model_bytes']:,} bytes")
                if report['model_bytes'] >= COMPACT_MAX_BYTES:
                    raise RuntimeError(f"Compact ONNX must be smaller than {COMPACT_MAX_BYTES:,} bytes")

            if args.integer_conv:
                report['integer_runtime_profile'] = profile_integer_convolutions(
                    candidate_path, verification_images[0], imgsz, inference_providers,
                )
            score_max, nonzero_scores, output_shape = smoke_test(candidate_path, verification_images[0], imgsz)
            print(f"Smoke test output_shape={output_shape} score_max={score_max:.6g} nonzero_scores={nonzero_scores}")
            if score_max <= 0.0 or nonzero_scores == 0:
                raise RuntimeError("INT8 export produced zero class scores. Do not use this ONNX file.")
            verify_end2end_quality(
                fp32_path=fp32_path,
                int8_path=candidate_path,
                image_paths=verification_images,
                imgsz=imgsz,
                conf=args.verify_conf,
                min_match_recall=args.min_match_recall,
                providers=inference_providers,
            )
            if not args.skip_map_check:
                if args.reference_report:
                    reference = load_pt_reference(args.reference_report, report)
                    report['pt_reference_report'] = str(args.reference_report.resolve())
                    report['pt_reference_report_sha256'] = sha256(args.reference_report)
                else:
                    reference = validate_map(model_path, data_yaml, args.verify_split, imgsz, device, project)
                report['pt_metrics'] = reference
                candidate = validate_map(candidate_path, data_yaml, args.verify_split, imgsz, device, project)
                report['int8_metrics'] = candidate
                report['map_drop'] = check_map_drop(reference, candidate, args.max_map_drop, args.max_class_map_drop,
                                                    loss_only=args.integer_conv)
                report['accuracy_verified'] = True
            if sha256(model_path) != checkpoint_sha256:
                raise RuntimeError('Checkpoint changed during export; rerun with a stable checkpoint')
            report['int8_sha256'] = sha256(candidate_path)
            candidate_path.replace(int8_path)
    except Exception as error:
        report['error'] = str(error)
        int8_path.with_suffix('.rejected.accuracy.json').write_text(json.dumps(report, indent=2))
        raise
    int8_path.with_suffix('.accuracy.json').write_text(json.dumps(report, indent=2))

    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    int8_size = int8_path.stat().st_size / 1024 / 1024
    print(f"FP32 ONNX: {fp32_path} ({fp32_size:.1f} MiB)")
    print(f"INT8 ONNX: {int8_path} ({int8_size:.1f} MiB)")


if __name__ == "__main__":
    main()
