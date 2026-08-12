from pathlib import Path

import onnx
from ultralytics import YOLO


MODEL_PATH = Path("runs/detect/yolo26l_scratch_768/weights/best_int8.onnx")


def resolve_validation_imgsz(model_path: Path, model: YOLO) -> int:
    """Match validation to a static ONNX input, or use the PT checkpoint's training size."""
    if model_path.suffix.lower() != ".onnx":
        return int(model.overrides.get("imgsz", 640))

    graph = onnx.load(str(model_path), load_external_data=False).graph
    if not graph.input:
        raise RuntimeError(f"ONNX model has no inputs: {model_path}")
    dims = graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4 or not dims[-2].dim_value or not dims[-1].dim_value:
        raise RuntimeError(f"Expected a static NCHW ONNX input: {model_path}")

    height, width = int(dims[-2].dim_value), int(dims[-1].dim_value)
    if height != width:
        raise RuntimeError(f"Ultralytics validation requires a square static input, got {height}x{width}")
    print(f"Using ONNX input image size: {height}")
    return height


best_model = YOLO(str(MODEL_PATH))
validation_imgsz = resolve_validation_imgsz(MODEL_PATH, best_model)

val_metrics = best_model.val(
    data="robin_extra.yaml",
    split="val",
    imgsz=validation_imgsz,
    batch=1,
    device=0,
    plots=True,
)

print("VAL mAP50-95:", val_metrics.box.map)
print("VAL mAP50:", val_metrics.box.map50)
print("VAL mAP75:", val_metrics.box.map75)
print("VAL per-class mAP50-95:", val_metrics.box.maps)
