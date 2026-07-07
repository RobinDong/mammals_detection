from ultralytics import YOLO


best_model = YOLO("yolov8m_800_20240202_int8.onnx")

val_metrics = best_model.val(
    data="robin_old.yaml",
    split="val",
    imgsz=640,
    batch=1,
    device="cpu",
    plots=True,
)

print("VAL mAP50-95:", val_metrics.box.map)
print("VAL mAP50:", val_metrics.box.map50)
print("VAL mAP75:", val_metrics.box.map75)
print("VAL per-class mAP50-95:", val_metrics.box.maps)
