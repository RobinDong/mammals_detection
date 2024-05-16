from ultralytics import YOLO

model = YOLO('runs/detect/train15/weights/best.pt')
model.export(format="openvino", imgsz=800, **{"int8": True, "data": "coco.yaml"})
