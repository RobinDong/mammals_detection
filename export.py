from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')
model.export(format="onnx", int8=True)
