from ultralytics import YOLO

# Load a model
model = YOLO('yolov8l.pt')
# model = YOLO("/media/data2/sanbai/mammals_detection/runs/detect/train11/weights/last.pt")

# Train the model
# results = model.train(data='coco.yaml', epochs=600, batch=20, imgsz=640, save_period=2, resume=True, cache=False, device=0)
results = model.train(data='coco.yaml', epochs=600, batch=16, imgsz=1024, save_period=2, resume=False, device=0, cache=False)

model.export(format="onnx", int8=True)
model.export(format="openvino", imgsz=640, **{"int8": True, "data": "coco.yaml"})
