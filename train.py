from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8m.pt')
model = YOLO("/media/data2/sanbai/mammals_detection/runs/detect/train11/weights/last.pt")
# model = YOLOv10.from_pretrained("jameslahm/yolov10m")
# model = YOLOv10("/media/data2/sanbai/yolov10/runs/detect/train3/weights/last.pt")

# Train the model
# results = model.train(data='coco.yaml', epochs=600, batch=20, imgsz=640, save_period=2, resume=True, cache=False, device=0)
results = model.train(data='coco.yaml', epochs=600, batch=8, imgsz=640, save_period=2, resume=True, device=0, cache=False, multi_scale=True)

model.export(format="onnx", int8=True)
model.export(format="openvino", imgsz=640, **{"int8": True, "data": "coco.yaml"})
