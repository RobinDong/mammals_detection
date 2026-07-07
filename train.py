from ultralytics import YOLO

# Load a model
model = YOLO('yolo26l.pt')
# model = YOLO("/media/data2/sanbai/mammals_detection/runs/detect/train11/weights/last.pt")
# model = YOLOv10.from_pretrained("jameslahm/yolov10m")
# model = YOLOv10("/media/data2/sanbai/yolov10/runs/detect/train3/weights/last.pt")

# Train the model
# results = model.train(data='coco.yaml', epochs=600, batch=20, imgsz=640, save_period=2, resume=True, cache=False, device=0)
results = model.train(data='robin_extra.yaml', epochs=100, batch=16, imgsz=640, save_period=2, resume=False, device=[0, 1], cache='disk', 
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.05,
    cutmix=0.0,
    close_mosaic=10,
multi_scale=True)

model.export(format="onnx", int8=True)
#model.export(format="openvino", imgsz=640, **{"int8": True, "data": "coco.yaml"})
