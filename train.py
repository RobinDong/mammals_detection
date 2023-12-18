from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco.yaml', epochs=600, batch=24, imgsz=800, resume=False)
