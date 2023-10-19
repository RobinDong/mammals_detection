import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load a model
model = YOLO("best.pt")

# Use the model
for file in ["plage.jpg", "pb1.webp", "pb2.webp", "pb3.jpeg", "pb4.webp", "pb5.jpeg", "bp1.webp", "bp2.avif"]:
    img = cv2.imread(f"/Users/robin/Downloads/{file}")  # predict on an image

    results = model.predict(img)
    for res in results:
        annotator = Annotator(img)
        boxes = res.boxes
        for box in boxes:
            b = box.xyxy[0]
            print(b)
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()
    cv2.imshow('Detection', img)
    cv2.waitKey(0)
# path = model.export(format="onnx")  # export the model to ONNX format
