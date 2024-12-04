import glob
import cv2
import torch
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load a model
model = YOLO("best.pt")
# model = YOLO("yolov8x-oiv7.pt")

bcs = glob.glob("/Users/robin/Downloads/counting_eval/*")
print(bcs)

def detect(filename):
    if not os.path.isfile(filename):
        return
    img = cv2.imread(filename)  # predict on an image
    if img is None:
        print("Error:", filename)
        return
    results = model(img, imgsz=(1024, 1024), conf=0.45)
    for res in results:
        annotator = Annotator(img)
        boxes = res.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            score = box.conf.item()
            if score > 0.1:
                annotator.box_label(b, "       " + model.names[int(c)])
                annotator.box_label(b, str(int(score*100)))

    img = annotator.result()
    cv2.imshow('Detection', img)
    cv2.waitKey(0)

for file in bcs:
    detect(file)
