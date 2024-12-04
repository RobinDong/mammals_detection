import glob
import cv2
import torch

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load a model
model = YOLO("best.pt")
# model = YOLO("yolov8x-oiv7.pt")

bcs = glob.glob("/Users/robin/Downloads/bad_cases/*.JPG")
print(bcs)

def detect(filename):
    img = cv2.imread(filename)  # predict on an image
    '''orig_size = (img.shape[1], img.shape[0])
    print("orig_size:", orig_size)
    img = cv2.resize(img, (640, 640))
    img = cv2.resize(img, orig_size)'''
    '''height, width, _ = img.shape
    height = height - (height % 32)
    width = width - (width % 32)
    img = cv2.resize(img, (640, 640))'''

    #timg = torch.tensor(img, device="mps").permute(2, 0, 1).unsqueeze(0) / 255.0
    results = model(img, imgsz=(640, 640), conf=0.45)
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

detect("/Users/robin/Downloads/foxes.jpg")

for file in bcs:
    detect(file)

# Use the model
for file in ["3lobster.webp", "three.jpg", "shrimp.jpeg", "shrimp2.png", "lobster2.jpg", "lobster.webp", "toms.jpg", "bird.jpg", "owl.jpg", "tunas.jpeg", "two_salmon.webp", "hunt_salmon.jpeg", "yes.jpg", "yes2.jpg", "plate.jpg", "pb1.webp", "pb2.webp", "pb3.jpeg", "pb4.webp", "pb5.jpeg", "bp1.webp", "hu1.jpg", "hu2.jpg"]:
    filename = f"/Users/robin/Downloads/{file}"  # predict on an image
    detect(filename)
