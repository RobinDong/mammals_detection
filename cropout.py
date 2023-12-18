import os
import sys
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def main(path):
    filename, file_extension = os.path.splitext(path)

    # Load a model
    model = YOLO("best.pt")

    # Use the model
    img = cv2.imread(path)  # predict on an image

    results = model.predict(img)
    for res in results:
        boxes = res.boxes
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            c = box.cls

            roi = img[y1:y2, x1:x2]
            cv2.imwrite(f"{filename}_{index}{file_extension}", roi)

if __name__ == "__main__":
    main(sys.argv[1])
