import os
import sys
import cv2
import numpy as np

from ultralytics import YOLO

IMG_SIZE = 2048

def main(path):
    filename, file_extension = os.path.splitext(path)

    # Load segment model of yolov8
    model = YOLO("yolov8x-seg.pt")
    img = cv2.imread(path)

    results = model(img, imgsz=(img.shape[1], img.shape[0]))

    index = 0
    for res in results:
        for mask in res.masks.xy:
            polygan = mask.reshape((-1, 1, 2)).astype(np.int32)
            x, y, w, h = cv2.boundingRect(polygan)

            binary_mask = np.ones(img.shape, dtype=np.uint8) * 255
            # Fill the polygan with zero
            cv2.fillPoly(binary_mask, [polygan], (0, 0, 0))
            # Add zero polygan with origin image could keep object
            out_img = cv2.add(img, binary_mask)[y:y+h, x:x+w]

            cv2.imwrite(f"{filename}_{index}{file_extension}", out_img)
            index += 1
    print(f"Total: {index}")

if __name__ == "__main__":
    main(sys.argv[1])
