import os
import sys
import cv2
import numpy as np

from ultralytics import YOLO

IMG_SIZE = 1024
IOU_THRES = 0.5


def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def main(path):
    filename, file_extension = os.path.splitext(path)

    # Load segment model of yolov8
    model = YOLO("yolov8x-seg.pt")
    img = cv2.imread(path)

    results = model(img, imgsz=(img.shape[1], img.shape[0]))

    index = 0
    boxes = []
    for res in results:
        for mask in res.masks.xy:
            polygan = mask.reshape((-1, 1, 2)).astype(np.int32)
            x, y, w, h = cv2.boundingRect(polygan)
            # check overlap with other boxes
            curr = [x, y, x+w, y+h]
            overlap = False
            for box in boxes:
                iou = IOU(box, curr)
                if iou > IOU_THRES:
                    overlap = True
                    break
            if overlap:
                continue
            boxes.append(curr)

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
