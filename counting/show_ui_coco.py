"""
Show images and bounding-boxes for COCO dataset
"""

import cv2
import glob

import matplotlib.pyplot as plt


if __name__ == "__main__":
    img_lst = glob.glob("/Users/robin/Downloads/bird_images/images/*/*.jpg")
    for iname in img_lst:
        lname = iname.replace("/images", "/labels").replace("jpg", "txt")
        print(lname)

        # Load image
        image = cv2.imread(iname)
        if image is None:
            print(f"Failed to read {iname}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        ax = plt.gca()
        with open(lname, "r") as fp:
            for line in fp:
                parts = line.strip().split()
                x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

                # Convert YOLO format to pixel coordinates
                x_center_pixel = x_center * width
                y_center_pixel = y_center * height
                bbox_width_pixel = bbox_width * width
                bbox_height_pixel = bbox_height * height

                x_min = int(x_center_pixel - bbox_width_pixel / 2)
                y_min = int(y_center_pixel - bbox_height_pixel / 2)
                x_max = int(x_center_pixel + bbox_width_pixel / 2)
                y_max = int(y_center_pixel + bbox_height_pixel / 2)

                # Draw the bounding box
                color = (0, 255, 0)  # Green color for bounding box
                print(x_min, y_min, x_max, y_max)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        plt.imshow(image)
        plt.axis("off")
        plt.show()
