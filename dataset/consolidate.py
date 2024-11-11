import os
import glob
import yaml
import subprocess
import pandas as pd
import pyarrow.csv as pv

from tabulate import tabulate
from functools import partial
from multiprocessing import Pool
from collections import defaultdict


NR_THREAD = 8
COCO_PATH = "/media/data2/sanbai/yolo_dataset/clear_coco"
COCO_DIR = "robin"
REPORT_WIDTH = 6


def copy_coco() -> bool:
    command = f"cp -r {COCO_PATH} {COCO_DIR}"
    result = subprocess.run(command, shell=True, text=True)
    return result.returncode == 0


def clear_coco_process(pair):
    label_file, img_file = pair
    filename = COCO_DIR + "/" + label_file[2:]  # skip "./" at head
    if not os.path.exists(filename):
        return False
    file_lines = []
    with open(filename, "r") as fp:
        for line in fp:
            cat = int(line.split()[0])
            if cat == 0 or 14 <= cat <= 23:  # person, or from bird to giraffe
                if cat == 0:  # person -> 0
                    new_cat = 0
                elif cat == 14:  # bird -> 1
                    new_cat = 1
                else:
                    new_cat = 2  # mammals -> 2
                new_line = str(new_cat) + " " + " ".join(line.split(" ")[1:])
                file_lines.append(new_line)

    if not file_lines:
        print("remove:", label_file[2:])
        try:
            os.remove(COCO_DIR + "/" + label_file[2:])
            os.remove(COCO_DIR + "/" + img_file[2:])
        except OSError:
            pass

    try:
        # remove old label file because we will create a new one
        os.remove(filename)
    except OSError:
        pass

    if file_lines:
        with open(filename, "w") as fp:
            for line in file_lines:
                fp.write(line.strip() + "\n")
    return True


def clear_coco():
    for index_file in ["train2017.txt", "val2017.txt"]:
        lines = []
        with open(COCO_DIR + "/" + index_file, "r") as fp:
            pair_list = []
            for line in fp:
                line = line.strip()
                line = line.split("/")
                line[1] = "labels"
                line = "/".join(line)
                label_file = line.replace(".jpg", ".txt")
                pair_list.append((label_file, line))

            with Pool(NR_THREAD) as pool:
                array = pool.map(clear_coco_process, pair_list)
                for val, pair in zip(array, pair_list):
                    if val:  # if True, append img_file to lines
                        lines.append(pair[1])

        with open(COCO_DIR + "/" + index_file + ".new", "w") as fp:
            for line in lines:
                fp.write(line + "\n")


def extract_hero_process(label_file, tag_map, prefix):
    # Re-tag the label file
    with open(label_file, "r") as fp:
        new_lines = []
        for line in fp:
            cols = line.split()
            new_cat = tag_map[cols[0]]
            new_lines.append(new_cat + " " + " ".join(cols[1:]))

    if not new_lines:
        return

    with open(
        COCO_DIR + f"/labels/{prefix}/" + os.path.basename(label_file), "w"
    ) as fp:
        for line in new_lines:
            fp.write(line + "\n")

    # Copy the image file
    img_file = label_file[:-3] + "jpg"
    command = f"cp {img_file} {COCO_DIR}/images/{prefix}/{os.path.basename(img_file)}"
    subprocess.run(command, shell=True, text=True)


def extract_hero(pathes, tag_map):
    for path in pathes:
        all_files = set(glob.glob(path + "/*.txt"))
        val_files = set(glob.glob(path + "/*5.txt"))
        train_files = all_files - val_files

        for prefix, file_set in [("train2017", train_files), ("val2017", val_files)]:
            with Pool(NR_THREAD) as pool:
                pool.map(
                    partial(extract_hero_process, tag_map=tag_map, prefix=prefix),
                    file_set,
                )


def extract_open_image(data_source, prefix, coco_prefix):
    path = data_source["path"]
    img_lst = glob.glob(path + f"/{prefix}/data/*.jpg")
    print(f"Number of images for {prefix}: {len(img_lst)}")
    img_set = {os.path.basename(fname)[:-4] for fname in img_lst}  # without ".jpg"

    df = pv.read_csv(path + f"/{prefix}/labels/detections.csv").to_pandas()
    df = df[df["ImageID"].isin(img_set)]

    nr_cat = defaultdict(int)  # How many objects for each category
    tag_map = data_source["tag_map"]
    id_coord_map = defaultdict(list)
    for _, row in df.iterrows():
        label = row["LabelName"]
        if label not in tag_map:
            continue
        x_center = (float(row["XMin"]) + float(row["XMax"])) / 2
        y_center = (float(row["YMin"]) + float(row["YMax"])) / 2
        width = float(row["XMax"]) - float(row["XMin"])
        height = float(row["YMax"]) - float(row["YMin"])
        id_coord_map[row["ImageID"]].append(
            f"{tag_map[label]} {x_center} {y_center} {width} {height}"
        )
        nr_cat[label] += 1

    # Load "Label -> Name"
    cat_name = {}
    ln = pd.read_csv(path + f"/{prefix}/metadata/classes.csv", header=None)
    for _, row in ln.iterrows():
        cat_name[row[0]] = row[1]

    for id, coordinates in id_coord_map.items():
        if not coordinates:
            continue
        # Copy image file
        img_file = path + f"/{prefix}/data/" + id + ".jpg"
        if not os.path.exists(img_file):
            continue
        command = f"cp {img_file} {COCO_DIR}/images/{coco_prefix}2017/"
        subprocess.run(command, shell=True, text=True)
        # Write label file
        label_file = f"{COCO_DIR}/labels/{coco_prefix}2017/" + id + ".txt"
        with open(label_file, "w") as fp:
            for coord in coordinates:
                fp.write(f"{coord}\n")

    # Statistics report
    arr = sorted(
        [[cat_name[name], nr] for name, nr in nr_cat.items()],
        key=lambda pair: pair[1],
        reverse=True,
    )
    data, buff = [], []
    index = 0
    for name, nr in arr:
        buff += [name, nr]
        index += 1
        if index >= REPORT_WIDTH:
            data.append(buff)
            buff = []
            index = 0

    print(
        tabulate(
            data, headers=["Category", "Number of Objs"] * REPORT_WIDTH, tablefmt="grid"
        )
    )


def build_dataset():
    with open("consolidate.yml", "r") as fp:
        obj = yaml.safe_load(fp)

    res = copy_coco()
    if not res:
        print("Failed to copy COCO!")
        return
    # By using clear_coco/, we don't need to do clear_coco() again
    # clear_coco()

    # build mapping of "src_tag" -> "dest_tag"
    tag_map = {str(cat["src_tag"]): str(cat["dest_tag"]) for cat in obj["hero_source"]}

    # Extract images/labels from dataset created by hero.jianmei
    for cat in obj["hero_source"]:
        extract_hero(cat["path"], tag_map)

    # Extract images/labels from part of open-image-v7
    for prefix, coco_prefix in [("train", "train"), ("validation", "val")]:
        extract_open_image(obj["open_image_source"], prefix, coco_prefix)


if __name__ == "__main__":
    build_dataset()
