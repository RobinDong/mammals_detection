import os

parent_dir = "coco/"


def extract_categories(label_file: str, img_file: str):
    filename = parent_dir + label_file[2:]
    print(f"{filename} check")
    if not os.path.exists(filename):
        return []
    file_lines = []
    with open(filename, "r") as fp:
        for line in fp:
            cat = int(line.split()[0])
            print("cat:", cat)
            if cat in {0, 200, 201, 202, 203, 204, 205, 206} or 14 <= cat <= 23:
                if cat == 0:  # person -> 0
                    new_cat = 0
                elif cat == 14 or cat == 200:  # bird -> 1
                    new_cat = 1
                elif cat == 202:  # reptile
                    new_cat = 3
                elif cat == 203:  # amphibia
                    new_cat = 4
                elif cat == 204:  # fish
                    new_cat = 5
                elif cat == 205:  # shrimp
                    new_cat = 6
                elif cat == 206:  # others
                    new_cat = 7
                else:
                    new_cat = 2  # mammals -> 2
                new_line = str(new_cat) + " " + " ".join(line.split(" ")[1:])
                file_lines.append(new_line)

    print(f"{filename} lines:", file_lines)
    if len(file_lines) <= 0:
        print("remove:", label_file[2:])
        try:
            os.remove(parent_dir + label_file[2:])
            os.remove(parent_dir + img_file[2:])
        except OSError:
            pass

    try:
        os.remove(filename)
    except OSError:
        pass
    if len(file_lines) > 0:
        with open(filename, "w") as fp:
            for line in file_lines:
                fp.write(line.strip() + "\n")
    return file_lines


for index_file in ["train2017.txt", "val2017.txt"]:
    lines = []
    with open(parent_dir + index_file, "r") as fp:
        for line in fp:
            line = line.strip()
            line = line.split("/")
            line[1] = "labels"
            line = "/".join(line)
            print("line:", line)
            label_file = line.replace(".jpg", ".txt")
            array = extract_categories(label_file, line)
            if array:
                lines.append(line)

    with open(parent_dir + index_file + ".new", "w") as fp:
        for line in lines:
            fp.write(line + "\n")
