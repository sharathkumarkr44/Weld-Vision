import os
from pascal_voc_writer import Writer
from xml.etree import ElementTree as ET


def read_txt(img_name, img):
    """read_txt
    Read the txt annotation file and extract the bounding boxes if exists.

    Args:
        img_file(str): the image file.
    Returns:
        rects(list): the bounding boxes of the image.
    """
    file_name = img_name.split('/')[-1]
    file_name = file_name.split('.')[0]
    img_width, img_height = img.size
    read_file_name = f"retrain\labels\{file_name}.txt"
    if not os.path.exists(read_file_name):
        read_file_name = f"predictions\{file_name}.txt"
    try:
        with open(read_file_name, "r") as file:
            lines = file.readlines()  # Read all lines from the file
    except FileNotFoundError:
        print(read_file_name)
        print("File not found. Please provide the correct file path.")

    rects = []

    for line in lines:
        annotation = line.split(' ')
        label = int(annotation[0])
        x = float(annotation[1])
        y = float(annotation[2])
        w = float(annotation[3])
        h = float(annotation[4])

        box_width = w * img_width
        box_height = h * img_height
        x_min = int(x * img_width - (box_width / 2))
        y_min = int(y * img_height - (box_height / 2))
        x_max = int(x * img_width + (box_width / 2))
        y_max = int(y * img_height + (box_height / 2))

        rects.append(
            {
                "left":  x_min,
                "top": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "label": label,
            }
        )
    return rects


def output_txt(img_file, img, rects):
    """output_txt
    Output the txt image annotation file

    Args:
        img_file(str): the image file.
        img(PIL.Image): the image object.
        rects(list): the bounding boxes of the image.
    """
    file_name = img_file.split(".")[0]
    annotations = []
    for rect in rects:
        label = rect["label"]
        left = rect["left"]
        top = rect["top"]
        width = rect["width"]
        height = rect["height"]

        x = (left + width / 2) / img.width
        y = (top + height / 2) / img.height
        w = width / img.width
        h = height / img.height

        annotation = f"{label} {x} {y} {w} {h}"
        annotations.append(annotation)
    # print(img_file)
    save_file_name1 = img_file.split('/')[-1]
    save_file_name = save_file_name1.split('.')[0]
    # print(save_file_name)
    filename = f"retrain\labels\{save_file_name}.txt"
    with open(filename, 'w') as file:
        for annotation in annotations:
            file.write(annotation + "\n")
