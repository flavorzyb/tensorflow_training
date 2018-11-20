# -*- coding: utf-8 -*-

from skimage import io, transform
from PIL import Image
import numpy as np
import os

project_root_path = os.path.abspath(os.path.dirname(__file__))
flower_photos_path = os.path.join(project_root_path, "flower_photos")


def get_all_images_files(path):
    result = []
    labels = []
    for name in os.listdir(path):
        sub_path = os.path.join(path, name)
        if os.path.isdir(sub_path):
            for file_name in os.listdir(sub_path):
                file_path = os.path.join(sub_path, file_name)
                _, ext = os.path.splitext(file_path)
                if ext.lower() == ".jpg":
                    result.append(file_path)
                    labels.append(name)

    return result, labels


def find_min_width_and_height(files):
    width = 9999999
    height = 9999999
    for path in files:
        img = Image.open(path)
        if img.width < width:
            width = img.width

        if img.height < height:
            height = img.height

    return width, height


def resize_images(files, w, h):
    result = []
    for path in files:
        img = io.imread(path)
        img = transform.resize(img, (w, h))
        result.append(img)

    return result


image_files, image_labels = get_all_images_files(flower_photos_path)
min_width, min_height = find_min_width_and_height(image_files)

width = 100
height = 100

image_array = resize_images(image_files, width, height)

arr = np.asarray(image_array, np.float32)

print min_width, min_height
