# -*- coding: utf-8 -*-

from skimage import io, transform
from PIL import Image
import numpy as np
import os
import tensorflow as tf

project_root_path = os.path.abspath(os.path.dirname(__file__))
# flower_photos_path = os.path.join(project_root_path, "flower_photos")
flower_photos_path = os.path.join(project_root_path, "flower_tests")


def get_all_images_files(path):
    result = []
    labels = []
    labels_index = []
    index = 0
    for name in os.listdir(path):
        sub_path = os.path.join(path, name)
        if os.path.isdir(sub_path):
            for file_name in os.listdir(sub_path):
                file_path = os.path.join(sub_path, file_name)
                _, ext = os.path.splitext(file_path)
                if ext.lower() == ".jpg":
                    result.append(file_path)
                    labels.append(name)
                    labels_index.append(index)
                    index = index + 1

    return result, labels, labels_index


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


image_files, image_labels, image_label_index = get_all_images_files(flower_photos_path)
min_width, min_height = find_min_width_and_height(image_files)

width = 100
height = 100
c = 3

image_array = resize_images(image_files, width, height)
image_array = np.asarray(image_array, np.float32)
image_label_index = np.asarray(image_label_index, np.int32)

num_example = image_array.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)

image_array = image_array[arr]
image_label_index = image_label_index[arr]

ratio = 0.8
s = np.int(num_example * ratio)

x_train = image_array[:s]
y_train = image_label_index[:s]

x_val = image_array[s:]
y_val = image_label_index[s:]


x = tf.placeholder(tf.float32, shape=[None, width, height, c], name="x")
y = tf.placeholder(tf.float32, shape=[None, ], name="y")

