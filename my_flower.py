# -*- coding: utf-8 -*-

from skimage import io, transform
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import time

project_root_path = os.path.abspath(os.path.dirname(__file__))
flower_photos_path = os.path.join(project_root_path, "flower_photos")
# flower_photos_path = os.path.join(project_root_path, "flower_tests")


def get_type_with_name(name):
    if name == "daisy":
        return 0
    elif name == "dandelion":
        return 1
    elif name == "roses":
        return 2
    elif name == "sunflowers":
        return 3
    elif name == "tulips":
        return 4
    else:
        return 5


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
                    labels.append(get_type_with_name(name))

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
c = 3

image_array = resize_images(image_files, width, height)
image_array = np.asarray(image_array, np.float32)
image_labels = np.asarray(image_labels, np.int32)

num_example = image_array.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)

image_array = image_array[arr]
image_labels = image_labels[arr]

ratio = 0.8
s = np.int(num_example * ratio)

x_train = image_array[:s]
y_train = image_labels[:s]

x_val = image_array[s:]
y_val = image_labels[s:]


x = tf.placeholder(tf.float32, shape=[None, width, height, c], name="x")
y = tf.placeholder(tf.int32, shape=[None, ], name="y")

# 第一个卷积层（100——>50)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(50->25)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第三个卷积层(25->12)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第四个卷积层(12->6)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

# 全连接层
dense1 = tf.layers.dense(inputs=re1,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits = tf.layers.dense(inputs=dense2,
                         units=5,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# ---------------------------网络结束---------------------------

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 10
batch_size = 64
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y: y_train_a})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y: y_val_a})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))

sess.close()
