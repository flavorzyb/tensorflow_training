# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

session = tf.Session()
iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

batch_size = 20

x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_out = tf.subtract(x1_data, my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_out, labels=y_target)

my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

init = tf.initialize_all_variables()
session.run(init)

print len(iris_2d)

for i in range(1000):
    rang_index = np.random.choice(len(iris_2d), size=batch_size)
    range_x = iris_2d[rang_index]
    range_x1 = np.array([[x[0]] for x in range_x])
    range_x2 = np.array([[x[1]] for x in range_x])
    range_y = np.array([[y] for y in binary_target[rang_index]])
    session.run(train_step, feed_dict={x1_data: range_x1, x2_data: range_x2, y_target: range_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(session.run(A)) + ', b = ' + str(session.run(b)))

# Visualize Results
# Pull out slope/intercept
[[slope]] = session.run(A)
[[intercept]] = session.run(b)

# Create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []

for i in x:
    ablineValues.append(slope*i+intercept)

# Plot the fitted line over the data
setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()