# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np

sess = tf.Session()
x_shape = [3, 4, 4, 2]
x_val = np.random.uniform(size=x_shape)
print (x_val)
