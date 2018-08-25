from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()

if False:
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)  # also tf.float32 implicitly
    total = a + b
    print(a)
    print(b)
    print(total)

if False:
    vec = tf.random_uniform(shape=(3,))
    out1 = vec + 1
    out2 = vec + 2
    print(sess.run(vec))
    print(sess.run(vec))
    print(sess.run((out1, out2)))

if False:
    my_data = [
        [0, 1, ],
        [2, 3, ],
        [4, 5, ],
        [6, 7, ],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()

    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break

if False:
    r = tf.random_normal([10, 4])
    dataset = tf.data.Dataset.from_tensor_slices(r)
    iterator = dataset.make_initializable_iterator()
    next_row = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_row))
        except tf.errors.OutOfRangeError:
            break

if False:
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    print(sess.run(linear_model.kernel_initializer))
    y = linear_model(x)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, {x: [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ]}))

if True:
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_model = tf.layers.Dense(units=1)

    y_pred = linear_model(x)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        _, loss_value = sess.run((train, loss))
        # print(loss_value)

    print(sess.run(y_pred))
