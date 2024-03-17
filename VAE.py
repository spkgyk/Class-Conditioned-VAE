from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import six
import tensorflow as tf
import zhusuan as zs
from six.moves import cPickle as pickle

from examples import conf
from examples.utils import dataset, save_image_collections


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(y, x_dim, z_dim, n):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1)

    # Concatenate z and y
    z = tf.concat(axis=1, values=[z, y])

    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)

    x_mean = bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1, dtype=tf.float32)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, y, z_dim):
    bn = zs.BayesianNet()

    # Concatenate x and y
    x = tf.concat(axis=1, values=[x, y])

    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)

    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1)
    return bn


def main():
    data_path = os.path.join("data", "mnist.pkl.gz")
    x_train, y_train, _, _, _, _ = dataset.load_mnist_realval(data_path)
    x_train = np.random.binomial(1, x_train, size=x_train.shape)

    epochs = 30
    batch_size = 128
    iterations = x_train.shape[0] // batch_size
    z_dim = 40
    y_dim = y_train.shape[1]
    x_dim = x_train.shape[1]

    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_gen(y, x_dim, z_dim, n)
    variational = build_q_net(x, y, z_dim)

    lower_bound = zs.variational.elbo(model, {"x": x}, variational=variational)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    infer_op = optimizer.minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])

    one_hots = []
    for i in range(10):
        one_hot = np.zeros((100, 10))
        one_hot[:, i] = 1
        one_hots.append(one_hot)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            running_time = -time.time()
            lbs = []
            for t in range(iterations):
                x_batch = x_train[t*batch_size:(t+1)*batch_size]
                y_batch = y_train[t*batch_size:(t+1)*batch_size]
                lb = sess.run([infer_op, lower_bound], feed_dict={
                              x: x_batch, y: y_batch, n: batch_size})[1]
                lbs.append(lb)

            running_time += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch + 1, running_time, np.mean(lbs)))

            for i in range(10):
                images = sess.run(x_gen, feed_dict={y: one_hots[i], n: 100})
                name = os.path.join("results", 'cvae', str(
                    epoch + 1).zfill(3), "{}.png".format(i))
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
