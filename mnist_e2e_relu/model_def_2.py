# This file based on : https://jmetzen.github.io/notebooks/vae.ipynb
# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, R0902

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf
from mnist_e2e import utils
import numpy as np

class Hparams(object):
    def __init__(self):
        self.n_hidden_recog_1 = 500  # 1st layer encoder neurons
        self.n_hidden_recog_2 = 500  # 2nd layer encoder neurons
        self.n_hidden_gener_1 = 500  # 1st layer decoder neurons
        self.n_hidden_gener_2 = 500  # 2nd layer decoder neurons
        self.n_input = 784           # MNIST data input (img shape: 28*28)
        self.n_z = 20                # dimensionality of latent space
        self.transfer_fct = tf.nn.relu


def encoder(hparams, x_ph, scope_name, reuse):
    with tf.compat.v1.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.compat.v1.get_variable('w1', initializer=utils.xavier_init(hparams.n_input, hparams.n_hidden_recog_1))
        b1 = tf.compat.v1.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_recog_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(x_ph, w1) + b1)

        w2 = tf.compat.v1.get_variable('w2', initializer=utils.xavier_init(hparams.n_hidden_recog_1, hparams.n_hidden_recog_2))
        b2 = tf.compat.v1.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_recog_2], dtype=tf.float32))
        hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)

        w3 = tf.compat.v1.get_variable('w3', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b3 = tf.compat.v1.get_variable('b3', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_mean = tf.matmul(hidden2, w3) + b3

        w4 = tf.compat.v1.get_variable('w4', initializer=utils.xavier_init(hparams.n_hidden_recog_2, hparams.n_z))
        b4 = tf.compat.v1.get_variable('b4', initializer=tf.zeros([hparams.n_z], dtype=tf.float32))
        z_log_sigma_sq = tf.matmul(hidden2, w4) + b4

    return z_mean, z_log_sigma_sq


def generator(hparams, z, scope_name, reuse):

    with tf.compat.v1.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.compat.v1.get_variable('w1', shape=[20, 500],initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(500)))
        b1 = tf.compat.v1.get_variable('b1', initializer=tf.zeros([hparams.n_hidden_gener_1], dtype=tf.float32))
        hidden1 = hparams.transfer_fct(tf.matmul(z, w1) + b1)

        w2 = tf.compat.v1.get_variable('w2', shape=[hparams.n_hidden_gener_1, hparams.n_hidden_gener_2],initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(500)) )
        b2 = tf.compat.v1.get_variable('b2', initializer=tf.zeros([hparams.n_hidden_gener_2], dtype=tf.float32))
        # hidden2 = hparams.transfer_fct(tf.matmul(hidden1, w2) + b2)
        hidden2  = hidden1
        w3 = tf.compat.v1.get_variable('w3', shape=[hparams.n_hidden_gener_2, hparams.n_input],initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(784)))
        b3 = tf.compat.v1.get_variable('b3', initializer=tf.zeros([hparams.n_input], dtype=tf.float32))
        logits = tf.matmul(hidden2, w3) + b3
        x_reconstr_mean = tf.nn.relu(logits)

    return logits, x_reconstr_mean


def get_loss(x, logits, z_mean, z_log_sigma_sq):
    reconstr_losses = tf.reduce_sum(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), axis=1)
    latent_losses = -0.5 * tf.reduce_sum(input_tensor=1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), axis=1)
    total_loss = tf.reduce_mean(input_tensor=reconstr_losses + latent_losses, name='total_loss')
    return total_loss


def get_z_var(hparams, batch_size):
    z = tf.Variable(tf.random.normal((batch_size, 20)), name='z')
    return z


def gen_restore_vars():
    restore_vars = ['gen/w1',
                    'gen/b1',
                    'gen/w2',
                    'gen/b2',
                    'gen/w3',
                    'gen/b3']
    return restore_vars
