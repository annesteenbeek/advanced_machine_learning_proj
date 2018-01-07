#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from config import cfg
from caps_layers import conv_caps_layer, fully_connected_caps_layer


class CapsNet(object):

    def __init__(self):
        self.graph = tf.Graph()

        nwidth = cfg.nwidth
        nheight = cfg.nheight
        self.tf_images = tf.placeholder(tf.float32, [None, nwidth, nheight, 3], name='images')
        self.tf_labels = tf.placeholder(tf.int64, [None], name='labels')

        # Translate labels to one hot array
        one_hot_labels = tf.one_hot(self.tf_labels, depth=cfg.nb_labels)

        # Build the model
        self.caps1, self.caps2 = self.build_main_network(self.tf_images)
        self.decoder = self._build_decoder(self.caps2, one_hot_labels, cfg.batch_size)
        self._summary()

        # Build the loss
        _loss = self._build_loss(self.caps2,
                            one_hot_labels,
                            self.tf_labels,
                            self.decoder,
                            self.tf_images)
        (self.tf_loss_squared_rec,
        self.tf_margin_loss_sum,
        self.tf_predicted_class,
        self.tf_correct_prediction,
        self.tf_accuracy,
        self.tf_loss,
        self.tf_margin_loss,
        self.tf_reconstruction_loss) = _loss

        # setup the optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.tf_loss, global_step=self.global_step)
    
    def build_main_network(self, images_ph):
        '''Create the main network'''
        # First block:
        # Layer 1: Convolutional part
        shape = (cfg.conv_1_size, cfg.conv_1_size, 3, cfg.conv_1_nb)
        conv1 = self._conv_layer(images_ph,
                           shape,
                           relu=True,
                           max_pooling=False,
                           padding='VALID')

        # Create the first capsules layer
        caps1 = conv_caps_layer(
            input_layer = conv1,
            capsules_size=cfg.caps_1_vec_len,
            nb_filters=cfg.caps_1_nb_filter,
            kernel_size=cfg.caps_1_size
        )

        # Second capsules layer
        caps2 = fully_connected_caps_layer(
            input_layer=caps1,
            capsules_size=cfg.caps_2_vec_len,
            nb_capsules=cfg.nb_labels,
            iterations=cfg.routing_steps
        )

        return caps1, caps2

    def _conv_layer(self, prev, shape, padding='VALID', strides=[1, 1, 1, 1], relu=False,
                    max_pooling=False, mp_ksize=[1, 2, 2, 1], mp_strides=[1, 2, 2, 1]):
        # TODO: How is this convulutional layer build exactly...
        conv_w = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=0.1,seed=0))
        conv_b = tf.Variable(tf.zeros(shape[-1]))
        conv = tf.nn.conv2d(prev, conv_w, strides=strides, padding=padding) + conv_b

        #TODO: add relu or max_pooling

        return conv

    def _build_decoder(self, caps2, one_hot_labels, batch_size):
        """
            Build the decoder part from the last capsule layer
            **input:
                *Caps2:  Output of second Capsule layer
                *one_hot_labels
                *batch_size
        """
        labels = tf.reshape(one_hot_labels, (-1,cfg.nb_labels, 1))
        # squeeze(caps2):   [?, len_v_j,    capsules_nb]
        # labels:           [?, NB_LABELS,  1] with capsules_nb == NB_LABELS
        mask = tf.matmul(tf.squeeze(caps2), labels, transpose_a=True)
        # Select the good capsule vector
        capsule_vector = tf.reshape(mask, shape=(batch_size, cfg.caps_2_vec_len))
        # capsule_vector: [?, len_v_j]

        # Reconstruct image
        fc1 = tf.contrib.layers.fully_connected(capsule_vector, num_outputs=400)
        fc1 = tf.reshape(fc1, shape=(batch_size, 5, 5, 16))
        upsample1 = tf.image.resize_nearest_neighbor(fc1, (8, 8))
        conv1 = tf.layers.conv2d(upsample1, 4, (3,3), padding='same', activation=tf.nn.relu)

        upsample2 = tf.image.resize_nearest_neighbor(conv1, (16, 16))
        conv2 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)

        upsample3 = tf.image.resize_nearest_neighbor(conv2, (32, 32))
        conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)

        # 3 channel for RGG
        logits = tf.layers.conv2d(conv6, 3, (3,3), padding='same', activation=None)
        decoded = tf.nn.sigmoid(logits, name='decoded')
        tf.summary.image('reconstruction_img', decoded)

        return decoded

    def _build_loss(self, caps2, one_hot_labels, labels, decoded, images):
        """
            Build the loss of the graph
        """
        # Get the length of each capsule
        capsules_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True))

        max_l = tf.square(tf.maximum(0., 0.9 - capsules_length))
        max_l = tf.reshape(max_l, shape=(-1,cfg.nb_labels))
        max_r = tf.square(tf.maximum(0., capsules_length - 0.1))
        max_r = tf.reshape(max_r, shape=(-1,cfg.nb_labels))
        t_c = one_hot_labels
        m_loss = t_c * max_l + 0.5 * (1 - t_c) * max_r
        margin_loss_sum = tf.reduce_sum(m_loss, axis=1)
        margin_loss = tf.reduce_mean(margin_loss_sum)

        # Reconstruction loss
        loss_squared_rec = tf.square(decoded - images)
        reconstruction_loss = tf.reduce_mean(loss_squared_rec)

        # 3. Total loss
        loss = margin_loss + (0.0005 * reconstruction_loss)

        # Accuracy
        predicted_class = tf.argmax(capsules_length, axis=1)
        predicted_class = tf.reshape(predicted_class, [tf.shape(capsules_length)[0]])
        correct_prediction = tf.equal(predicted_class, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return (loss_squared_rec, margin_loss_sum, predicted_class, correct_prediction, accuracy,
                loss, margin_loss, reconstruction_loss)


    def _summary(self):
        '''Give a summary of progress'''
        pass
