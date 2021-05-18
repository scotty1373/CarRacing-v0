#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_probability as tfp
import gym

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.01


class ddpg_net():
    def __init__(self, shape_in, num_output):
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.actor_model = self.actor_net_builder()
        self.actor_history = tf.TensorArray(dtype=tf.float32, size=0,
                                            dynamic_size=True,
                                            clear_after_read=False)
        self.critic_history = tf.TensorArray(dtype=tf.float32, size=0,
                                             dynamic_size=True,
                                             clear_after_read=False)
        self.reward_history = tf.TensorArray(dtype=tf.float32, size=0,
                                             dynamic_size=True,
                                             clear_after_read=False)

    def actor_net_builder(self):
        input_ = keras.Input(shape=self.input_shape, dtype='float')
        common = keras.layers.Conv2D(32, (5, 5), strides=(3, 3),
                                     activation='relu')(input_)             # 32, 32, 32
        common = keras.layers.MaxPooling2D((2, 2))(common)                   # 32, 16, 16
        common = keras.layers.Conv2D(64, (3, 3), padding='SAME',
                                     strides=(3, 3),
                                     activation='relu')(common)             # 64, 4, 4
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(1, 1),
                                     activation='relu')(common)
        common = keras.layers.Dense(units=1024, activation='relu')(common)
        common = keras.layers.Dense(units=512, activation='relu')(common)
        common = keras.layers.Dense(units=128, activation='relu')(common)
        actor_mu = keras.layers.Dense(units=self.out_shape)(common)
        actor_sigma = keras.layers.Dense(units=self.out_shape)(common)

        model = keras.Model(inputs=input_, outputs=[actor_mu, actor_sigma])
        return model


if __name__ == '__main__':
    shape_in = (96, 96, 3)
    init = ddpg_net(shape_in, 1)
    init.actor_model.summary()


        

