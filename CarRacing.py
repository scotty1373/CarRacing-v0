#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_probability as tfp
import gym

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.01


class ddpg_Net:
    def __init__(self, shape_in, num_output):
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.gamma = 0.9
        self.sigma_fixed = 2
        self.critic_input_action_shape = 1
        self.actor_model = self.actor_net_builder()
        self.critic_model = self.critic_net_build()
        self.actor_target_model = self.actor_net_builder()
        self.critic_target_model = self.critic_net_build()

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
        input_ = keras.Input(shape=self.input_shape, dtype='float', name='actor_input')
        common = keras.layers.Conv2D(32, (5, 5), strides=(3, 3),
                                     activation='relu')(input_)             # 32, 32, 32
        common = keras.layers.MaxPooling2D((2, 2))(common)                   # 32, 16, 16
        common = keras.layers.Conv2D(64, (3, 3), padding='SAME',
                                     strides=(3, 3),
                                     activation='relu')(common)             # 64, 4, 4
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(1, 1),
                                     activation='relu')(common)
        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=128, activation='relu')(common)
        actor_mu = keras.layers.Dense(units=self.out_shape)(common)

        model = keras.Model(inputs=input_, outputs=actor_mu, name='actor')
        return model

    def critic_net_build(self):
        input_state = keras.Input(shape=self.input_shape,
                                  dtype='float', name='critic_state_input')
        input_action = keras.Input(shape=self.critic_input_action_shape,
                                   dtype='float', name='critic_action_input')
        common = keras.layers.Conv2D(32, (5, 5), strides=(3, 3),
                                     activation='relu')(input_state)             # 32, 32, 32
        common = keras.layers.MaxPooling2D((2, 2))(common)                   # 32, 16, 16
        common = keras.layers.Conv2D(64, (3, 3), padding='SAME',
                                     strides=(3, 3),
                                     activation='relu')(common)             # 64, 4, 4
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(1, 1),
                                     activation='relu')(common)
        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=128, activation='relu')(common)

        actor_in = keras.layers.Dense(units=32, activation='relu')(input_action)
        concatenated_layer = keras.layers.Concatenate(axis=-1)([common, actor_in])

        critic_output = keras.layers.Dense(units=self.out_shape, activation='relu')(concatenated_layer)
        model = keras.Model(inputs=[input_state, input_action], outputs=critic_output, name='critic')
        return model

    def weight_update(self):
        self.actor_target_model.set_weights(self.actor_model.get_weights())
        self.critic_target_model.set_weights(self.critic_model.get_weights())

    def critic_loss(self, s, r, s_t1, a, a_t1):
        # critic model q real
        q_real = self.critic_model(s, a)
        # target critic model q estimate
        q_estimate = self.critic_target_model(s_t1, a_t1)
        # TD-target
        q_target = r + q_estimate * self.gamma
        return q_target, q_real

    def learn(self, s, ):


if __name__ == '__main__':
    shape_in = (96, 96, 3)
     = ddpg_Net(shape_in, 1)
    init.actor_model.summary()
    init.critic_model.summary()



        

