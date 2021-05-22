#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_probability as tfp
import gym
from collections import deque

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.01
MAX_MEMORY_LEN = 32000
accele_range = 1
angle_range = 3


class ddpg_Net:
    def __init__(self, shape_in, num_output):
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.memory = deque(maxlen=MAX_MEMORY_LEN)
        self.train_start = 100
        self.batch_size = 64
        self.gamma = 0.9
        self.sigma_fixed = 2
        self.critic_input_action_shape = 1
        self.angle_range = angle_range
        self.accele_range = accele_range
        self.actor_model = self.actor_net_builder()
        self.critic_model = self.critic_net_build()
        self.actor_target_model = self.actor_net_builder()
        self.actor_target_model.trainable = False
        self.critic_target_model = self.critic_net_build()
        self.critic_target_model.trainable = False

        self.actor_history = tf.TensorArray(dtype=tf.float32, size=0,
                                            dynamic_size=True,
                                            clear_after_read=False)
        self.critic_history = tf.TensorArray(dtype=tf.float32, size=0,
                                             dynamic_size=True,
                                             clear_after_read=False)
        self.reward_history = tf.TensorArray(dtype=tf.float32, size=0,
                                             dynamic_size=True,
                                             clear_after_read=False)

    def state_store_memory(self, s, a, r, s_t1):
        self.memory.append((s, a, r, s_t1))

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
        actor_angle = keras.layers.Dense(units=self.out_shape, activation='tanh')(common)

        actor_accela = keras.layers.Dense(units=self.out_shape, activation='tanh')(common)

        model = keras.Model(inputs=input_, outputs=[actor_angle, actor_accela], name='actor')
        return model

    def critic_net_build(self):
        input_state = keras.Input(shape=self.input_shape,
                                  dtype='float', name='critic_state_input')
        input_actor_angle = keras.Input(shape=self.critic_input_action_shape,
                                   dtype='float', name='critic_action_angle_input')
        input_actor_accele = keras.Input(shape= self.critic_input_action_shape,
                                         dtype='float', name='critic_action_accele_input')
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

        actor_angle_in = keras.layers.Dense(units=32, activation='relu')(input_actor_angle)
        actor_accele_in = keras.layers.Dense(units=32, activation='relu')(input_actor_accele)
        concatenated_layer = keras.layers.Concatenate(axis=-1)([common, actor_angle_in, actor_accele_in])

        critic_output = keras.layers.Dense(units=self.out_shape,
                                           activation='relu')(concatenated_layer)
        model = keras.Model(inputs=[input_state, input_actor_angle,
                                    input_actor_accele],
                            outputs=critic_output,
                            name='critic')
        return model

    def action_choose(self, s):
        angle_, accele_ = self.actor_model(s)
        angle_ = tf.multiply(angle_, self.angle_range)
        accele_ = tf.multiply(accele_, self.accele_range)
        return angle_, accele_

    def weight_update(self):
        self.actor_target_model.set_weights(self.actor_model.get_weights())
        self.critic_target_model.set_weights(self.critic_model.get_weights())

    '''
    for now the critic loss return target and real q value, that's
    because I wanna tape the gradient in one gradienttape, if the result
    is not good enought, split the q_real in another gradientrapt to update
    actor network!!!
    '''
    def critic_loss(self, s, r, s_t1, a):
        # critic model q real
        q_real = self.critic_model([s, a])
        # target critic model q estimate
        a_t1 = self.actor_target_model(s_t1)                # actor denormalization waiting!!!, doesn't matter with the truth action
        q_estimate = self.critic_target_model([s_t1, a_t1])
        # TD-target
        q_target = r + q_estimate * self.gamma
        return q_target, q_real

    def learn(self):




    def (self, s, r, s_t1, a):
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)

        # parameters initiation
        optimizer_actor = keras.optimizers.Adam(-self.learning_rate_a)
        optimizer_critic = keras.optimizers.Adam(self.learning_rate_c)

        with tf.GradientTape(persistent=True) as tape:
            q_target, q_real = self.critic_loss(s, r, s_t1, a)
            loss_policy = q_real
            loss_value = keras.losses.mean_squared_error(q_target, q_real)

        optimizer_actor.minimize(loss_policy, var_list=self.actor_model.trainable_weights, tape=tape)
        optimizer_critic.minimize(loss_value, var_list=self.critic_model.trainable_weights, tape=tape)


if __name__ == '__main__':
    shape_in = (96, 96, 3)
    init = ddpg_Net(shape_in, 1)
    init.actor_model.summary()
    init.critic_model.summary()
    a = 1



        

