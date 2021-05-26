#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import numpy
import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow_probability as tfp
from collections import deque
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
import platform
import gym
import time
import os

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.001
MAX_MEMORY_LEN = 32000
MAX_STEP_EPISODE = 480
TRAINABLE = True
DECAY = 0.99
channel = 1


if platform.system() == 'windows':
    temp = os.getcwd()
    CURRENT_PATH = temp.replace('\\', '/')
else:
    CURRENT_PATH = os.getcwd()
CURRENT_PATH = os.path.join(CURRENT_PATH, 'save_Model')
if not os.path.exists(CURRENT_PATH):
    os.makedirs(CURRENT_PATH)


class ddpg_Net:
    def __init__(self, shape_in, num_output, accele_range, angle_range):
        self.input_shape = shape_in
        self.out_shape = num_output
        self.learning_rate_a = LEARNING_RATE_ACTOR
        self.learning_rate_c = LEARNING_RATE_CRITIC
        self.memory = deque(maxlen=MAX_MEMORY_LEN)
        self.channel = 1
        self.train_start = 200
        self.batch_size = 64
        self.gamma = 0.9
        self.sigma_fixed = 2
        self.critic_input_action_shape = 1
        self.angle_range = angle_range
        self.accele_range = accele_range
        self.actor_model = self.actor_net_builder()
        self.critic_model = self.critic_net_build()
        self.actor_target_model = self.actor_net_builder()

        self.critic_target_model = self.critic_net_build()

        # self.actor_target_model.trainable = False
        # self.critic_target_model.trainable = False

        self.actor_history = []
        self.critic_history = []
        self.reward_history = []
        self.weight_hard_update()

    def state_store_memory(self, s, s_r, s_l, a, r, s_t1, s_t1_r, s_t1_l):
        self.memory.append((s, s_r, s_l, a, r, s_t1, s_t1_r, s_t1_l))

    def actor_net_builder(self):
        input_ = keras.Input(shape=self.input_shape, dtype='float', name='actor_input')
        input_right = keras.Input(shape=(10,), dtype='float', name='Rotation angle right')
        input_left = keras.Input(shape=(10,), dtype='float', name='Rotation angle left')
        common = keras.layers.Conv2D(32, (5, 5),
                                     strides=(1, 1),
                                     activation='relu')(input_)  # 32, 36, 36
        common = keras.layers.Conv2D(64, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 64, 12, 12
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 128, 4, 4
        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=128, activation='relu')(common)
        # right_layer = keras.layers.Dense(units=16, activation='relu')(input_right)
        # left_layer = keras.layers.Dense(units=16, activation='relu')(input_left)
        concate_layer = keras.layers.Concatenate()([common, input_right, input_left])

        actor_angle = keras.layers.Dense(units=self.out_shape, activation='tanh')(concate_layer)

        actor_accela = keras.layers.Dense(units=self.out_shape, activation='sigmoid')(concate_layer)

        model = keras.Model(inputs=[input_, input_right, input_left], outputs=[actor_angle, actor_accela], name='actor')
        return model

    def critic_net_build(self):
        input_state = keras.Input(shape=self.input_shape,
                                  dtype='float', name='critic_state_input')
        input_right = keras.Input(shape=(10,), dtype='float', name='Rotation angle right')
        input_left = keras.Input(shape=(10,), dtype='float', name='Rotation angle left')
        input_actor_angle = keras.Input(shape=self.critic_input_action_shape,
                                        dtype='float', name='critic_action_angle_input')
        input_actor_accele = keras.Input(shape=self.critic_input_action_shape,
                                         dtype='float', name='critic_action_accele_input')

        common = keras.layers.Conv2D(32, (5, 5),
                                     strides=(1, 1),
                                     activation='relu')(input_state)  # 32, 36, 36
        common = keras.layers.Conv2D(64, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 64, 12, 12
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 128, 4, 4

        common = keras.layers.Flatten()(common)
        common = keras.layers.Dense(units=128, activation='relu')(common)
        concated = keras.layers.Concatenate()([common, input_right, input_left])
        actor_angle_in = keras.layers.Dense(units=16, activation='relu')(input_actor_angle)
        actor_accele_in = keras.layers.Dense(units=16, activation='relu')(input_actor_accele)
        concatenated_layer = keras.layers.Concatenate(axis=-1)([concated, actor_angle_in, actor_accele_in])

        critic_output = keras.layers.Dense(units=self.out_shape, activation='softplus')(concatenated_layer)
        model = keras.Model(inputs=[input_state, input_right, input_left,
                                    input_actor_angle, input_actor_accele],
                            outputs=critic_output,
                            name='critic')
        return model

    @staticmethod
    def image_process(obs):
        origin_obs = rgb2gray(obs)
        car_shape = origin_obs[44:84, 28:68].reshape(1, 40, 40, 1)
        state_bar = origin_obs[84:, 12:]
        right_position = state_bar[6, 36:46].reshape(-1, 10)
        left_position = state_bar[6, 26:36].reshape(-1, 10)
        car_range = origin_obs[44:84, 28:68][22:27, 17:22]

        return car_shape, right_position, left_position, car_range

    def action_choose(self, s, right_, left_):
        angle_, accele_ = self.actor_model([s, right_, left_])
        angle_ = tf.multiply(angle_, self.angle_range)
        accele_ = tf.multiply(accele_, self.accele_range)
        return angle_, accele_

    # Exponential Moving Average update weight
    def weight_soft_update(self):
        # self.actor_target_model.set_weights(self.actor_model.get_weights())
        # self.critic_target_model.set_weights(self.critic_model.get_weights())
        for i, j in zip(self.critic_model.trainable_weights, self.critic_target_model.trainable_weights):
            j.assign(j * DECAY + i * (1 - DECAY))
        for i, j in zip(self.actor_model.trainable_weights, self.actor_target_model.trainable_weights):
            j.assign(j * DECAY + i * (1 - DECAY))

    def weight_hard_update(self):
        self.actor_target_model.set_weights(self.actor_model.get_weights())
        self.critic_target_model.set_weights(self.critic_model.get_weights())

    '''
    for now the critic loss return target and real q value, that's
    because I wanna tape the gradient in one gradienttape, if the result
    is not good enough, split the q_real in another gradienttape to update
    actor network!!!
    '''
    def critic_loss(self, s, s_r, s_l, r, s_t1, s_t1_r, s_t1_l, a):
        # critic model q real
        q_real = self.critic_model([s, s_r, s_l, a[:, 0, :], a[:, 1, :]])
        # target critic model q estimate
        a_t1 = self.actor_target_model([s_t1, s_t1_r, s_t1_l])    # actor denormalization waiting!!!, doesn't matter with the truth action
        a_t1_ang, a_t1_acc = tf.split(a_t1, 2, axis=0)
        q_estimate = self.critic_target_model([s_t1, s_t1_r, s_t1_l,
                                               tf.squeeze(a_t1_ang, axis=0),
                                               tf.squeeze(a_t1_acc, axis=0)])
        # TD-target
        q_target = r + q_estimate * self.gamma
        return q_target, q_real

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_data = random.sample(self.memory, self.batch_size)
        s_, s_r_, s_l_, a_, r_, s_t1_, s_t1_r_, s_t1_l_ = zip(*batch_data)
        s_ = np.array(s_, dtype='float').squeeze(axis=1)
        s_r_ = np.array(s_r_, dtype='float').squeeze()
        s_l_ = np.array(s_l_, dtype='float').squeeze()
        a_ = np.array(a_, dtype='float').squeeze(axis=2)   # ang = a[:, 0, :], acc = a[:, 1, :]

        r_ = np.array(r_, dtype='float').reshape(self.batch_size, -1)

        s_t1_ = np.array(s_t1_, dtype='float').squeeze(axis=1)
        s_t1_r_ = np.array(s_t1_r_, dtype='float').squeeze()
        s_t1_l_ = np.array(s_t1_l_, dtype='float').squeeze()
        # parameters initiation
        optimizer_actor = keras.optimizers.Adam(self.learning_rate_a)
        optimizer_critic = keras.optimizers.Adam(self.learning_rate_c)

        with tf.GradientTape() as tape:
            q_target, q_real = self.critic_loss(s_, s_r_, s_l_, r_, s_t1_, s_t1_r_, s_t1_l_, a_)
            # q_target = tf.reduce_mean(q_target)
            # q_real = tf.reduce_mean(q_real)
            # loss_value = keras.losses.mean_squared_error(q_real, q_target)
            # td-error
            loss = q_target - q_real

        grad_critic_loss = tape.gradient(q_real, agent.critic_model.trainable_weights, output_gradients=loss)
        optimizer_critic.apply_gradients(zip(grad_critic_loss, agent.critic_model.trainable_weights))

        with tf.GradientTape(persistent=True) as tape:
            a = self.actor_model([s_, s_r_, s_l_])
            a_ang, a_acc = tf.split(a, 2, axis=0)
            q = -self.critic_model([s_, s_r_, s_l_, tf.squeeze(a_ang, axis=[0]), tf.squeeze(a_acc, axis=[0])])
        grad_list = tape.gradient(q, a)
        grad_a = tape.gradient(a, agent.actor_model.trainable_weights, output_gradients=grad_list)
        optimizer_actor.apply_gradients(zip(grad_a, agent.actor_model.trainable_weights))
        del tape


if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    env.seed(1)
    env = env.unwrapped

    test_train_flag = TRAINABLE

    action_shape = env.action_space.shape
    state_shape = np.array(env.observation_space.shape)
    state_shape[2] = 1
    action_range = env.action_space.high            # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    agent = ddpg_Net((40, 40, 1), np.ndim(action_shape), action_range[1], action_range[0])
    agent.actor_model.summary()
    agent.critic_model.summary()
    epochs = 400
    timestep = 0

    count = 0
    while True:
        _ = env.reset()
        for discard_index in range(50):
            action = np.array((0, 0, 0), dtype='float')
            obs, _, _, _ = env.step(action)

        obs, right, left, _ = agent.image_process(obs)
        # obs = np.stack((obs, obs, obs, obs), axis=2).reshape(1, obs.shape[0], obs.shape[1], -1)
        outrange_count = 0
        temp = []
        ep_history = np.array(temp)
        acc_ang_flag = 0

        for index in range(MAX_STEP_EPISODE):
            env.render()
            ang_net, acc_net = agent.action_choose(obs, right, left)
            ang = np.clip(np.random.normal(loc=ang_net, scale=agent.sigma_fixed),
                          -action_range[0], action_range[0])
            acc = np.clip(np.random.normal(loc=acc_net, scale=agent.sigma_fixed),
                          0, action_range[1])

            # action = np.array((ang, acc, 0.05), dtype='float')
            if acc_ang_flag == 0 and acc > 0:
                action = np.array((0, acc, 0), dtype='float')
                ang = np.array(0).reshape(-1, channel)
                acc_ang_flag += 2
            elif acc_ang_flag == 0 and acc <= 0:
                action = np.array((0, 0, acc), dtype='float')
                ang = np.array(0).reshape(-1, channel)
                acc_ang_flag += 2
            elif acc_ang_flag != 0:
                action = np.array((ang, 0, 0), dtype='float')
                acc = np.array(0).reshape(-1, channel)
                acc_ang_flag -= 1

            obs_t1, reward, done, _ = env.step(action)

            obs_t1, right_t1, left_t1, reward_recalculate_index = agent.image_process(obs_t1)
            # obs_t1 = np.append(obs[:, :, :, 1:], obs_t1, axis=3)

            c_v = agent.critic_model([obs_t1, right_t1, left_t1, ang, acc])
            c_v_target = agent.critic_target_model([obs_t1, right_t1, left_t1, ang, acc])

            if reward_recalculate_index.mean() < 0.4:
                reward += 0.9
                outrange_count = 0
            else:
                reward -= 0.5
                outrange_count += 1

            if done is True:
                print(f'terminated by environment, timestep: {timestep},'
                      f'epoch: {count}, reward: {reward}, angle: {ang},'
                      f'acc: {acc}, reward_mean: {np.array(ep_history).sum()}')
                break

            if outrange_count == 10:
                print('out of range')
                reward = -50
                ep_history = np.append(ep_history, reward)
                agent.state_store_memory(obs, right, left, [ang, acc], reward, obs_t1, right_t1, left_t1)
                break

            ep_history = np.append(ep_history, reward)
            agent.state_store_memory(obs, right, left, [ang, acc], reward, obs_t1, right_t1, left_t1)

            if test_train_flag is True:
                agent.train_replay()

            print(f'timestep: {timestep},'
                  f'epoch: {count}, reward: {reward}, angle: {ang},'
                  f'acc: {acc}, reward_mean: {np.array(ep_history).sum()} '
                  f'c_r: {c_v}, c_t: {c_v_target}')

            timestep += 1
            right = right_t1
            left = left_t1
            obs = obs_t1

        agent.weight_soft_update()
        if count % 10 == 0:
            timestamp = time.time()
            agent.actor_model.save(CURRENT_PATH + '/' + f'action_model{timestamp}.h5')
            agent.critic_model.save(CURRENT_PATH + '/' + f'critic_model{timestamp}.h5')
        count += 1