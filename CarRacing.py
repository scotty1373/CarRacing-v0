#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import numpy
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from gym_torcs import TorcsEnv
from ou_noise import OUNoise
import pandas as pd
import numpy as np
import platform
import gym
import time
import os

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.001
MAX_MEMORY_LEN = 32000
MAX_STEP_EPISODE = 1000
TRAINABLE = True
VISION = True
DECAY = 0.99
CHANNEL = 1


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
        self.channel = CHANNEL
        self.train_start = 2000
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
        self.OUnoise = OUNoise(2)

        # self.actor_target_model.trainable = False
        # self.critic_target_model.trainable = False

        self.actor_history = []
        self.critic_history = []
        self.reward_history = []
        self.weight_hard_update()

    def state_store_memory(self, s, focus_, track_, a, r, s_t1, focus_t1_, track_t1_):
        self.memory.append((s, focus_, track_, a, r, s_t1, focus_t1_, track_t1_))

    def actor_net_builder(self):
        input_ = keras.Input(shape=self.input_shape, dtype='float', name='actor_input')
        common = keras.layers.Conv2D(8, (5, 5),
                                     strides=(1, 1),
                                     activation='relu')(input_)  # 8, 60, 60
        common = keras.layers.Conv2D(64, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 64, 20, 20
        common = keras.layers.Conv2D(128, (3, 3),
                                     strides=(3, 3),
                                     activation='relu')(common)     # 128, 6, 6
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
        actor_angle_in = keras.layers.Dense(units=16, activation='relu')(input_actor_angle)
        actor_accele_in = keras.layers.Dense(units=16, activation='relu')(input_actor_accele)
        concatenated_layer = keras.layers.Concatenate(axis=-1)([common, actor_angle_in, actor_accele_in])

        critic_output = keras.layers.Dense(units=self.out_shape, activation='tanh')(concatenated_layer)
        model = keras.Model(inputs=[input_state, input_actor_angle, input_actor_accele],
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

    @staticmethod
    def data_pcs(self, obs_: dict):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'opponents',
                 'rpm',
                 'track',
                 'wheelSpinVel',
                 'img']
        for i in names:
            exec("i=obs_.get(i)")
        return focus, speedX, speedY, speedZ, opponent, rpm, track, wheelSpinel, img

    def action_choose(self, s):
        angle_, accele_ = self.actor_model(s)
        # angle_ = tf.multiply(angle_, self.angle_range)
        # accele_ = tf.multiply(accele_, self.accele_range)
        noise = self.OUnoise.noise()
        # angle_ = tf.add(angle_, noise[0])
        return angle_, accele_

    # Exponential Moving Average update weight
    def weight_soft_update(self):
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
    def critic_loss(self, s, r, s_t1, a):
        # critic model q real
        q_real = self.critic_model([s, a[:, 0, :], a[:, 1, :]])
        # target critic model q estimate
        a_t1 = self.actor_target_model(s_t1)    # actor denormalization waiting!!!, doesn't matter with the truth action
        a_t1_ang, a_t1_acc = tf.split(a_t1, 2, axis=0)
        q_estimate = self.critic_target_model([s_t1,
                                               tf.squeeze(a_t1_ang, axis=0),
                                               tf.squeeze(a_t1_acc, axis=0)])
        # TD-target
        q_target = r + q_estimate * self.gamma
        return q_target, q_real

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_data = random.sample(self.memory, self.batch_size)
        s_, focus_,  track_, a_, r_, focus_t1_, track_t1_ = zip(*batch_data)
        s_ = np.array(s_, dtype='float').squeeze(axis=1)
        focus_ = np.array(focus_, dtype='float').squeeze()
        track_ = np.array(track_, dtype='float').squeeze()
        a_ = np.array(a_, dtype='float').squeeze(axis=2)   # ang = a[:, 0, :], acc = a[:, 1, :]

        r_ = np.array(r_, dtype='float').reshape(self.batch_size, -1)

        s_t1_ = np.array(s_t1_, dtype='float').squeeze(axis=1)
        focus_t1_ = np.array(focus_t1_, dtype='float').squeeze()
        track_t1_ = np.array(track_t1_, dtype='float').squeeze()
        # parameters initiation
        optimizer_actor = keras.optimizers.Adam(self.learning_rate_a)
        optimizer_critic = keras.optimizers.Adam(self.learning_rate_c)

        with tf.GradientTape() as tape:
            q_target, q_real = self.critic_loss(s_, r_, s_t1_, a_)
            q_target = tf.reduce_mean(q_target)
            q_real = tf.reduce_mean(q_real)
            # loss_value = keras.losses.mean_squared_error(q_real, q_target)
            # td-error
            loss = q_target - q_real

        grad_critic_loss = tape.gradient(q_real, agent.critic_model.trainable_weights, output_gradients=loss)
        optimizer_critic.apply_gradients(zip(grad_critic_loss, agent.critic_model.trainable_weights))

        with tf.GradientTape(persistent=True) as tape:
            a = self.actor_model(s_)
            a_ang, a_acc = tf.split(a, 2, axis=0)
            q = -self.critic_model([s_, tf.squeeze(a_ang, axis=[0]), tf.squeeze(a_acc, axis=[0])])
        grad_list = tape.gradient(q, a)
        grad_a = tape.gradient(a, agent.actor_model.trainable_weights, output_gradients=grad_list)
        optimizer_actor.apply_gradients(zip(grad_a, agent.actor_model.trainable_weights))
        del tape


if __name__ == '__main__':

    env = TorcsEnv(vision=VISION, throttle=True)

    test_train_flag = TRAINABLE

    action_shape = env.action_space.shape
    state_shape = np.array(env.observation_space.shape)
    state_shape[2] = 1
    action_range = env.action_space.high            # [1., 1., 1.]  ~  [-1.,  0.,  0.]

    agent = ddpg_Net((64, 64, 1), np.ndim(action_shape), action_range[1], action_range[0])
    agent.actor_model.summary()
    agent.critic_model.summary()
    epochs = 400
    timestep = 0
    count = 0

    while True:
        if np.mod(count, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        ep_history = np.array([])
        acc_ang_flag = 0
        live_time = 0

        for index in range(MAX_STEP_EPISODE):
            focus, _, _, _, _, _, track, _, obs = agent.data_pcs(ob)
            ang_net, acc_net = agent.action_choose(obs)
            ang = np.clip(np.random.normal(loc=ang_net, scale=agent.sigma_fixed),
                          -action_range[0], action_range[0])
            acc = np.clip(np.random.normal(loc=acc_net, scale=agent.sigma_fixed),
                          -action_range[1], action_range[1])

            action = np.array((0, acc, 0), dtype='float')
            ang = np.array(0).reshape(-1, CHANNEL)

            ob_t1, reward, done, _ = env.step(action)

            focus_t1, _, _, _, _, _, track_t1, _, obs_t1 = agent.data_pcs(ob_t1)
            c_v = agent.critic_model([obs_t1, ang, acc])
            c_v_target = agent.critic_target_model([obs_t1, ang, acc])

            if done:
                agent.state_store_memory(obs, focus, track, [ang, acc], reward, obs_t1, focus_t1, track_t1)
                print(f'terminated by environment, timestep: {timestep},'
                      f'epoch: {count}, reward: {reward}, angle: {ang},'
                      f'acc: {acc}, reward_mean: {np.array(ep_history).sum()}')
                break

            ep_history = np.append(ep_history, reward)
            agent.state_store_memory(obs, focus, track, [ang, acc], reward, obs_t1, focus_t1, track_t1)

            if test_train_flag is True:
                agent.train_replay()

            print(f'timestep: {timestep},'
                  f'epoch: {count}, reward: {reward}, angle: {ang},'
                  f'acc: {acc}, reward_mean: {np.array(ep_history).sum()} '
                  f'c_r: {c_v}, c_t: {c_v_target}, line_time: {live_time}')

            timestep += 1
            obs = obs_t1
            live_time += 1

        agent.weight_soft_update()
        if count == epochs:
            break
        elif count % 10 == 0:
            timestamp = time.time()
            agent.actor_model.save(CURRENT_PATH + '/' + f'action_model{timestamp}.h5')
            agent.critic_model.save(CURRENT_PATH + '/' + f'critic_model{timestamp}.h5')
        count += 1

    env.end()
    print("Finish.")