from cmath import pi
import math
import random
import time
import os

import pickle

import numpy as np
import matplotlib.pyplot as plt

import gym


"""
@ref: https://blog.csdn.net/qq_32892383/article/details/89576003
"""

env = gym.make('CartPole-v1')

# env.env.theta_threshold_radians = 100 * 2 * math.pi / 360
# env.env.x_threshold = 22.4


def discretize(x):
    return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))


def create_bins(i, num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]


print("Sample bins for interval (-5,5) with 10 bins\n", create_bins((-5, 5), 10))

ints = [(-5, 5), (-2, 2), (-0.5, 0.5), (-2, 2)]  # intervals of values for each parameter
nbins = [20, 20, 10, 10]  # number of bins for each parameter
bins = [create_bins(ints[i], nbins[i]) for i in range(4)]
print(bins)


def discretize_bins(x):
    return tuple(np.digitize(x[i], bins[i]) for i in range(4))


# Q-Learning
q_table_obj_file = './publish-cartpole-v20220129-5.pkl.obj'
q_table_obj = {}
if os.path.isfile(q_table_obj_file):
    with open(q_table_obj_file, 'rb') as _f:
        q_table_obj = pickle.load(_f)
Q = q_table_obj
actions = (0, 1)


def qvalues(state):
    return [Q.get((state, a), 0) for a in actions]


def probs(v, eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v


if not Q:
    # 训练
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90

    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(35000):
        obs = env.reset()
        done = False
        cum_reward = 0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random() < epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions, weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)

            obs, rew, done, info = env.step(a)
            cum_reward += rew
            ns = discretize(obs)
            Q[(s, a)] = (1 - alpha) * Q.get((s, a), 0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch % 5000 == 0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards = []

    with open(q_table_obj_file, 'wb') as _f:
        pickle.dump(Q, _f)

    plt.plot(rewards)
    plt.show()

    def running_average(x, window):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    plt.plot(running_average(rewards, 100))
    plt.show()


# Seeing the result in action
obs = env.reset()
done_cnt = 0
steps_group = []
last_s = 0
for s_ in range(1000000):
    s = discretize(obs)
    env.render()
    time.sleep(1/24)
    # a = env.action_space.sample()  # take a random action
    v = probs(np.array(qvalues(s)))
    a = random.choices(actions, weights=v)[0]
    obs, reward, done, info = env.step(a)
    # print(discretize_bins(obs))
    # print(discretize(obs))
    if done:
        # print(s_)
        done_cnt += 1
        steps_group.append(s_-last_s)
        last_s = s_
        time.sleep(0.8)
        env.reset()
        break
env.close()

print(done_cnt)
print(len(steps_group), steps_group)
print(last_s, last_s/done_cnt)
