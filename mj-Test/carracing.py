from cmath import pi
import math
import random
import time
import os

import pickle

import numpy as np
import matplotlib.pyplot as plt

import gym

env = gym.make('CarRacing-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('MountainCarContinuous-v0')



# Seeing the result in action
obs = env.reset()
steps_group = []
last_s = 0
done_cnt = 0
for s_ in range(1000000):
    env.render()
    # time.sleep(1/60)
    a = env.action_space.sample()  # take a random action
    obs, reward, done, info = env.step(a)
    if done:
        # print(s_)
        done_cnt += 1
        steps_group.append(s_-last_s)
        last_s = s_
        time.sleep(0.8)
        # env.reset()
        break
time.sleep(1000)
env.close()

print(done_cnt)
print(len(steps_group), steps_group)
print(last_s, last_s/done_cnt)
