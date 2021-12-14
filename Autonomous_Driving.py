# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:06:31 2021

@author: chait
"""
import gym
from gym import Env
from gym.spaces import Discrete,Box,Dict,Tuple,MultiBinary,MultiDiscrete

import numpy as np
import os
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

class ShowerEnv(Env):
    def __init__(self):
       
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        
    def step(self, action):
        self.state += action -1 
        self.shower_length -= 1 
        
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
    
        info = {}

        return self.state, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        self.shower_length = 60 
        return self.state


env=ShowerEnv()
env.observation_space.sample()
env.reset()
from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

log_path = os.path.join('D:\Training', 'Logs')




