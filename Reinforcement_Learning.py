# -*- coding: utf-8 -*-
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

environment_name='CartPole-v0'
env=gym.make(environment_name)

episodes=10

for episode in range(1,episodes+1):
  state=env.reset()
  done=False
  score=0
  while not done:
    env.render()
    action=env.action_space.sample()
    n_state,reward,done,info=env.step(action)
    score+=reward
  print('episode:{} score:{}'.format(episode,score))
env.close()

env.observation_space.sample()
env.step(1)

#make directories
log_path=os.path.join('D:\Training','logs')

env=gym.make(environment_name)
env=DummyVecEnv([lambda: env])
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

#train the model
model.learn(total_timesteps=20000)

#save model
PPO_Path=os.path.join('D:\Training','Saved Models','PPO_Model_Cartpole')
model.save(PPO_Path)

#core metrics to look at is avg reward, avg episode length
evaluate_policy(model,env,n_eval_episodes=10,render=True)

#instead of using random sample we use the model to predict the action based on the observations
episodes=10
for episode in range(1,episodes+1):
  obs=env.reset()
  done=False
  score=0
  while not done:
    env.render()
    action,_=model.predict(obs)
    obs,reward,done,info=env.step(action)
    score+=reward
  print('episode:{} score:{}'.format(episode,score))
env.close()

#applying callback feature
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold
save_path=os.path.join('D:\Training','Saved Models')
stop_callback=StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback=EvalCallback(env,callback_on_new_best=stop_callback,eval_freq=10000,best_model_save_path=save_path,verbose=1)

model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000,callback=eval_callback)

#changing policies
net_arch=[dict(pi=[128,128,128,128],vf=[128,128,128,128])]
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,policy_kwargs={'net_arch':net_arch})
model.learn(total_timesteps=20000,callback=eval_callback)

from stable_baselines3 import DQN
model=DQN('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000)
DQN.load