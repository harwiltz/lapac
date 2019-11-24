import numpy as np
import gym
import gym.envs.box2d

class ModifiedCarRacing(gym.envs.box2d.CarRacing):
    def __init__(self):
        super(ModifiedCarRacing, self).__init__()

    def step(self, action):
        img, rew, done, info = super(ModifiedCarRacing, self).step(action)
        return img[7:-8,:-15,:], rew, done, info

    def custom_observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(81,81,3), dtype=np.uint8)
