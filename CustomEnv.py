import gymnasium as gym
import numpy as np


class ActionUncertaintyWrapper(gym.Wrapper):
    def __init__(self, env, prob=0.1):
        super(ActionUncertaintyWrapper, self).__init__(env)
        self.prob = prob  # Probability of changing the action

    def step(self, action):
        if np.random.rand() < self.prob:
            action = self.env.action_space.sample()  # Randomly change the action
        return self.env.step(action)
