import numpy as np

class agent:
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        
    def __call__(self, obs):
        return np.random.randint(self.action_space)