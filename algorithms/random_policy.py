import numpy as np

class agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        
    def __call__(self, obs):
        return np.random.randint(self.n_actions)