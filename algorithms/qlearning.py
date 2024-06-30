import numpy as np

class agent:
    # state_space - tuple, discrete
    # action_space - tuple, discrete
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros(state_space + action_space)
        self.epsilon = 1  # exploration factor
        self.gamma = 1      # discount rate
        self.lr = 0.1

    def __call__(self, state):
        if np.random.rand() < self.epsilon:
            return tuple(np.random.randint(self.action_space))
        index = np.argmax(self.q_table[state])
        return np.unravel_index(index, self.action_space)
    
    def learn(self, state, action, reward, next_state):
        td_error = reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
