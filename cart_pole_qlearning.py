import numpy as np
from algorithms import qlearning, random_policy
import gymnasium as gym

NUM_EPS = 100

env = gym.make('CartPole-v1')

# pos, vel, ang, angVel
state_bounds = ((-2.4, 2.4), (-4.8, 4.8), (-.21, .21), (-1, 1))
state_buckets = (1, 1, 6, 3)
# assign observation values to buckets
def discretize(obs): 
    state = []
    for i, ob in enumerate(obs):
        l, u = state_bounds[i]
        b = state_buckets[i]
        x = int(b * (ob - l) / (u - l))
        x = np.clip(x, 0, b - 1)
        state.append(x)
    return tuple(state)

agent = qlearning.agent(state_buckets, (env.action_space.n,))

for ep in range(NUM_EPS):
    state, _ = env.reset()
    state = discretize(state)
    ep_reward = 0

    while True:
        action = agent(state)
        next_state, reward, terminated, truncated, _ = env.step(action[0])
        next_state = discretize(next_state)
        ep_reward += reward

        agent.learn(state, action, reward, next_state)

        state = next_state

        if terminated or truncated:
            print(int(agent.epsilon * 100), ep_reward)
            agent.epsilon *= 0.9 # epsilon decay
            break

    # print(agent.q_table[0, 0])
        
env.close()