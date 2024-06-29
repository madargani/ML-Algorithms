from algorithms import random_policy
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
agent = random_policy.agent(env.action_space.n)

obs, _ = env.reset()
for _ in range(100):
    action = agent(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        
env.close()