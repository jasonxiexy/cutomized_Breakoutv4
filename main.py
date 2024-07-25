import gymnasium as gym
import numpy as np
from QLearningAgent import Agent, saveQ, loadAgent
from CustomEnv import ActionUncertaintyWrapper
import pygame

pygame.init()

# Initialize the environment
env = gym.make("Breakout-v4", obs_type="rgb", render_mode="human")
# env = gym.make('ALE/Breakout-v5', render_mode="human")  # remove render_mode in training

wrapped_env = ActionUncertaintyWrapper(env, prob=0.1)

# Get the number of states and actions
n_states = 210 * 160 * 3  # Simplification, use appropriate state space representation
# n_states = env.observation_space.n
n_actions = env.action_space.n
# n_actions = wrapped_env.action_space.n

print(f'{n_states} states')
print(f'{n_actions} actions')

# Set the parameters
gamma = 0.99
alpha = 0.1
epsilon = 0.1

# Load your implemented Agent
# agent = Agent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, env=wrapped_env)
agent = Agent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, env=env)


# Set the number of episodes
n_episodes = 10

# Train by Q-Learning
agent.QLearning(n_episodes)
# Save the Q-function
saveQ(agent)

# Evaluate the trained agent
n_test = 100  # Number of testing episodes
agent = loadAgent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, env=wrapped_env)
success_rate = agent.eval(n_test, agent.Q)
print(f'Q-Learning Success Rate: {success_rate}')

pygame.quit()
