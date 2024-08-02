# import gymnasium as gym
# import numpy as np
# from QLearningAgent import QLearningAgent, saveQ, loadAgent
# from CustomEnv import ActionUncertaintyWrapper
# import pygame
#
# pygame.init()
#
# # Initialize the environment
# env = gym.make("Breakout-v4", obs_type="rgb", render_mode=None)
# # env = gym.make("Breakout-v4", obs_type="rgb", render_mode="human")
#
# # env = gym.make('ALE/Breakout-v5', render_mode="human")  # remove render_mode in training
#
# wrapped_env = ActionUncertaintyWrapper(env, prob=0.1)
#
# # Get the number of states and actions
# n_states = 210 * 160 * 3  # Simplification, use appropriate state space representation
# # n_states = env.observation_space.n
# n_actions = env.action_space.n
# # n_actions = wrapped_env.action_space.n
#
# print(f'{n_states} states')
# print(f'{n_actions} actions')
#
# # Set the parameters
# gamma = 0.99    # discount factor
# alpha = 0.1     # learning rate
# epsilon = 0.99   # exploration rate
# epsilon_decay = 0.995
# min_epsilon = 0.01
#
# # Load your implemented Agent
# agent = QLearningAgent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, env=wrapped_env)
# # agent = QLearningAgent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, env=wrapped_env)
#
#
# # Set the number of episodes
# n_episodes = 50000
#
# # Train by Q-Learning
# agent.QLearning(n_episodes)
# # Save the Q-function
# saveQ(agent)
#
# # Evaluate the trained agent
# n_test = 10  # Number of testing episodes
# # agent = loadAgent(n_states, n_actions, discount=gamma, lr=alpha, epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, env=wrapped_env)
# # reward_array = agent.eval(n_test, agent.Q)
# # print(f'Q-Learning Success Rate: {success_rate}')
#
# pygame.quit()

##############################################################
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from QLearningAgent import QLearningAgent, saveQ, loadAgent
from DQNAgent import DQNAgent
from Hyperparameters import Hyperparameters
from CustomEnv import ActionUncertaintyWrapper
import pygame
from tqdm import trange

pygame.init()

def run_qlearning_train(hp, wrapped_env, n_episodes):
    # K = trange(n_episodes)
    # Initialize Q-learning agent
    agent = QLearningAgent(n_states=210 * 160 * 3, n_actions=wrapped_env.action_space.n, discount=hp.discount_factor, lr=hp.learning_rate, epsilon=1.0, epsilon_decay=hp.epsilon_decay, min_epsilon=hp.min_epsilon, env=wrapped_env)

    # Train by Q-Learning
    Q, reward_array, reward_per_episode_array = agent.QLearning(n_episodes)
    saveQ(agent)

    # Plotting learning curve
    plt.figure('Training Learning Curve (Q-Learning)')
    plt.plot([k + 1 for k in range(n_episodes)], reward_array, color='black', linewidth=0.5)
    plt.ylabel('Average Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title(f'Learning by Q-Learning for {n_episodes} Episodes', fontsize=12)
    plt.savefig('Training_QLearningAverageReward.png', format='png', dpi=900)
    plt.show()

    plt.figure('Training Total Reward per Episode Curve (Q-Learning)')
    plt.plot([k + 1 for k in range(n_episodes)], reward_per_episode_array, color='black', linewidth=0.5)
    plt.ylabel('Total Reward', fontsize=12)
    plt.xlabel('Episode', fontsize=12)
    plt.title(f'Reward by Q-Learning for {n_episodes} Episodes', fontsize=12)
    plt.savefig('Training_QLearningTotalReward.png', format='png', dpi=900)
    plt.show()

def run_qlearning_play(hp, wrapped_env, n_test):
    agent = loadAgent(210 * 160 * 3, wrapped_env.action_space.n, hp.discount_factor, hp.learning_rate, 1.0, hp.epsilon_decay, hp.min_epsilon, wrapped_env)
    reward_list = agent.eval(n_test, agent.Q)

    plt.figure('Testing Learning Curve (Q-Learning)')
    plt.plot([k + 1 for k in range(n_test)], reward_list, color='black', linewidth=0.5)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title(f'Average Reward by Q-Learning for {n_test} Episodes (Testing)', fontsize=12)
    plt.savefig('Testing_QLearningReward.png', format='png', dpi=900)
    plt.show()

def run_dqn_train(hp, wrapped_env, num_episodes):
    K = trange(num_episodes)

    R_avg = 0
    # Initialize DQN agent
    agent = DQNAgent(wrapped_env, hp)
    print(agent.device)
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    total_reward_per_episode = np.zeros(num_episodes)
    average_reward_array = np.zeros(num_episodes)

    for k in K:
        state = wrapped_env.reset()[0]
        state = np.transpose(state, (2, 0, 1))  # Transpose to match the input shape of the CNN
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = wrapped_env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))
            agent.replay_buffer.store(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward
            agent.optimize_model()

            if done:
                R_avg = R_avg + (total_reward - R_avg) / (k + 1)
                average_reward_array[k] = R_avg
                total_reward_per_episode[k] = total_reward
                agent.update_epsilon()

        if k+1 % hp.targetDQN_update_rate == 0:
            agent.update_target_net()

        K.set_description(f"Episode {k+1}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        K.refresh()

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_breakout.pth")

    # Plotting learning curve of total reward per episode
    plt.plot(average_reward_array)
    plt.ylabel('Average Reward')
    plt.xlabel('Episode')
    plt.title('Training Average Reward per Episode Curve (DQL)')
    plt.savefig('Training_DQN_Average_Reward.png', format='png', dpi=900)
    plt.show()

    # Plotting learning curve of total reward per episode
    plt.plot(total_reward_per_episode)
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.title('Training Total Reward per Episode Curve (DQL)')
    plt.savefig('Training_DQN_Total_Reward.png', format='png', dpi=900)
    plt.show()

def run_dqn_play(hp, wrapped_env, n_test):
    agent = DQNAgent(wrapped_env, hp)
    agent.policy_net.load_state_dict(torch.load("dqn_breakout.pth"))
    agent.policy_net.eval()

    rewards = []

    for episode in range(n_test):
        state = wrapped_env.reset()[0]
        state = np.transpose(state, (2, 0, 1))  # Transpose to match the input shape of the CNN
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = wrapped_env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Test Episode {episode + 1}, Reward: {total_reward}")

    plt.plot(rewards)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.title('Testing Reward Curve (DQN)')
    plt.savefig('Testing_DQNReward.png', format='png', dpi=900)
    plt.show()

if __name__ == "__main__":
    # Initialize hyperparameters
    hp = Hyperparameters()
    env = gym.make("Breakout-v4", obs_type="rgb", render_mode=None)
    wrapped_env = ActionUncertaintyWrapper(env, prob=0.1)

    # Set the number of episodes
    n_episodes = 10000
    n_test = 10

    # Choose the model to run and mode
    model_type = 'DQN'  # Change to 'Q-Learning' to run the Q-learning model/ 'DQN'
    train = True  # Change to False to run in play mode

    if model_type == 'Q-Learning':
        if train:
            run_qlearning_train(hp, wrapped_env, n_episodes)
        else:
            run_qlearning_play(hp, wrapped_env, n_test)
    elif model_type == 'DQN':
        if train:
            run_dqn_train(hp, wrapped_env, n_episodes)
        else:
            run_dqn_play(hp, wrapped_env, n_test)

pygame.quit()