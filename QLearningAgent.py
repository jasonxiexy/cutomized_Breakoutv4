import gymnasium.wrappers.time_limit
import numpy as np
from tqdm import trange
import gymnasium
import matplotlib.pyplot as plt
import pickle

class Agent():
    def __init__(self, n_states, n_actions, discount, lr, epsilon, env: gymnasium.wrappers.time_limit.TimeLimit):
        self.gamma = discount
        self.alpha = lr
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.env = env
        # self.Q = np.zeros((self.n_states, self.n_actions))
        self.Q = {}
    # def epsGreedy(self, Q, s):
    #     if np.random.rand() < self.epsilon:
    #         return np.random.randint(self.n_actions)
    #     else:
    #         max_value = np.max(Q[s])
    #         print(Q[s].shape)
    #         print(max_value)
    #         max_actions = np.where(Q[s] == max_value)[0]
    #         print(np.where(Q[s] == max_value)[0].shape)
    #         print(max_actions)
    #         return np.random.choice(max_actions)
    def discretize_state(self, observation):
        # Example of discretizing state, you can use other methods
        return tuple((observation // 255).astype(np.int32).flatten())

    def epsGreedy(self, Q, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            if s in Q:
                max_value = np.max(Q[s])
                max_actions = np.where(Q[s] == max_value)[0]
                return np.random.choice(max_actions)
            else:
                return np.random.randint(self.n_actions)

    def QLearning(self, n_episodes):
        K = trange(n_episodes)
        R_avg = 0
        reward_array = np.zeros(n_episodes)
        Q = self.Q

        for k in K:
            total_reward = 0
            observation, _ = self.env.reset()
            s = self.discretize_state(observation)
            terminated = False

            while not terminated:
                a = self.epsGreedy(Q, s)
                observation, reward, terminated, _, _ = self.env.step(a)
                s_next = self.discretize_state(observation)
                total_reward += reward

                if s not in Q:
                    Q[s] = np.zeros(self.n_actions)
                if s_next not in Q:
                    Q[s_next] = np.zeros(self.n_actions)

                Q[s][a] += self.alpha * (reward + self.gamma * np.max(Q[s_next]) - Q[s][a])
                s = s_next

                if terminated:
                    K.set_description(f'Episode {k+1} ended')
                    K.refresh()
                    R_avg = R_avg + (total_reward - R_avg) / (k+1)
                    reward_array[k] = R_avg

        self.env.close()
        self.Q = Q

        plt.figure('Learning Curve')
        plt.plot([k+1 for k in range(n_episodes)], reward_array, color='black', linewidth=0.5)
        plt.ylabel('Average Reward', fontsize=12)
        plt.xlabel('Episode', fontsize=12)
        plt.title(f'Learning by Q-Learning for {n_episodes} Episodes', fontsize=12)
        plt.show()

        return Q, reward_array

    def eval(self, n_episodes, Q=None):
        if Q is not None:
            self.Q = Q
        K = trange(n_episodes)
        success_rate = 0

        for k in K:
            observation, _ = self.env.reset()
            s = self.discretize_state(observation)
            if s in Q:
                max_value = np.max(Q[s])
                max_actions = np.where(Q[s] == max_value)[0]
                a = np.random.choice(max_actions)
            else:
                a = np.random.randint(self.n_actions)
            terminated = False

            while not terminated:
                observation, reward, terminated, _, _ = self.env.step(a)
                s_next = self.discretize_state(observation)
                if s_next in Q:
                    max_value_next = np.max(Q[s_next])
                    max_actions_next = np.where(Q[s_next] == max_value_next)[0]
                    a_next = np.random.choice(max_actions_next)
                else:
                    a_next = np.random.randint(self.n_actions)
                if terminated:
                    K.set_description(f'Episode {k+1} ended with Reward {reward}')
                    K.refresh()
                    success_rate += reward
                    break
                s, a = s_next, a_next

        self.env.close()
        return success_rate

def saveQ(agent: Agent):
    with open('Qfunction.pkl', 'wb') as outp:
        pickle.dump(agent.Q, outp, pickle.HIGHEST_PROTOCOL)

def loadAgent(n_states, n_actions, discount, lr, epsilon, env):
    agent = Agent(n_states, n_actions, discount, lr, epsilon, env)
    with open('Qfunction.pkl', 'rb') as inp:
        agent.Q = pickle.load(inp)
    return agent
