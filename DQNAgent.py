import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
from DQN import DQN


class ReplayBuffer:
    def __init__(self, size, device):
        self.size = size
        self.device = device
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)

    def store(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
        self.priorities.append(max(self.priorities, default=1))

    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        states, actions, next_states, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.hp = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(env.action_space.n).to(self.device)
        self.target_net = DQN(env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hp.learning_rate)
        self.replay_buffer = ReplayBuffer(self.hp.buffer_size, self.device)
        self.steps_done = 0
        self.epsilon = 1.0
        self.epsilon_decay = self.hp.epsilon_decay
        self.min_epsilon = self.hp.min_epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.hp.batch_size:
            return
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.hp.batch_size)

        # Ensure actions tensor has the correct shape
        actions = actions.unsqueeze(1)

        # Compute Q values for the selected actions
        state_action_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Compute the next state values using the target network
        next_state_values = self.target_net(next_states).max(1)[0].detach()

        # Compute the expected state-action values
        expected_state_action_values = (next_state_values * self.hp.discount_factor) * (1 - dones) + rewards

        # Compute the loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
