import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
from DQN import DQN
import random
from collections import namedtuple

# Create a named tuple to more semantically transform transitions and batches of transitions
Transition = namedtuple('transition', ('state', 'action', 'reward', 'state_', 'done', 'raw_state'))


class ReplayBuffer:
    def __init__(self, size, device):
        self.buffer = []
        self.max_size = size
        self.pointer = 0
        self.device = device

    def add_transition(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)

        self.buffer[self.pointer] = Transition(*args)
        self.pointer = int((self.pointer + 1) % self.max_size)

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)

        # Converts batch of transitions to transitions of batches
        batch = Transition(*zip(*batch))

        return batch

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, state_space, action_space, hyperparameters):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.hp = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.state_space, self.action_space).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hp.learning_rate)
        self.loss = torch.nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(self.hp.buffer_size, self.device)
        self.steps_done = 0
        self.batch_size = self.hp.batch_size
        self.epsilon = self.hp.epsilon
        self.epsilon_decay = self.hp.epsilon_decay
        self.min_epsilon = self.hp.min_epsilon
        self.replace_target_cnt = self.hp.replace_target_cnt # After how many training iterations the target network should update
        self.learn_counter = 0
        self.update_target_net()

    # Returns the greedy action according to the policy net
    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        # action = self.policy_net(obs).argmax().item()
        q_values = self.policy_net(obs).squeeze()
        max_value = q_values.max().item()
        max_indices = torch.nonzero(q_values == max_value).squeeze().tolist()
        if isinstance(max_indices, int):
            action = max_indices
        else:
            action = random.choice(max_indices)
        return action

    # epsilon greedy
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.choice([x for x in range(self.action_space)])
        else:
            action = self.greedy_action(state)
        return action

    def optimize_model(self, num_it):
        if len(self.replay_buffer) < self.hp.batch_size:
            return

        for i in range(num_it):
            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate best next action value from the target net and detach from graph
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)
            # Using q_next and reward, calculate q_target
            # (1-done) ensures q_target is 0 if transition is in a terminating state
            q_target = (1 - done) * (reward + self.hp.discount_factor * q_next) + (done * reward)

            # Compute the loss
            # loss = self.loss(q_target, q_eval).to(self.device)
            loss = self.loss(q_eval, q_target).to(self.device)

            # Perform backward propagation and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Increment learn_counter (for dec_eps and replace_target_net)
            self.learn_counter += 1

            # Check replace target net
            self.update_target_net()

            # Save model & decrement epsilon
        self.policy_net.save_model()
        self.update_epsilon()

    def sample_batch(self):
        batch = self.replay_buffer.sample_batch(self.hp.batch_size)
        state_shape = batch.state[0].shape

        # Convert lists of numpy arrays to single numpy arrays
        states_array = np.array(batch.state)
        states_array_ = np.array(batch.state_)

        # Convert numpy arrays to tensors with correct dimensions
        state = torch.tensor(states_array).view(self.hp.batch_size, -1, state_shape[1], state_shape[2]).float().to(
            self.device)
        state_ = torch.tensor(states_array_).view(self.hp.batch_size, -1, state_shape[1], state_shape[2]).float().to(
            self.device)

        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # Updates the target net to have same weights as policy net
    def update_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print('Target network replaced')

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon \
                        else self.min_epsilon

    # Stores a transition into memory
    def store_transition(self, *args):
        self.replay_buffer.add_transition(*args)
