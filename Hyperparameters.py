class Hyperparameters:
    def __init__(self):
        # non_DRL learning_rate = 0.1
        # DQN learning_rate = 5e-4
        self.learning_rate = 5e-4
        self.discount_factor = 0.99
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes = 50000
        self.epsilon_decay = 0.999
        self.buffer_size = 10000
        self.min_epsilon = 0.01
