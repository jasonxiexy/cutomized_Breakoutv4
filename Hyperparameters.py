class Hyperparameters:
    def __init__(self):
        # non_DRL learning_rate = 0.1
        # DQN learning_rate = 5e-4
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.batch_size = 32
        self.targetDQN_update_rate = 10
        self.num_episodes = 30000
        self.epsilon = 1.0
        self.epsilon_decay = 5e-5
        self.buffer_size = 50000
        self.min_epsilon = 0.1
        self.replace_target_cnt = 5000
