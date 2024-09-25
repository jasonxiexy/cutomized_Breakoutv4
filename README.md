# **Project Title: Breakout Game with Reinforcement Learning**

![plot](./charts/gamescreen.png)

Welcome to the Breakout Game Reinforcement Learning project! This repository contains the source code and documentation for our experiments with different reinforcement learning approaches to solve the classic Breakout game. The project explores traditional Q-learning and Deep Q-learning (DQN) methods with various parameters and setups.

## **Branches Overview**

### **Main Branch**
The `main` branch serves as the baseline of our repository. It contains the initial setup and fundamental codebase required to run the Breakout game environment.

### **DQN Iterative Versions**

#### **Branch: DQN_version1**
This branch contains the first iteration of the DQN model, focusing on initial parameter tuning and baseline performance evaluation.

#### **Branch: DQN_version2&3**
Building upon `DQN_version1`, this branch includes refinements and optimizations to improve the performance and stability of the DQN model.

### **Final Version**

#### **Branch: finalversion**
The `finalversion` branch contains the final submission for our project. This branch includes comprehensive experiments and results with four distinct models:

- **Q-learning-v1**: Our traditional Q-learning model trained with 50,000 episodes.
- **dqn_breakoutv1**: The first version of our Deep Q-learning model, trained with 30,000 episodes and a 0.1 wrapper probability.
- **dqn_breakoutv2**: An improved version of DQN-v1 with a lower wrapper probability of 0.001, trained with 30,000 episodes.
- **dqn_breakoutv3**: A benchmark model not using action uncertainty during training, tested under different conditions to evaluate its robustness and performance.

## **Results and Analysis**
The `finalversion` branch contains detailed reports and analyses of our experiments. The documentation includes comparisons of the models' performance, the impact of different parameters, and the effectiveness of action uncertainty in training.

Thank you for checking out our project! We hope you find our exploration of reinforcement learning in the Breakout game insightful and inspiring.
