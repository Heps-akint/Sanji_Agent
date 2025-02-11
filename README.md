# Learning Procedural Knowledge Through Reinforcement Learning

This repository contains a reinforcement learning project developed as a side project before commencing my PhD. The project demonstrates how to teach an agent procedural knowledge—in this case, following a cooking recipe—using a Deep Q-Network (DQN) approach implemented in PyTorch. The work involved research, design of a custom environment, implementation of a neural network with residual blocks, and the training loop. The code is intended to be self-contained and executable on a standard computer with an RTX 3060 Ti GPU.

---

## Table of Contents

- [Overview](#overview)
- [Motivation and Research](#motivation-and-research)
- [Environment Description](#environment-description)
- [Methodology](#methodology)
  - [Deep Q-Network (DQN)](#deep-q-network-dqn)
  - [Neural Network Architecture](#neural-network-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training and Evaluating the Agent](#training-the-agent)
- [Results and Analysis](#results-and-analysis)
- [References](#references)

---

## Overview

This project implements a reinforcement learning (RL) agent to learn procedural knowledge by following a simulated cooking recipe. The agent interacts with an environment that represents a recipe as a sequence of steps, receiving rewards for correct actions and penalties for errors. The training is carried out using a Deep Q-Network (DQN), with the goal of eventually having the agent form an internal representation of the task.

---

## Motivation and Research

The project was conceived as a side project to complement my forthcoming PhD research on _Learning Procedural Knowledge through Human-AI Interaction_. I aimed to develop an RL agent that can learn a series of actions in a way analogous to how animals or humans learn sequential tasks in the natural world. The decision to use a cooking recipe was motivated by the structured yet sequential nature of everyday tasks, making it an ideal testbed for procedural learning.

---

## Environment Description

The environment simulates a cooking recipe as a sequence of four steps, represented by the indices `[0, 1, 2, 3]`. At each step, the agent selects an action. If the action matches the expected step in the recipe, the agent receives a positive reward and progresses to the next step. Otherwise, the episode terminates with a penalty.

This design is inspired by natural processes—similar to an animal verifying its path step by step—and is described in standard RL texts such as Sutton & Barto (2018).

---

## Methodology

### Deep Q-Network (DQN)

The agent uses a DQN to approximate the Q-function, which estimates the value of taking a given action in a specific state. Experience replay and an epsilon‑greedy strategy are employed to balance exploration and exploitation. The epsilon parameter decays over episodes, shifting the policy from random exploration to exploiting learned behaviour.

### Neural Network Architecture

The network is built using PyTorch and includes residual blocks. This design choice was influenced by the success of residual networks in maintaining information flow, a concept analogous to how natural systems maintain stability across sequential learning tasks.

The architecture consists of:
- An initial convolutional layer for feature extraction.
- A stack of residual blocks for deeper representation.
- Fully connected layers that output policy logits and state value estimates.

---

## Installation

Ensure you have Python 3.10 or later installed. Then install the required packages:

```bash
pip install torch numpy
```

Clone this repository:

```bash
git clone https://github.com/Heps-akint/Sanji_Agent.git
cd Sanji_Agent.ipynb
```

---

## Usage

### Training and Evaluating the Agent

To train the agent, simply run all the cells within the notebook. This will start the training loop and log metrics.

During training, metrics such as episode number, steps taken, loss, and epsilon (exploration rate) are printed. These provide insight into the agent’s learning process.

## Results and Analysis

The training output displays key metrics:

- **Episode:** Indicates the current episode out of the total (e.g. "Episode 1/1000").
- **Steps:** The number of actions taken in the episode.
- **Loss:** The network’s loss value, which indicates the error between predicted and target Q-values.
- **Epsilon:** The exploration rate, which decays over time.

For example, a typical log line reads:

```
Episode 397/1000, Steps: 4, Loss: 0.0215, Epsilon: 0.1367
```

This suggests that by episode 387 the agent took four steps with a loss of 0.0215 while the exploration rate was approximately 0.1367. At the end of training, an average reward and success rate are computed to summarise performance.

You could visualise these metrics (e.g. using Matplotlib or TensorBoard) to obtain a clearer picture of the learning process.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. [Link](http://incompleteideas.net/book/the-book.html)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. [Link](https://arxiv.org/abs/1512.03385)
- PyTorch Documentation. [Link](https://pytorch.org/docs/stable/index.html)
- Matplotlib Documentation. [Link](https://matplotlib.org/stable/contents.html)

---

By following this repository, you will witness how procedural tasks can be learned via reinforcement learning, a key area of research in human-AI interaction. This project serves as both a practical exercise and a foundation for future research in my PhD.
