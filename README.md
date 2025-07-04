# PPO-LunarLander-v3

## 📝 Overview
This project implements Proximal Policy Optimization (PPO) — one of the most powerful and stable policy gradient algorithms in Reinforcement Learning — entirely from scratch in PyTorch. The environment used for training is LunarLander-v3 from the OpenAI Gymnasium suite.

The implementation features a clean, modular, research-grade PPO pipeline with extensive modern improvements, along with visualization tools for analyzing training progress.

It is designed as an educational yet robust framework to train, evaluate, and visualize PPO on classic control problems, which can also be adapted to more complex environments.

✅ End-to-End PPO Pipeline:

- Defines a PPO agent with separate actor-critic architecture using a shared feature extractor.

- Uses Generalized Advantage Estimation (GAE) for computing advantages.

- Trains with clipped surrogate objective, value loss with optional clipping, and an entropy bonus to encourage exploration.

- Includes early stopping based on KL divergence for stable training.

✅ Advanced Features:

- Orthogonal initialization for stable learning.

- Layer normalization in the network to improve learning dynamics.

- Learning rate scheduling (linearly decaying) and gradient clipping.

- Adaptive policy updates with early stopping if KL divergence exceeds a threshold.

- Tracks and logs various training statistics including policy loss, value loss, entropy, KL divergence, clipping fraction, and explained variance.

### Benefits of PPO

✅ Stable learning due to conservative policy updates via clipping.

✅ Sample efficiency — reuses collected data for multiple gradient steps with small policy updates.

✅ Wide applicability — works well for both discrete and continuous action spaces.

✅ Simple to tune and robust, unlike older algorithms (e.g., vanilla policy gradients which suffer from large variance).

