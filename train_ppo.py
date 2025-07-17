import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time
from typing import Tuple, List

# Set device and reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

class PPONetwork(nn.Module):
    """
    State-of-the-art PPO network with shared backbone and separate heads.
    Includes proper initialization, layer normalization, and orthogonal weights.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(PPONetwork, self).__init__()
        
        # Shared backbone with layer normalization
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for policy head (smaller std)
        for module in self.policy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                
        # Special initialization for value head
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both policy logits and value"""
        features = self.backbone(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value.squeeze(-1)
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """Get action, log probability, entropy, and value"""
        logits, value = self.forward(state)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

class PPOBuffer:
    """
    Advanced PPO buffer with GAE (Generalized Advantage Estimation) computation
    """
    def __init__(self, size: int, state_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0
        self.path_start_idx = 0
        
        # Buffers
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a single transition"""
        assert self.ptr < self.size
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0):
        """
        Compute GAE advantages and returns for the last trajectory.
        Call this at the end of each episode or when buffer is full.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute GAE advantages
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Compute returns
        self.returns[path_slice] = self.advantages[path_slice] + self.values[path_slice]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """Get all data and normalize advantages"""
        assert self.ptr == self.size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages for better training stability
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            log_probs=self.log_probs,
            values=self.values
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}
    
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum (magic from rllab)"""
        return np.array([np.sum(discount**np.arange(len(x)-i) * x[i:]) for i in range(len(x))])

class PPOAgent:
    """
    State-of-the-art PPO agent with all modern improvements:
    - GAE for advantage estimation
    - Clipped objective with entropy bonus
    - Value function clipping
    - Adaptive KL penalty
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping based on KL divergence
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
        hidden_size: int = 256
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Networks and optimizer
        self.network = PPONetwork(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )
        
        # Metrics tracking
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clipfrac': [],
            'explained_variance': []
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """Get action from current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = self.network(state_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.zeros(1)
            else:
                probs = Categorical(logits=logits)
                action = probs.sample()
                log_prob = probs.log_prob(action)
            
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def update(self, buffer: PPOBuffer, update_epochs: int = 10, batch_size: int = 64):
        """Update policy using PPO objective"""
        data = buffer.get()
        
        states = data['states']
        actions = data['actions'].long()
        returns = data['returns']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        old_values = data['values']
        
        # Training loop
        for epoch in range(update_epochs):
            # Create minibatches
            batch_indices = np.arange(len(states))
            np.random.shuffle(batch_indices)
            
            epoch_policy_loss = []
            epoch_value_loss = []
            epoch_entropy_loss = []
            epoch_kl_div = []
            epoch_clipfrac = []
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = batch_indices[start:end]
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    states[batch_idx], actions[batch_idx]
                )
                
                # Compute ratios and clipped objective
                ratios = torch.exp(new_log_probs - old_log_probs[batch_idx])
                
                # Policy loss with clipping
                surr1 = ratios * advantages[batch_idx]
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with optional clipping
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - returns[batch_idx]) ** 2
                    v_clipped = old_values[batch_idx] + torch.clamp(
                        new_values - old_values[batch_idx], -self.clip_ratio, self.clip_ratio
                    )
                    v_loss_clipped = (v_clipped - returns[batch_idx]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values - returns[batch_idx]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    kl_div = ((old_log_probs[batch_idx] - new_log_probs) ** 2).mean()
                    clipfrac = ((ratios - 1.0).abs() > self.clip_ratio).float().mean()
                    
                    epoch_policy_loss.append(policy_loss.item())
                    epoch_value_loss.append(value_loss.item())
                    epoch_entropy_loss.append(entropy_loss.item())
                    epoch_kl_div.append(kl_div.item())
                    epoch_clipfrac.append(clipfrac.item())
            
            # Early stopping based on KL divergence
            avg_kl = np.mean(epoch_kl_div)
            if avg_kl > 1.5 * self.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching max KL: {avg_kl:.4f}")
                break
        
        # Update learning rate
        self.scheduler.step()
        
        # Store metrics
        self.training_metrics['policy_loss'].append(np.mean(epoch_policy_loss))
        self.training_metrics['value_loss'].append(np.mean(epoch_value_loss))
        self.training_metrics['entropy_loss'].append(np.mean(epoch_entropy_loss))
        self.training_metrics['kl_divergence'].append(avg_kl)
        self.training_metrics['clipfrac'].append(np.mean(epoch_clipfrac))
        
        # Explained variance
        y_true = returns.cpu().numpy()
        y_pred = old_values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / var_y if var_y > 0 else 0
        self.training_metrics['explained_variance'].append(explained_var)
        
        return {
            'policy_loss': np.mean(epoch_policy_loss),
            'value_loss': np.mean(epoch_value_loss),
            'entropy_loss': np.mean(epoch_entropy_loss),
            'kl_divergence': avg_kl,
            'clipfrac': np.mean(epoch_clipfrac),
            'explained_variance': explained_var,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

def train_ppo(
    n_episodes: int = 1000,
    steps_per_episode: int = 2048,
    update_epochs: int = 10,
    batch_size: int = 64,
    save_freq: int = 100,
    eval_freq: int = 50
):
    env = gym.make('LunarLander-v3')
    eval_env = gym.make('LunarLander-v3')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print("="*50)
    
    agent = PPOAgent(state_size, action_size)
    buffer = PPOBuffer(steps_per_episode, state_size)
    
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    best_eval_reward = -np.inf
    
    start_time = time.time()
    
    for step in range(n_episodes * steps_per_episode):
        
        action, log_prob, value = agent.get_action(state)
        
        # Take environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        buffer.store(state, action, reward, value, log_prob, done)
        
        # Update tracking
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # Handle episode end
        if done:
            buffer.finish_path(0)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Update policy when buffer is full
        if (step + 1) % steps_per_episode == 0:
            if not done:
                # Bootstrap value for incomplete episode
                _, _, last_value = agent.get_action(state)
                buffer.finish_path(last_value)
            
            # Update agent
            update_info = agent.update(buffer, update_epochs, batch_size)
            
            # Logging
            recent_rewards = episode_rewards[-10:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            
            print(f"Update {step // steps_per_episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Length: {np.mean(episode_lengths[-10:]):5.1f} | "
                  f"KL: {update_info['kl_divergence']:.4f} | "
                  f"LR: {update_info['learning_rate']:.2e}")
        
        # Evaluation
        if (step + 1) % (eval_freq * steps_per_episode) == 0:
            eval_reward = evaluate_agent(agent, eval_env, n_episodes=10)
            eval_rewards.append(eval_reward)
            
            print(f"Evaluation: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(agent.network.state_dict(), 'ppo_lunar_lander_best.pth')
                print(f"New best model saved! Eval reward: {eval_reward:.2f}")
        
        # Regular model saving
        if (step + 1) % (save_freq * steps_per_episode) == 0:
            torch.save(agent.network.state_dict(), f'ppo_lunar_lander_step_{step+1}.pth')
    
    # Final evaluation and cleanup
    final_eval_reward = evaluate_agent(agent, eval_env, n_episodes=100)
    print(f"\nFinal evaluation (100 episodes): {final_eval_reward:.2f}")
    
    # Save final model
    torch.save(agent.network.state_dict(), 'ppo_lunar_lander_final.pth')
    
    env.close()
    eval_env.close()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return agent, episode_rewards, eval_rewards 

def evaluate_agent(agent: PPOAgent, env, n_episodes: int = 10, render: bool = False):
    """Evaluate agent performance"""
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

def plot_training_results(episode_rewards: List[float], eval_rewards: List[float], save_path: str = 'ppo_training_results.png'):
    """Plot comprehensive training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
    if len(episode_rewards) > 100:
        smoothed = [np.mean(episode_rewards[max(0, i-100):i+1]) for i in range(len(episode_rewards))]
        ax1.plot(smoothed, label='Moving Average (100 episodes)')
    ax1.set_title('Training Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Evaluation rewards
    if eval_rewards:
        ax2.plot(eval_rewards, 'o-', label='Evaluation Rewards')
        ax2.axhline(y=200, color='r', linestyle='--', label='Solved Threshold')
        ax2.set_title('Evaluation Performance')
        ax2.set_xlabel('Evaluation Round')
        ax2.set_ylabel('Average Reward')
        ax2.legend()
        ax2.grid(True)
    
    # Recent performance distribution
    if len(episode_rewards) > 100:
        recent_rewards = episode_rewards[-100:]
        ax3.hist(recent_rewards, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.mean(recent_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(recent_rewards):.1f}')
        ax3.axvline(x=200, color='g', linestyle='--', label='Solved: 200')
        ax3.set_title('Recent Performance Distribution (Last 100 Episodes)')
        ax3.set_xlabel('Episode Reward')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
    
    # Training statistics
    if len(episode_rewards) > 0:
        window_size = min(100, len(episode_rewards))
        rolling_mean = [np.mean(episode_rewards[max(0, i-window_size):i+1]) for i in range(len(episode_rewards))]
        rolling_std = [np.std(episode_rewards[max(0, i-window_size):i+1]) for i in range(len(episode_rewards))]
        
        ax4.plot(rolling_mean, label='Rolling Mean')
        ax4.fill_between(range(len(rolling_mean)), 
                        np.array(rolling_mean) - np.array(rolling_std),
                        np.array(rolling_mean) + np.array(rolling_std),
                        alpha=0.3, label='±1 Std Dev')
        ax4.axhline(y=200, color='r', linestyle='--', label='Solved Threshold')
        ax4.set_title('Training Stability')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def load_and_test_agent(model_path: str, n_episodes: int = 10, render: bool = False):
    """Load and test a saved PPO agent"""
    env = gym.make('LunarLander-v3', render_mode='human' if render else None)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = PPOAgent(state_size, action_size)
    
    if os.path.exists(model_path):
        agent.network.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
        
        eval_reward = evaluate_agent(agent, env, n_episodes, render)
        print(f"Average reward over {n_episodes} episodes: {eval_reward:.2f}")
        
        env.close()
        return agent, eval_reward
    else:
        print(f"Model file {model_path} not found!")
        env.close()
        return None, None
    
def do_training():
    agent, episode_rewards, eval_rewards = train_ppo(
        n_episodes=1000,
        steps_per_episode=2048,
        update_epochs=10,
        batch_size=64,
        eval_freq=25,
    )
    
    plot_training_results(episode_rewards, eval_rewards)
    
    print(f"\nTraining Summary:")
    print(f"Episodes trained: {len(episode_rewards)}")
    print(f"Final 100-episode average: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best evaluation score: {max(eval_rewards) if eval_rewards else 'N/A':.2f}")

if __name__ == "__main__":
    # do_training()
    load_and_test_agent('ppo_lunar_lander_final.pth', n_episodes=10, render=True)