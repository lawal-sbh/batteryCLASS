"""
src/training/train_hierarchical.py
Train hierarchical Commander-Tactician agent on real UK grid data

Usage:
    python src/training/train_hierarchical.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from collections import deque
import random

# ============================================
# NETWORK ARCHITECTURES
# ============================================

class CommanderNetwork(nn.Sequential):
    """Day-ahead strategic planning: 5D → 48D"""
    def __init__(self, state_dim=5, action_dim=48, hidden_dim=256):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

class ValueNetwork(nn.Sequential):
    """Value network for advantage estimation"""
    def __init__(self, state_dim=5, hidden_dim=256):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

class TacticianNetwork(nn.Sequential):
    """Real-time execution: 5D → 2D"""
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=128):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

# ============================================
# BATTERY ENVIRONMENT
# ============================================

class BatteryEnvironment:
    """Dual battery environment with realistic UK grid physics"""
    def __init__(self, episode_data, norm_params):
        self.episode_data = episode_data
        self.norm_params = norm_params
        self.capacity = 5.0  # MWh per battery
        self.power_limit = 1.0  # MW per battery
        self.efficiency = 0.95
        self.degradation_cost = 0.01  # £/MWh
        
        self.step_idx = 0
        self.soc = [0.5, 0.5]
        
    def reset(self):
        self.step_idx = 0
        self.soc = [0.5, 0.5]
        return self._get_state()
    
    def _get_state(self):
        """Get current normalized state"""
        if self.step_idx >= len(self.episode_data):
            return None
        
        row = self.episode_data.iloc[self.step_idx]
        
        # Normalize price using z-score
        price_norm = (row['system_price'] - self.norm_params['price_mean']) / self.norm_params['price_std']
        price_norm = np.clip(price_norm, -3, 3) / 3  # Clip to [-1, 1]
        
        state = np.array([
            row['hour'] / 24.0,
            price_norm,
            self.soc[0],
            self.soc[1],
            row['grid_stress']
        ], dtype=np.float32)
        
        return state
    
    def step(self, actions):
        """Execute actions and return next state, reward, done"""
        if self.step_idx >= len(self.episode_data):
            return None, 0, True, {}
        
        row = self.episode_data.iloc[self.step_idx]
        price = row['system_price']
        grid_stress = row['grid_stress']
        
        total_reward = 0
        violations = 0
        
        for i, action in enumerate(actions):
            # Scale action
            power = action * self.power_limit
            energy = power * 0.5  # Half-hour
            
            # Apply efficiency
            if energy > 0:
                energy_actual = energy * self.efficiency
            else:
                energy_actual = energy / self.efficiency
            
            # Update SOC
            new_soc = self.soc[i] + (energy_actual / self.capacity)
            new_soc = np.clip(new_soc, 0.0, 1.0)
            
            # Check violations
            violated = (new_soc < 0.05 or new_soc > 0.95)
            if violated:
                violations += 1
            
            # Calculate reward
            revenue = -energy * price  # Negative because we buy when charging
            degradation = abs(energy) * self.degradation_cost
            grid_support = 0
            if grid_stress > 0.7 and energy < 0:
                grid_support = abs(energy) * 5  # Grid support bonus
            
            reward = revenue - degradation + grid_support
            
            # Penalty for violations
            if violated:
                reward -= 50
            
            # Update state
            self.soc[i] = new_soc
            total_reward += reward
        
        self.step_idx += 1
        next_state = self._get_state()
        done = (self.step_idx >= len(self.episode_data))
        
        info = {
            'violations': violations,
            'price': price,
            'soc': self.soc.copy()
        }
        
        return next_state, total_reward, done, info

# ============================================
# REPLAY BUFFER
# ============================================

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array([s if s is not None else np.zeros(5) for s in next_states]),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============================================
# TRAINING AGENT
# ============================================

class HierarchicalTrainer:
    """Trains Commander and Tactician networks"""
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Initialize networks
        self.tactician = TacticianNetwork().to(self.device)
        self.tactician_value = ValueNetwork(state_dim=5, hidden_dim=128).to(self.device)
        
        self.commander = CommanderNetwork().to(self.device)
        self.commander_value = ValueNetwork().to(self.device)
        
        # Optimizers
        self.tactician_optimizer = optim.Adam(self.tactician.parameters(), lr=3e-4)
        self.tactician_value_optimizer = optim.Adam(self.tactician_value.parameters(), lr=3e-4)
        
        self.commander_optimizer = optim.Adam(self.commander.parameters(), lr=1e-4)
        self.commander_value_optimizer = optim.Adam(self.commander_value.parameters(), lr=1e-4)
        
        # Replay buffers
        self.tactician_buffer = ReplayBuffer(capacity=50000)
        self.commander_buffer = ReplayBuffer(capacity=10000)
        
        print(f"✓ Trainer initialized on {self.device}")
    
    def select_action(self, state, network, explore=True, epsilon=0.1):
        """Select action with exploration"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = network(state_tensor).cpu().numpy()[0]
            
            if explore and random.random() < epsilon:
                # Add exploration noise
                noise = np.random.normal(0, 0.2, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            return action
    
    def train_tactician_step(self, batch_size=128):
        """Train tactician on one batch"""
        if len(self.tactician_buffer) < batch_size:
            return 0, 0
        
        states, actions, rewards, next_states, dones = self.tactician_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Train value network
        values = self.tactician_value(states)
        next_values = self.tactician_value(next_states).detach()
        target_values = rewards + 0.99 * next_values * (1 - dones)
        value_loss = nn.MSELoss()(values, target_values)
        
        self.tactician_value_optimizer.zero_grad()
        value_loss.backward()
        self.tactician_value_optimizer.step()
        
        # Train policy network
        predicted_actions = self.tactician(states)
        advantages = (target_values - values).detach()
        
        # Policy loss: want to maximize advantage-weighted actions
        policy_loss = -(predicted_actions * advantages).mean()
        action_loss = nn.MSELoss()(predicted_actions, actions)  # Also match actions
        
        total_policy_loss = policy_loss + 0.5 * action_loss
        
        self.tactician_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.tactician.parameters(), 1.0)
        self.tactician_optimizer.step()
        
        return value_loss.item(), total_policy_loss.item()
    
    def train_commander_step(self, batch_size=64):
        """Train commander on aggregated episode data"""
        if len(self.commander_buffer) < batch_size:
            return 0, 0
        
        states, actions, rewards, next_states, dones = self.commander_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Train value network
        values = self.commander_value(states)
        next_values = self.commander_value(next_states).detach()
        target_values = rewards + 0.99 * next_values * (1 - dones)
        value_loss = nn.MSELoss()(values, target_values)
        
        self.commander_value_optimizer.zero_grad()
        value_loss.backward()
        self.commander_value_optimizer.step()
        
        # Train policy network
        predicted_actions = self.commander(states)
        advantages = (target_values - values).detach()
        policy_loss = -(predicted_actions.mean() * advantages).mean()
        
        self.commander_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.commander.parameters(), 1.0)
        self.commander_optimizer.step()
        
        return value_loss.item(), policy_loss.item()
    
    def save_checkpoint(self, episode, save_dir='models_retrained'):
        """Save model checkpoints"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Commander
        commander_path = Path(save_dir) / 'commander'
        commander_path.mkdir(exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_state_dict': self.commander.state_dict(),
            'value_state_dict': self.commander_value.state_dict(),
            'optimizer_state_dict': self.commander_optimizer.state_dict(),
            'state_size': 5,
            'action_size': 48
        }, commander_path / f'commander_episode_{episode}.pth')
        
        # Tactician
        tactician_path = Path(save_dir) / 'tactician'
        tactician_path.mkdir(exist_ok=True)
        torch.save({
            'episode': episode,
            'policy_state_dict': self.tactician.state_dict(),
            'value_state_dict': self.tactician_value.state_dict(),
            'optimizer_state_dict': self.tactician_optimizer.state_dict(),
            'state_size': 5,
            'action_size': 2
        }, tactician_path / f'tactician_episode_{episode}.pth')
        
        print(f"  ✓ Checkpoint saved at episode {episode}")

# ============================================
# MAIN TRAINING LOOP
# ============================================

def train():
    print("="*70)
    print("TRAINING HIERARCHICAL AGENT ON REAL UK DATA")
    print("="*70)
    
    # Load data
    train_data = pd.read_csv('data/training/train.csv')
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    
    with open('data/training/normalization_params.json') as f:
        norm_params = json.load(f)
    
    # Create episodes
    episodes = []
    for date in train_data['datetime'].dt.date.unique():
        episode = train_data[train_data['datetime'].dt.date == date].copy()
        if len(episode) == 48:
            episodes.append(episode.reset_index(drop=True))
    
    print(f"\n✓ Loaded {len(episodes)} training episodes")
    print(f"✓ Price normalization: mean=£{norm_params['price_mean']:.2f}, std=£{norm_params['price_std']:.2f}")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = HierarchicalTrainer(device=device)
    
    # Training parameters
    num_epochs = 10  # Go through all episodes 10 times
    batch_size = 128
    save_every = 100
    
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION:")
    print(f"{'='*70}")
    print(f"Episodes per epoch:  {len(episodes)}")
    print(f"Total epochs:        {num_epochs}")
    print(f"Batch size:          {batch_size}")
    print(f"Device:              {device}")
    print(f"{'='*70}\n")
    
    episode_count = 0
    best_reward = -float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Shuffle episodes
        random.shuffle(episodes)
        
        epoch_rewards = []
        epoch_violations = []
        
        for ep_idx, episode_data in enumerate(episodes):
            # Create environment
            env = BatteryEnvironment(episode_data, norm_params)
            state = env.reset()
            
            episode_reward = 0
            episode_violations = 0
            
            # Exploration rate (decay over time)
            epsilon = max(0.1, 1.0 - (episode_count / (len(episodes) * num_epochs)))
            
            # Run episode
            done = False
            while not done:
                # Get action from tactician
                action = trainer.select_action(state, trainer.tactician, explore=True, epsilon=epsilon)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                trainer.tactician_buffer.push(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_violations += info['violations']
                
                state = next_state
                
                if next_state is None:
                    break
            
            # Train networks
            if len(trainer.tactician_buffer) >= batch_size:
                for _ in range(4):  # Multiple updates per episode
                    value_loss, policy_loss = trainer.train_tactician_step(batch_size)
            
            # Store commander experience (aggregated episode state)
            # This is simplified - commander learns from episode outcomes
            if episode_reward > -1000:  # Only if episode wasn't terrible
                initial_state = episodes[0].iloc[0]
                state_vec = np.array([
                    12.0 / 24.0,  # Average hour
                    (episode_data['system_price'].mean() - norm_params['price_mean']) / norm_params['price_std'] / 3,
                    0.5, 0.5,  # Initial SOC
                    episode_data['grid_stress'].mean()
                ], dtype=np.float32)
                
                trainer.commander_buffer.push(
                    state_vec,
                    np.zeros(48),  # Placeholder
                    episode_reward / 48,  # Average reward per step
                    state_vec,
                    True
                )
                
                if len(trainer.commander_buffer) >= 64:
                    for _ in range(2):
                        trainer.train_commander_step(64)
            
            epoch_rewards.append(episode_reward)
            epoch_violations.append(episode_violations)
            episode_count += 1
            
            # Print progress
            if (ep_idx + 1) % 50 == 0:
                avg_reward = np.mean(epoch_rewards[-50:])
                avg_violations = np.mean(epoch_violations[-50:])
                print(f"  Episode {ep_idx+1}/{len(episodes)}: "
                      f"Avg Reward=£{avg_reward:.2f}, "
                      f"Avg Violations={avg_violations:.1f}, "
                      f"Epsilon={epsilon:.3f}")
            
            # Save checkpoint
            if episode_count % save_every == 0:
                trainer.save_checkpoint(episode_count)
        
        # Epoch summary
        avg_epoch_reward = np.mean(epoch_rewards)
        avg_epoch_violations = np.mean(epoch_violations)
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1} SUMMARY:")
        print(f"{'='*70}")
        print(f"Avg Reward:     £{avg_epoch_reward:.2f}")
        print(f"Avg Violations: {avg_epoch_violations:.2f}")
        print(f"Best Episode:   £{max(epoch_rewards):.2f}")
        print(f"Worst Episode:  £{min(epoch_rewards):.2f}")
        
        # Save if best
        if avg_epoch_reward > best_reward:
            best_reward = avg_epoch_reward
            trainer.save_checkpoint(f'best_epoch{epoch+1}')
            print(f"✓ New best model saved!")
    
    # Save final
    trainer.save_checkpoint('final')
    
    print(f"\n{'='*70}")
    print("✓✓✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best average reward: £{best_reward:.2f}")
    print(f"Models saved to: models_retrained/")

if __name__ == "__main__":
    train()