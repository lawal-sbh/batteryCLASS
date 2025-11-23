# notebooks/train_hierarchical.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
from datetime import datetime

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv
from src.settlement_engine import SettlementEngine

class SimpleRLAgent:
    """Simple neural network policy for RL"""
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()  # Output in [-1, 1] for continuous actions
        )
        
        # Value network (for PPO)
        self.value_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + 
                                   list(self.value_net.parameters()), lr=3e-4)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
    def get_action(self, state, exploration=True):
        """Get action from policy with optional exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean = self.policy_net(state_tensor).numpy()[0]
        
        if exploration:
            # Add noise for exploration
            noise = np.random.normal(0, 0.1, size=self.action_size)
            action = np.clip(action_mean + noise, -1, 1)
        else:
            action = action_mean
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on past experiences"""
        if len(self.memory) < self.batch_size:
            return 0
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Simple policy gradient update
        action_probs = self.policy_net(states)
        values = self.value_net(states).squeeze()
        
        # Calculate advantages (simplified)
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            targets = rewards + 0.99 * next_values * (1 - dones)
        
        advantages = targets - values
        
        # Policy loss
        policy_loss = -(action_probs * advantages.unsqueeze(1)).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values, targets)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

class CommanderAgent(SimpleRLAgent):
    """Day-ahead planning agent"""
    def __init__(self, state_size=5, action_size=48):  # 24h * 2 batteries
        super().__init__(state_size, action_size, hidden_size=256)
        self.agent_type = "commander"
    
    def plan_soc_trajectory(self, state):
        """Generate SOC targets for next 24 hours"""
        action = self.get_action(state, exploration=False)
        
        # Reshape to 24 hours x 2 batteries
        soc_targets = action.reshape(24, 2)
        
        # Ensure SOC targets are valid [0, 1]
        soc_targets = (soc_targets + 1) / 2  # Convert [-1,1] to [0,1]
        soc_targets = np.clip(soc_targets, 0.1, 0.9)  # Keep safe margins
        
        return soc_targets.tolist()

class TacticianAgent(SimpleRLAgent):
    """Real-time execution agent"""
    def __init__(self, state_size=5, action_size=2):  # 2 batteries
        super().__init__(state_size, action_size, hidden_size=128)
        self.agent_type = "tactician"

def save_agent(agent, episode, models_dir="models"):
    """Save trained agent to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create agent directory if it doesn't exist
    agent_dir = os.path.join(models_dir, agent.agent_type)
    os.makedirs(agent_dir, exist_ok=True)
    
    # Save model
    filename = f"{agent.agent_type}_episode_{episode}_{timestamp}.pth"
    filepath = os.path.join(agent_dir, filename)
    
    torch.save({
        'episode': episode,
        'policy_state_dict': agent.policy_net.state_dict(),
        'value_state_dict': agent.value_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'state_size': agent.state_size,
        'action_size': agent.action_size
    }, filepath)
    
    print(f"✓ Saved {agent.agent_type} to {filepath}")
    return filepath

def load_agent(agent_type, episode, models_dir="models"):
    """Load trained agent from file"""
    agent_dir = os.path.join(models_dir, agent_type)
    
    # Find the latest checkpoint for this episode
    checkpoints = [f for f in os.listdir(agent_dir) 
                  if f.startswith(f"{agent_type}_episode_{episode}")]
    
    if not checkpoints:
        print(f"⚠️ No checkpoint found for {agent_type} episode {episode}")
        return None
    
    latest_checkpoint = sorted(checkpoints)[-1]
    filepath = os.path.join(agent_dir, latest_checkpoint)
    
    checkpoint = torch.load(filepath)
    
    # Create agent
    if agent_type == "commander":
        agent = CommanderAgent(checkpoint['state_size'], checkpoint['action_size'])
    else:
        agent = TacticianAgent(checkpoint['state_size'], checkpoint['action_size'])
    
    # Load state
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.value_net.load_state_dict(checkpoint['value_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded {agent_type} from {filepath}")
    return agent

def train_hierarchical_rl():
    """Main training function for hierarchical RL"""
    print("=== HIERARCHICAL RL TRAINING ===")
    print("Training Commander (Day-Ahead) + Tactician (Real-Time)\n")
    
    # Initialize components
    env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'], use_calibrated_data=True)
    settlement = SettlementEngine()
    
    # Initialize agents
    commander = CommanderAgent(state_size=5, action_size=48)  # 24h * 2 batteries
    tactician = TacticianAgent(state_size=5, action_size=2)   # 2 batteries
    
    # Training parameters
    total_episodes = 1000
    save_interval = 100
    print_interval = 50
    
    # Track performance
    episode_rewards = []
    episode_economic = []
    episode_stability = []
    
    print(f"Starting training for {total_episodes} episodes...")
    print("-" * 80)
    
    for episode in range(total_episodes):
        # Reset environment
        obs, info = env.reset()
        
        # PHASE 1: Commander makes day-ahead plan
        commander_plan = commander.plan_soc_trajectory(obs)
        env.set_commander_target([target[0] for target in commander_plan])  # Set SOC targets
        
        # PHASE 2: Tactician executes in real-time
        episode_reward = 0
        episode_economic_reward = 0
        episode_stability_reward = 0
        
        step_count = 0
        for step in range(24):  # 24 hours
            # Tactician decides real-time action
            action = tactician.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store experience for tactician
            tactician.remember(obs, action, reward, next_obs, done)
            
            # Update rewards
            episode_reward += reward
            episode_economic_reward += info['economic']
            episode_stability_reward += info['stability']
            
            # Train tactician
            if step_count % 4 == 0:  # Train every 4 steps
                tactician.replay()
            
            obs = next_obs
            step_count += 1
            
            if done:
                break
        
        # PHASE 3: Settlement and learning
        episode_performance = {
            'economic': episode_economic_reward,
            'stability': episode_stability_reward,
            'total_reward': episode_reward
        }
        
        settlement_data = settlement.record_episode(
            {'expected_value': episode_economic_reward, 'strategy_type': 'adaptive'},
            episode_performance,
            stability_impact=episode_stability_reward
        )
        
        commander_feedback = settlement.calculate_commander_feedback()
        
        # Store performance
        episode_rewards.append(episode_reward)
        episode_economic.append(episode_economic_reward)
        episode_stability.append(episode_stability_reward)
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_economic = np.mean(episode_economic[-print_interval:])
            avg_stability = np.mean(episode_stability[-print_interval:])
            
            print(f"Episode {episode+1:4d}/{total_episodes} | "
                  f"Avg Reward: £{avg_reward:7.1f} | "
                  f"Economic: £{avg_economic:7.1f} | "
                  f"Stability: £{avg_stability:7.1f} | "
                  f"Update: {commander_feedback['strategy_update']}")
        
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            save_agent(commander, episode + 1)
            save_agent(tactician, episode + 1)
            
            # Save training progress
            training_progress = {
                'episode': episode + 1,
                'rewards': episode_rewards,
                'economic_rewards': episode_economic,
                'stability_rewards': episode_stability,
                'timestamp': datetime.now().isoformat()
            }
            
            progress_file = f"models/training_progress_episode_{episode+1}.json"
            with open(progress_file, 'w') as f:
                json.dump(training_progress, f, indent=2)
            
            print(f"💾 Saved training progress to {progress_file}")
    
    print("-" * 80)
    print("🎉 TRAINING COMPLETE!")
    
    # Final performance analysis
    final_avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
    final_avg_economic = np.mean(episode_economic[-100:])
    final_avg_stability = np.mean(episode_stability[-100:])
    
    print(f"\n📊 FINAL PERFORMANCE (last 100 episodes):")
    print(f"   Average Total Reward: £{final_avg_reward:.1f}")
    print(f"   Average Economic: £{final_avg_economic:.1f}")
    print(f"   Average Stability: £{final_avg_stability:.1f}")
    
    # Save final models
    save_agent(commander, total_episodes)
    save_agent(tactician, total_episodes)
    
    return commander, tactician, episode_rewards

def demonstrate_trained_agents(commander, tactician):
    """Demonstrate trained agents in action"""
    print("\n" + "="*60)
    print("DEMONSTRATING TRAINED AGENTS")
    print("="*60)
    
    env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'], use_calibrated_data=True)
    obs, info = env.reset()
    
    print("Running 24-hour demonstration with trained agents...")
    
    total_reward = 0
    hourly_actions = []
    
    # Commander makes plan
    soc_targets = commander.plan_soc_trajectory(obs)
    env.set_commander_target([target[0] for target in soc_targets])
    
    for hour in range(24):
        # Tactician executes
        action = tactician.get_action(obs, exploration=False)  # No exploration
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        hourly_actions.append({
            'hour': hour,
            'action_scotland': action[0],
            'action_london': action[1],
            'soc_scotland': info['soc'][0],
            'soc_london': info['soc'][1],
            'price': info['price'],
            'economic': info['economic'],
            'stability': info['stability']
        })
        
        if hour in [0, 6, 12, 18, 23] or done:
            print(f"Hour {hour:2d}: "
                  f"Actions: [{action[0]:+5.2f}, {action[1]:+5.2f}] | "
                  f"SOC: [{info['soc'][0]:.2f}, {info['soc'][1]:.2f}] | "
                  f"Price: £{info['price']:5.1f}/MWh")
        
        if done:
            break
    
    print(f"\n📈 Demonstration Complete:")
    print(f"   Total Reward: £{total_reward:.1f}")
    print(f"   Final SOC: Scotland {info['soc'][0]:.2f}, London {info['soc'][1]:.2f}")
    
    return hourly_actions

if __name__ == "__main__":
    # Check if model directories exist
    if not os.path.exists("models/commander"):
        print("⚠️ Model directories not found. Run setup_model_directories.py first!")
        print("Creating directories automatically...")
        os.makedirs("models/commander", exist_ok=True)
        os.makedirs("models/tactician", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
    
    try:
        # Train the hierarchical agents
        commander, tactician, rewards = train_hierarchical_rl()
        
        # Demonstrate trained agents
        demonstration_data = demonstrate_trained_agents(commander, tactician)
        
        print(f"\n🎯 TRAINING SUCCESSFUL!")
        print(f"   Models saved in: models/commander/ and models/tactician/")
        print(f"   Ready for thesis demonstration and analysis")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()