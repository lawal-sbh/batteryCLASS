import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_environment import BatteryTradingEnv
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

print("=== TRAINING FIRST AI TRADING AGENT ===")

# Create environment
env = BatteryTradingEnv()

# Test random agent performance (baseline)
print("Testing random agent...")
total_rewards = []
for episode in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = np.random.randint(0, 3)  # Random action
        state, reward, done, _ = env.step(action)
        total_reward += reward
    
    total_rewards.append(total_reward)
    print(f"Episode {episode}: £{total_reward:,.0f}")

print(f"Random agent average: £{np.mean(total_rewards):,.0f}")
print(f"Your smarter strategy: £4,693")
print(f"Perfect foresight: £11,700")
print()

print("Now training AI agent... (this will take a few minutes)")
print("The AI will learn to beat both random trading AND your human strategy!")

# Create a proper Gym environment
class GymBatteryEnv(gym.Env):
    def __init__(self):
        self.env = BatteryTradingEnv()
        self.observation_space = gym.spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0=HOLD, 1=CHARGE, 2=DISCHARGE
    
    def reset(self):
        state = self.env.reset()
        return np.array(state, dtype=np.float32), {}
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.array(state, dtype=np.float32), reward, done, False, info

# Train the AI
gym_env = GymBatteryEnv()
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=50000)  # Train for 50,000 steps

print("Training complete! Testing AI performance...")

# Test the trained AI
ai_rewards = []
for episode in range(5):
    state, _ = gym_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(state)
        state, reward, done, _, _ = gym_env.step(action)
        total_reward += reward
    
    ai_rewards.append(total_reward)
    print(f"AI Episode {episode}: £{total_reward:,.0f}")

print(f"\n=== RESULTS ===")
print(f"Random agent: £{np.mean(total_rewards):,.0f}")
print(f"Your strategy: £4,693")
print(f"AI agent: £{np.mean(ai_rewards):,.0f}")
print(f"Perfect foresight: £11,700")

if np.mean(ai_rewards) > 4693:
    print("🎉 AI BEAT HUMAN STRATEGY!")
else:
    print("AI needs more training...")