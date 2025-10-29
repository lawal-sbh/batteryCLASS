import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_environment import BatteryTradingEnv
from stable_baselines3 import PPO
import numpy as np

print("=== PROPER GYM AI TRAINING ===")

# Create the environment
env = BatteryTradingEnv()
print("Environment created successfully!")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Test the environment
state, info = env.reset()
print("Initial state:", state)

print("\nStarting AI training...")

# Create and train the model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=20000)  # Train for 20k steps

print("Training complete! Testing AI...")

# Test the trained AI
print("Training complete! Testing AI...")

# Test the trained AI
test_env = BatteryTradingEnv()
state, info = test_env.reset()
total_reward = 0

print("\nAI Trading Session:")
for step in range(24):
    action, _states = model.predict(state, deterministic=True)
    action = int(action)  # ← FIX: Convert numpy array to integer
    state, reward, done, truncated, info = test_env.step(action)
    total_reward += reward
    
    # Convert back to actual revenue
    actual_revenue = total_reward * 1000  # Reverse scaling
    
    action_names = {0: "HOLD", 1: "CHARGE", 2: "DISCHARGE"}
    print(f"Hour {step}: {action_names[action]:<10} | Revenue: £{actual_revenue:,.0f}")

print(f"\n=== FINAL RESULTS ===")
print(f"AI Revenue: £{actual_revenue:,.0f}")
print(f"Your heuristic: £4,693")
print(f"Performance: {'✅ BEAT' if actual_revenue > 4693 else '❌ below'} human strategy")