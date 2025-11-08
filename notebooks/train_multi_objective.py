import sys
import os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'src')))

from multi_objective_environment import MultiObjectiveBatteryEnv
from stable_baselines3 import PPO
import numpy as np

print("=== MULTI-OBJECTIVE AI TRAINING ===")
print("Training AI to balance: PROFIT vs FREQUENCY vs CONGESTION")

# Test different weight combinations
weight_configs = [
    {'profit_weight': 1.0, 'frequency_weight': 0.3, 'congestion_weight': 0.2},  # Balanced
    {'profit_weight': 1.0, 'frequency_weight': 0.0, 'congestion_weight': 0.0},  # Profit-only
    {'profit_weight': 0.5, 'frequency_weight': 0.5, 'congestion_weight': 0.5},  # Grid-focused
]

for i, weights in enumerate(weight_configs):
    print(f"\n--- Training Config {i+1}: {weights} ---")
    
    env = MultiObjectiveBatteryEnv(
        profit_weight=weights['profit_weight'],
        frequency_weight=weights['frequency_weight'], 
        congestion_weight=weights['congestion_weight']
    )
    
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=10000)
    
    # Test the trained AI
    test_env = MultiObjectiveBatteryEnv(
        profit_weight=weights['profit_weight'],
        frequency_weight=weights['frequency_weight'],
        congestion_weight=weights['congestion_weight']
    )
    
    state, info = test_env.reset()
    total_reward = 0
    total_profit = 0
    
    print("AI Trading Session (first 12 hours):")
    for step in range(24):
        action, _states = model.predict(state, deterministic=True)
        action = int(action)
        state, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        
        # Track actual profit (separate from multi-objective reward)
        current_price = test_env.market.get_price_at_hour(step)
        if action == 1:  # Charge
            total_profit -= current_price * 50
        elif action == 2:  # Discharge  
            total_profit += current_price * 50
        
        action_names = {0: "HOLD", 1: "CHARGE", 2: "DISCHARGE"}
        if step < 12:  # Only show first 12 hours to avoid clutter
            grid_info = f"Price:£{current_price}, Freq:{50+test_env.market.get_frequency_at_hour(step):.2f}Hz"
            print(f"  Hour {step}: {action_names[action]:<10} | Reward: {reward:6.2f} | {grid_info}")
    
    print(f"Final Multi-Objective Reward: {total_reward:.2f}")
    print(f"Final Economic Profit: £{total_profit:,.0f}")
    print(f"Final Battery SOC: {test_env.battery.soc:.2f}")

print("\n=== COMPARISON COMPLETE ===")
print("Different weight configurations produce different trading behaviors!")
print("This demonstrates the multi-objective trade-offs in grid operation.")