import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_environment import HierarchicalBatteryEnv, DayAheadPlanner
from stable_baselines3 import PPO
import numpy as np

print("=== HIERARCHICAL RL TRAINING ===")
print("Training two-level: Day-Ahead Planner + Real-Time Executor")

# Test different planning strategies
planner = DayAheadPlanner()
planning_strategies = {
    'arbitrage': planner.create_schedule(
        [35, 32, 30, 28, 25, 24, 26, 35, 45, 50, 55, 60, 
         65, 70, 75, 80, 85, 90, 95, 100, 90, 75, 60, 45], {}
    ),
    'conservative': planner.create_conservative_schedule(),
    'aggressive': planner.create_aggressive_schedule()
}

results = {}

for strategy_name, schedule in planning_strategies.items():
    print(f"\n--- Testing {strategy_name.upper()} Day-Ahead Strategy ---")
    print(f"Schedule: {schedule}")
    
    env = HierarchicalBatteryEnv()
    env.set_day_ahead_plan(schedule)
    
    # Train real-time executor
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    model.learn(total_timesteps=15000)
    
    # Test performance
    test_env = HierarchicalBatteryEnv()
    test_env.set_day_ahead_plan(schedule)
    state, info = test_env.reset()
    
    total_reward = 0
    plan_deviations = []
    
    print("Real-time Execution:")
    for step in range(24):
        action, _states = model.predict(state, deterministic=True)
        action = int(action)
        state, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        
        # Track plan adherence
        plan_deviation = state[4]  # Deviation from planned SOC
        plan_deviations.append(abs(plan_deviation))
        
        action_names = {0: "HOLD", 1: "CHARGE", 2: "DISCHARGE"}
        if step % 6 == 0:
            print(f"  Hour {step}: {action_names[action]:<10} | "
                  f"Reward: {reward:6.2f} | SOC: {test_env.battery.soc:.2f} | "
                  f"Plan Dev: {plan_deviation:+.2f}")
    
    avg_plan_deviation = np.mean(plan_deviations)
    results[strategy_name] = {
        'total_reward': total_reward,
        'final_soc': test_env.battery.soc,
        'avg_plan_deviation': avg_plan_deviation,
        'economic_profit': test_env.total_revenue
    }
    
    print(f"Final Total Reward: {total_reward:.2f}")
    print(f"Average Plan Deviation: {avg_plan_deviation:.3f}")
    print(f"Economic Profit: £{test_env.total_revenue:,.0f}")

print("\n=== HIERARCHICAL RESULTS COMPARISON ===")
for strategy, result in results.items():
    print(f"{strategy.upper():<12} | Reward: {result['total_reward']:6.2f} | "
          f"Profit: £{result['economic_profit']:>6,.0f} | "
          f"Plan Dev: {result['avg_plan_deviation']:.3f} | "
          f"Final SOC: {result['final_soc']:.2f}")

print("\n" + "="*50)
print("RESEARCH INSIGHT: Different day-ahead strategies enable")
print("different real-time behaviors and trade-offs!")