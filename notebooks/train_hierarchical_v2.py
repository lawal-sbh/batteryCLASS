# notebooks/train_hierarchical_v2.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv
from src.settlement_engine import SettlementEngine
import numpy as np

def generate_commander_plan(strategy_type, episode):
    """Generate different day-ahead strategies for testing"""
    if strategy_type == "arbitrage":
        # Charge at night, discharge at peak
        targets = [0.3] * 24
        for hour in [18, 19, 20]:
            targets[hour] = 0.8  # High SOC for evening peak
        for hour in [2, 3, 4]:
            targets[hour] = 0.9  # Charge overnight
        expected_value = 2000 + np.random.normal(0, 200)
        
    elif strategy_type == "conservative":
        # Maintain medium SOC throughout
        targets = [0.5] * 24
        expected_value = 500 + np.random.normal(0, 100)
        
    else:  # adaptive - learns over time
        # Start conservative, become more aggressive with learning
        base_soc = 0.4 + min(0.3, episode * 0.03)
        targets = [base_soc] * 24
        for hour in [17, 18, 19]:
            targets[hour] = base_soc + 0.4
        expected_value = 1000 + episode * 50
        
    return {
        'soc_targets': targets,
        'expected_value': expected_value,
        'strategy_type': strategy_type
    }

def simple_tactician_policy(observation):
    """Simple rule-based tactician for testing"""
    hour = int(observation[0] * 24)
    current_soc_1 = observation[2]  # Scotland SOC
    current_soc_2 = observation[3]  # London SOC
    commander_target_1 = observation[4]
    commander_target_2 = observation[5]
    grid_stress = observation[6]
    
    # Base actions
    action_scotland = 0
    action_london = 0
    
    # Time-based strategy
    if 0 <= hour < 6:  # Night - charge
        action_scotland = 0.3 if current_soc_1 < 0.8 else 0
        action_london = 0.3 if current_soc_2 < 0.8 else 0
        
    elif 16 <= hour < 20:  # Evening peak - discharge
        # Scotland discharges more aggressively but watches stability costs
        action_scotland = -0.8 if current_soc_1 > 0.3 and grid_stress < 0.7 else -0.3
        action_london = -0.6 if current_soc_2 > 0.3 else -0.2
        
    elif 6 <= hour < 16:  # Day - follow commander target
        deviation_1 = commander_target_1 - current_soc_1
        deviation_2 = commander_target_2 - current_soc_2
        action_scotland = np.clip(deviation_1 * 2, -0.5, 0.5)
        action_london = np.clip(deviation_2 * 2, -0.5, 0.5)
    
    return [action_scotland, action_london]

def train_hierarchical_with_settlement():
    print("=== HIERARCHICAL RL TRAINING WITH SETTLEMENT ENGINE ===")
    print("Training two-level: Day-Ahead Commander + Real-Time Tactician")
    print("With multi-objective rewards (Economic + Stability)\n")
    
    env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
    settlement = SettlementEngine()
    
    strategies = ['conservative', 'arbitrage', 'adaptive']
    
    for episode in range(10):  # Train for 10 episodes
        strategy_type = strategies[episode % len(strategies)]
        
        # Commander makes day-ahead plan
        commander_plan = generate_commander_plan(strategy_type, episode)
        env.set_commander_target(commander_plan['soc_targets'])
        
        # Tactician executes in real-time
        obs, info = env.reset()
        episode_performance = {
            'economic': 0, 
            'stability': 0,
            'total_reward': 0,
            'utilization': 0
        }
        
        step_count = 0
        for step in range(24):
            action = simple_tactician_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_performance['economic'] += info['economic']
            episode_performance['stability'] += info['stability']
            episode_performance['total_reward'] += reward
            episode_performance['utilization'] += abs(sum(action)) / 2  # Average utilization
            
            step_count += 1
            if done:
                break
        
        # Calculate averages
        episode_performance['utilization'] /= step_count
        
        # Settlement and learning
        settlement_data = settlement.record_episode(
            commander_plan, 
            episode_performance,
            stability_impact=episode_performance['stability']
        )
        
        commander_feedback = settlement.calculate_commander_feedback()
        
        # Print episode results
        print(f"Episode {episode:2d} ({strategy_type:12}) | "
              f"Economic: £{episode_performance['economic']:7.1f} | "
              f"Stability: £{episode_performance['stability']:7.1f} | "
              f"Total: £{episode_performance['total_reward']:7.1f} | "
              f"Flexibility: £{settlement_data['flexibility_value']:7.1f} | "
              f"Update: {commander_feedback['strategy_update']}")
        
        # Show dramatic events
        if abs(episode_performance['stability']) > 3000:
            print("  ⚡ MAJOR STABILITY EVENT: Grid stability heavily impacted!")
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Total episodes: {len(settlement.episode_data)}")
    print(f"Final settlement data available for analysis")
    
    return settlement

if __name__ == "__main__":
    settlement_engine = train_hierarchical_with_settlement()
    
    # Show learning progression
    if len(settlement_engine.episode_data) > 0:
        print(f"\n--- LEARNING PROGRESSION ---")
        first_episode = settlement_engine.episode_data[0]
        last_episode = settlement_engine.episode_data[-1]
        
        improvement = last_episode['actual_value'] - first_episode['actual_value']
        print(f"Value improvement: £{improvement:.1f}")
        print(f"Stability awareness: {abs(last_episode['stability_impact']) > 2000}")