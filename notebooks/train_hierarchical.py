# notebooks/train_hierarchical.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv
from src.settlement_engine import SettlementEngine
import numpy as np

def generate_commander_plan(episode):
    """Generate day-ahead SOC targets based on strategy"""
    strategies = ['conservative', 'arbitrage', 'adaptive']
    strategy_type = strategies[episode % len(strategies)]
    
    if strategy_type == "arbitrage":
        # Charge at cheap times, discharge at expensive times
        targets = [0.5] * 24
        for hour in [18, 19, 20]:  # Evening peak
            targets[hour] = 0.2  # Low SOC to discharge
        for hour in [2, 3, 4]:    # Night cheap
            targets[hour] = 0.8  # High SOC to charge
        expected_value = 1500
        
    elif strategy_type == "conservative":
        # Maintain medium SOC, avoid risks
        targets = [0.5] * 24
        expected_value = 500
        
    else:  # adaptive
        # Learn to be more aggressive over time
        aggression = min(0.3, episode * 0.05)
        targets = [0.5 - aggression] * 24
        for hour in [17, 18, 19]:
            targets[hour] = 0.2 - aggression
        expected_value = 800 + episode * 20
    
    return {
        'soc_targets': targets,
        'expected_value': expected_value,
        'strategy_type': strategy_type
    }

def tactician_policy(observation):
    """Simple rule-based real-time policy"""
    hour = int(observation[0] * 24)
    soc_scotland = observation[2]
    soc_london = observation[3] 
    commander_target_scotland = observation[4]
    commander_target_london = observation[5]
    grid_stress = observation[6]
    
    # Initialize actions
    action_scotland = 0.0
    action_london = 0.0
    
    # Time-based strategy with stability awareness
    if 0 <= hour < 6:  # Night - charge both
        action_scotland = 0.4 if soc_scotland < 0.9 else 0.0
        action_london = 0.4 if soc_london < 0.9 else 0.0
        
    elif 6 <= hour < 16:  # Day - follow commander targets
        # Move toward commander's SOC targets
        action_scotland = np.clip((commander_target_scotland - soc_scotland) * 3, -0.6, 0.6)
        action_london = np.clip((commander_target_london - soc_london) * 3, -0.6, 0.6)
        
    elif 16 <= hour < 22:  # Evening peak - discharge but watch stability
        # Scotland: discharge but be careful during high grid stress
        if grid_stress > 0.7:  # High congestion
            action_scotland = -0.3 if soc_scotland > 0.4 else 0.0
        else:
            action_scotland = -0.8 if soc_scotland > 0.3 else 0.0
            
        # London: discharge more aggressively (lower congestion costs)
        action_london = -0.7 if soc_london > 0.3 else 0.0
        
    else:  # Late night - minimal activity
        action_scotland = 0.0
        action_london = 0.0
    
    return [float(action_scotland), float(action_london)]

def train_hierarchical_with_settlement():
    print("=== HIERARCHICAL RL TRAINING WITH SETTLEMENT ===")
    print("Commander (Day-Ahead) → Tactician (Real-Time) → Settlement (Learning)")
    print("Multi-Objective: Economic Profit + Grid Stability\n")
    
    # Initialize components
    env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
    settlement = SettlementEngine()
    
    print("Starting training for 10 episodes...")
    print("-" * 90)
    
    for episode in range(10):
        # PHASE 1: Commander makes day-ahead plan
        commander_plan = generate_commander_plan(episode)
        env.set_commander_target(commander_plan['soc_targets'])
        
        # PHASE 2: Tactician executes in real-time
        obs, info = env.reset()
        episode_performance = {'economic': 0, 'stability': 0, 'total_reward': 0}
        
        for step in range(24):
            action = tactician_policy(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_performance['economic'] += info['economic']
            episode_performance['stability'] += info['stability']
            episode_performance['total_reward'] += reward
            
            if done:
                break
        
        # PHASE 3: Settlement and learning
        settlement_data = settlement.record_episode(
            commander_plan, 
            episode_performance,
            stability_impact=episode_performance['stability']
        )
        
        commander_feedback = settlement.calculate_commander_feedback()
        
        # Display results
        strategy_symbol = {
            'conservative': '🛡️ ',
            'arbitrage': '💰', 
            'adaptive': '🧠'
        }
        
        print(f"Ep {episode:2d} {strategy_symbol[commander_plan['strategy_type']]} "
              f"{commander_plan['strategy_type']:12} | "
              f"Economic: £{episode_performance['economic']:7.1f} | "
              f"Stability: £{episode_performance['stability']:7.1f} | "
              f"Flexibility: £{settlement_data['flexibility_value']:7.1f} | "
              f"Update: {commander_feedback['strategy_update']:15}")
    
    print("-" * 90)
    print("=== TRAINING COMPLETE ===")
    
    # Analysis
    if len(settlement.episode_data) > 1:
        first_total = settlement.episode_data[0]['actual_value'] 
        last_total = settlement.episode_data[-1]['actual_value']
        improvement = last_total - first_total
        
        print(f"\n--- PERFORMANCE ANALYSIS ---")
        print(f"Total value improvement: £{improvement:.1f}")
        print(f"Final strategy: {settlement.episode_data[-1].get('strategy_type', 'N/A')}")
        
        if improvement > 0:
            print("✅ POSITIVE TREND: Learning is occurring!")
        else:
            print("⚠️  Needs tuning: Consider adjusting reward weights")
    
    return settlement

if __name__ == "__main__":
    # First, test basic functionality
    print("Testing environment and settlement engine...")
    
    try:
        # Quick test of environment
        test_env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
        obs, info = test_env.reset()
        print(f"✅ Environment test: Observation shape {obs.shape}")
        
        # Quick test of settlement
        test_settlement = SettlementEngine()
        test_data = test_settlement.record_episode(
            {'expected_value': 1000, 'strategy_type': 'test'},
            {'economic': 1200, 'stability': -300},
            -300
        )
        print(f"✅ Settlement test: Flexibility £{test_data['flexibility_value']:.1f}")
        
        print("\n" + "="*60)
        
        # Run full training
        final_settlement = train_hierarchical_with_settlement()
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()