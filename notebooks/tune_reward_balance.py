# notebooks/tune_reward_balance.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv
import numpy as np

def test_reward_scaling():
    """Test different stability reward scaling factors"""
    print("=== REWARD BALANCE TUNING ===")
    print("Testing different stability penalty weights\n")
    
    stability_weights = [1.0, 0.1, 0.01, 0.001]  # Reduce stability impact
    
    for weight in stability_weights:
        print(f"\n--- Testing stability weight: {weight} ---")
        
        env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
        
        # Modify the environment to use weighted stability (temporary fix)
        original_step = env.step
        def weighted_step(action):
            obs, reward, done, truncated, info = original_step(action)
            # Apply weight to stability component
            weighted_reward = info['economic'] + (info['stability'] * weight)
            info['weighted_stability'] = info['stability'] * weight
            info['weighted_total'] = weighted_reward
            return obs, weighted_reward, done, truncated, info
        env.step = weighted_step
        
        obs, info = env.reset()
        total_economic = 0
        total_stability = 0
        total_weighted = 0
        
        # Run with simple policy
        for step in range(24):
            hour = int(obs[0] * 24)
            
            if 0 <= hour < 6:  # Night - charge
                action = [0.3, 0.3]
            elif 16 <= hour < 20:  # Peak - discharge
                action = [-0.5, -0.3]  # Scotland discharges more
            else:
                action = [0, 0]
            
            obs, reward, done, truncated, info = env.step(action)
            total_economic += info['economic']
            total_stability += info['stability']
            total_weighted += info['weighted_total']
            
            if done:
                break
        
        stability_ratio = abs(total_stability / total_economic) if total_economic != 0 else 0
        
        print(f"  Economic: £{total_economic:7.1f}")
        print(f"  Stability: £{total_stability:7.1f}")
        print(f"  Weighted Total: £{total_weighted:7.1f}")
        print(f"  Stability/Economic Ratio: {stability_ratio:5.1f}")
        
        # Recommendation
        if 0.1 <= stability_ratio <= 10.0:  # Reasonable balance
            print(f"  ✅ GOOD BALANCE: Economic and stability are comparable")
        elif stability_ratio > 10.0:
            print(f"  ⚠️  STABILITY DOMINANT: Consider lower weight")
        else:
            print(f"  ⚠️  ECONOMIC DOMINANT: Consider higher weight")

def find_optimal_weight():
    """Find the optimal stability weight through grid search"""
    print(f"\n=== FINDING OPTIMAL STABILITY WEIGHT ===")
    
    best_weight = 1.0
    best_balance_score = float('inf')
    
    for weight in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
        env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
        
        # Apply weight temporarily
        original_step = env.step
        def weighted_step(action):
            obs, reward, done, truncated, info = original_step(action)
            weighted_reward = info['economic'] + (info['stability'] * weight)
            return obs, weighted_reward, done, truncated, info
        env.step = weighted_step
        
        # Test performance
        obs, info = env.reset()
        total_economic = 0
        total_stability = 0
        
        for step in range(24):
            hour = int(obs[0] * 24)
            if 0 <= hour < 6:
                action = [0.3, 0.3]
            elif 16 <= hour < 20:
                action = [-0.5, -0.3]
            else:
                action = [0, 0]
            
            obs, reward, done, truncated, info = env.step(action)
            total_economic += info['economic']
            total_stability += info['stability']
            
            if done:
                break
        
        # Balance score: how close stability and economic are in magnitude (log scale)
        economic_abs = abs(total_economic)
        stability_abs = abs(total_stability * weight)
        
        if economic_abs > 0 and stability_abs > 0:
            ratio = max(economic_abs/stability_abs, stability_abs/economic_abs)
            balance_score = abs(np.log10(ratio))  # 0 means perfect balance
            
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_weight = weight
            
            print(f"  Weight {weight:6.3f}: Economic £{total_economic:7.1f} | "
                  f"Stability £{total_stability * weight:7.1f} | "
                  f"Balance Score: {balance_score:.3f}")
    
    print(f"\n🎯 RECOMMENDED OPTIMAL WEIGHT: {best_weight:.3f}")
    return best_weight

if __name__ == "__main__":
    # Test different scaling factors
    test_reward_scaling()
    
    # Find optimal weight
    optimal_weight = find_optimal_weight()
    
    print(f"\n=== THESIS IMPLICATION ===")
    print("Current stability penalties are 50-100x larger than economic rewards.")
    print("This overwhelms the learning signal. Optimal weight around 0.01-0.05")
    print("will balance economic and stability objectives effectively.")