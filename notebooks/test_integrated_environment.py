# notebooks/test_integrated_environment.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv

def main():
    print("=== TESTING INTEGRATED ENVIRONMENT WITH STABILITY REWARDS ===")
    
    # Test environment creation
    try:
        env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return
    
    # Test reset
    try:
        obs, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {obs.shape}")
        print(f"   Initial State: Hour {int(obs[0]*24)}, SOC: {obs[2]:.2f}, {obs[3]:.2f}")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return
    
    # Test a few steps
    print("\n--- Testing step function ---")
    try:
        for step in range(5):  # Just test first 5 steps
            if step == 0:
                action = [0.5, 0.5]   # Charge both
            else:
                action = [0.1, -0.1]  # Mixed actions
            
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"Step {step}: Reward £{reward:7.2f} | "
                  f"Economic £{info['economic']:7.2f} | "
                  f"Stability £{info['stability']:7.2f} | "
                  f"SOC: {info['soc'][0]:.2f}(SCT), {info['soc'][1]:.2f}(LON)")
            
            if done:
                break
                
        print("✅ Step function working correctly!")
        
    except Exception as e:
        print(f"❌ Step function failed: {e}")
        return
    
    # Test full episode
    print("\n--- Testing full episode ---")
    try:
        env = HierarchicalBatteryEnv(['SCOTLAND', 'LONDON'])
        obs, info = env.reset()
        
        for step in range(24):
            # Simple policy: charge at night, discharge at peak
            if 0 <= step < 6:   # Night - charge both
                action = [0.5, 0.5]   # Charge
            elif 18 <= step < 20: # Peak - discharge Scotland more
                action = [1.0, 0.3]   # Scotland discharges fully
            else:
                action = [0, 0]       # Do nothing
                
            obs, reward, done, truncated, info = env.step(action)
            
            if step in [0, 6, 12, 18] or done:
                print(f"Hour {step:2d}: Total £{reward:7.2f} | "
                      f"Economic £{info['economic']:7.2f} | "
                      f"Stability £{info['stability']:7.2f}")
        
        print(f"\nFINAL TOTALS: Economic £{env.economic_reward:.2f} | "
              f"Stability £{env.stability_reward:.2f} | "
              f"Total £{env.total_reward:.2f}")
        
        # Success validation
        if env.stability_reward < -500:
            print("✅ SUCCESS: Stability rewards significantly impact total reward!")
            print("   Multi-objective optimization is working!")
        else:
            print("⚠️  Stability impact may need tuning")
            
    except Exception as e:
        print(f"❌ Full episode test failed: {e}")

if __name__ == "__main__":
    main()