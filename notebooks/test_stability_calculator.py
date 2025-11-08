# notebooks/test_stability_calculator.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constraint_calculator import ConstraintStabilityCalculator

def main():
    print("=== TESTING CONSTRAINT STABILITY CALCULATOR ===")
    
    calc = ConstraintStabilityCalculator(['SCOTLAND', 'LONDON'])
    
    # Test 1: Scotland with congestion and RoCoF breach
    print("\n1. SCOTLAND - High congestion + RoCoF breach:")
    scotland_reward = calc.calculate_stability_rewards(
        power_dispatch=10,  # MW
        location='SCOTLAND', 
        grid_conditions={'hour': 18, 'frequency': 49.8, 'rocof': 0.6}
    )
    print(f"   {scotland_reward}")
    
    # Test 2: London with low congestion, stable grid
    print("\n2. LONDON - Low congestion + stable grid:")
    london_reward = calc.calculate_stability_rewards(
        power_dispatch=10,  # MW  
        location='LONDON',
        grid_conditions={'hour': 3, 'frequency': 49.9, 'rocof': 0.3}
    )
    print(f"   {london_reward}")
    
    # Analysis
    print(f"\n3. LOCATIONAL DIVERSITY ANALYSIS:")
    scotland_total = scotland_reward['total_stability']
    london_total = london_reward['total_stability']
    
    print(f"   Scotland: £{scotland_total:.2f}")
    print(f"   London: £{london_total:.2f}")
    print(f"   Difference: £{abs(scotland_total - london_total):.2f}")
    
    # Success criteria
    if scotland_total < -900 and london_total > -50:
        print("   ✅ SUCCESS: Strong locational diversity confirmed!")
        print("   - Scotland heavily penalized for congestion + instability")
        print("   - London has minimal stability costs")
    else:
        print("   ❌ NEEDS ADJUSTMENT: Reward scaling may need tuning")

if __name__ == "__main__":
    main()