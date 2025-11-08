# notebooks/portfolio_validation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_environment_v2 import HierarchicalBatteryEnv
import numpy as np

def compare_portfolio_vs_single():
    """Compare portfolio performance vs single battery"""
    print("=== PORTFOLIO VS SINGLE BATTERY COMPARISON ===")
    print("Testing: Portfolio (Scotland + London) vs Individual Batteries\n")
    
    # Test configurations
    configurations = {
        'Portfolio (SCT+LON)': ['SCOTLAND', 'LONDON'],
        'Scotland Only': ['SCOTLAND'],
        'London Only': ['LONDON']
    }
    
    results = {}
    
    for name, locations in configurations.items():
        print(f"--- Testing {name} ---")
        env = HierarchicalBatteryEnv(locations)
        obs, info = env.reset()
        
        total_economic = 0
        total_stability = 0
        total_reward = 0
        
        # Run a full day with simple policy
        for step in range(24):
            hour = int(obs[0] * 24)
            
            # Simple time-based policy
            if 0 <= hour < 6:  # Night - charge
                action = [0.3] * len(locations)
            elif 16 <= hour < 20:  # Peak - discharge
                action = [-0.5] * len(locations)
            else:  # Day - do nothing
                action = [0] * len(locations)
            
            obs, reward, done, truncated, info = env.step(action)
            total_economic += info['economic']
            total_stability += info['stability']
            total_reward += reward
            
            if done:
                break
        
        results[name] = {
            'economic': total_economic,
            'stability': total_stability,
            'total': total_economic + total_stability,
            'locations': locations
        }
        
        print(f"  Economic: £{total_economic:7.1f}")
        print(f"  Stability: £{total_stability:7.1f}")
        print(f"  Total: £{total_economic + total_stability:7.1f}")
        print()
    
    # Portfolio benefit analysis
    print("=== PORTFOLIO BENEFIT ANALYSIS ===")
    
    portfolio_total = results['Portfolio (SCT+LON)']['total']
    scotland_total = results['Scotland Only']['total']
    london_total = results['London Only']['total']
    sum_singles = scotland_total + london_total
    
    portfolio_benefit = portfolio_total - sum_singles
    benefit_percentage = (portfolio_benefit / abs(sum_singles)) * 100 if sum_singles != 0 else 0
    
    print(f"Portfolio total value: £{portfolio_total:.1f}")
    print(f"Scotland Only: £{scotland_total:.1f}")
    print(f"London Only: £{london_total:.1f}")
    print(f"Sum of individual batteries: £{sum_singles:.1f}")
    print(f"Portfolio benefit: £{portfolio_benefit:.1f} ({benefit_percentage:+.1f}%)")
    
    # Key insights
    print(f"\n=== KEY INSIGHTS ===")
    if portfolio_benefit > 0:
        print("✅ PORTFOLIO BENEFIT CONFIRMED!")
        print("   Two coordinated batteries outperform the sum of individual batteries")
        print("   This demonstrates the value of coordinated portfolio dispatch")
    else:
        print("⚠️  No portfolio benefit detected")
        print("   This may indicate need for better coordination or reward tuning")
    
    # Locational analysis
    scotland_stability_ratio = abs(results['Scotland Only']['stability'] / results['Scotland Only']['economic']) if results['Scotland Only']['economic'] != 0 else 0
    london_stability_ratio = abs(results['London Only']['stability'] / results['London Only']['economic']) if results['London Only']['economic'] != 0 else 0
    
    print(f"\n--- Locational Characteristics ---")
    print(f"Scotland stability/economic ratio: {scotland_stability_ratio:.1f}")
    print(f"London stability/economic ratio: {london_stability_ratio:.1f}")
    
    if scotland_stability_ratio > london_stability_ratio:
        print("Scotland has higher grid stability impact per £ earned")
    
    return results

def analyze_coordination_benefits(results):
    """Deeper analysis of coordination benefits"""
    print(f"\n=== COORDINATION BENEFIT ANALYSIS ===")
    
    portfolio = results['Portfolio (SCT+LON)']
    scotland = results['Scotland Only']
    london = results['London Only']
    
    # Economic coordination benefit
    economic_benefit = portfolio['economic'] - (scotland['economic'] + london['economic'])
    
    # Stability coordination benefit  
    stability_benefit = portfolio['stability'] - (scotland['stability'] + london['stability'])
    
    print(f"Economic coordination benefit: £{economic_benefit:.1f}")
    print(f"Stability coordination benefit: £{stability_benefit:.1f}")
    
    if economic_benefit > 0:
        print("✅ Positive economic coordination: Portfolio earns more than sum of parts")
    else:
        print("⚠️  Negative economic coordination: Need better tactical coordination")
    
    if stability_benefit > 0:  # Less negative is better
        print("✅ Positive stability coordination: Portfolio has lower grid impact")
    else:
        print("⚠️  Negative stability coordination: Portfolio increases grid stress")
    
    return {
        'economic_coordination': economic_benefit,
        'stability_coordination': stability_benefit
    }

if __name__ == "__main__":
    # Run the comparison
    results = compare_portfolio_vs_single()
    
    # Deep dive analysis
    coordination_benefits = analyze_coordination_benefits(results)
    
    print(f"\n🎯 THESIS IMPLICATION: ")
    if coordination_benefits['economic_coordination'] > 0 or coordination_benefits['stability_coordination'] > 0:
        print("Hierarchical coordination of battery portfolios provides measurable benefits")
        print("for both economic value and grid stability services.")
    else:
        print("Portfolio coordination shows potential but requires optimization.")