"""
scripts/final_rule_based_validation.py
Final validation with optimized rule-based agent
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt

class OptimizedRuleBasedAgent:
    """
    Optimized rule-based dispatch based on UK market analysis
    Uses price thresholds learned from training data
    """
    def __init__(self, train_prices):
        # Learn optimal thresholds from training data
        self.price_low = np.percentile(train_prices, 25)   # Bottom 25%
        self.price_high = np.percentile(train_prices, 75)  # Top 25%
        self.price_very_low = np.percentile(train_prices, 10)
        self.price_very_high = np.percentile(train_prices, 90)
        
        print(f"Learned thresholds from training data:")
        print(f"  Very low:  <£{self.price_very_low:.2f}")
        print(f"  Low:       <£{self.price_low:.2f}")
        print(f"  High:      >£{self.price_high:.2f}")
        print(f"  Very high: >£{self.price_very_high:.2f}")
        
    def predict(self, state_dict):
        """
        Smart dispatch logic:
        - Aggressive charge at very low prices
        - Moderate charge at low prices
        - Aggressive discharge at very high prices  
        - Moderate discharge at high prices
        - Respect SOC limits strictly
        """
        price = state_dict['price']
        soc1 = state_dict['soc1']
        hour = state_dict['hour']
        
        # Very low price: aggressive charging
        if price < self.price_very_low and soc1 < 0.85:
            action = min(1.0, (0.85 - soc1) * 2)  # Scale with available capacity
        
        # Low price: moderate charging
        elif price < self.price_low and soc1 < 0.80:
            action = min(0.6, (0.80 - soc1) * 2)
        
        # Very high price: aggressive discharging
        elif price > self.price_very_high and soc1 > 0.20:
            action = -min(0.9, (soc1 - 0.20) * 2)
        
        # High price: moderate discharging
        elif price > self.price_high and soc1 > 0.25:
            action = -min(0.6, (soc1 - 0.25) * 2)
        
        # Safety limits
        elif soc1 < 0.12:
            action = 0.7  # Emergency charge
        elif soc1 > 0.88:
            action = -0.5  # Prevent overcharge
        
        # Peak hours (4-8pm): prefer to discharge if price decent
        elif 16 <= hour <= 20 and price > self.price_low and soc1 > 0.30:
            action = -0.4
        
        # Off-peak (11pm-6am): prefer to charge if price decent
        elif (hour >= 23 or hour <= 6) and price < self.price_high and soc1 < 0.70:
            action = 0.4
        
        # Otherwise hold
        else:
            action = 0.0
        
        return float(action)

class BatteryEnvironment:
    """Simple battery environment for validation"""
    def __init__(self, capacity=5.0, power_limit=1.0, efficiency=0.95, degradation_cost=0.01):
        self.capacity = capacity
        self.power_limit = power_limit
        self.efficiency = efficiency
        self.degradation_cost = degradation_cost
        self.soc = 0.5
        
    def reset(self):
        self.soc = 0.5
        return self.soc
        
    def step(self, action, price, grid_stress=0.5):
        """Execute action and return reward"""
        power = float(action) * self.power_limit
        energy = power * 0.5
        
        if energy > 0:
            energy_actual = energy * self.efficiency
        else:
            energy_actual = energy / self.efficiency
        
        new_soc = self.soc + (energy_actual / self.capacity)
        violated = (new_soc < 0.0 or new_soc > 1.0)
        new_soc = np.clip(new_soc, 0.0, 1.0)
        
        revenue = -energy * price
        degradation = abs(energy) * self.degradation_cost
        grid_support = 0
        if grid_stress > 0.7 and energy < 0:
            grid_support = abs(energy) * 3
        
        reward = revenue - degradation + grid_support
        
        if violated:
            reward -= 50
        
        self.soc = new_soc
        
        return new_soc, reward, violated, revenue

# Main execution
print("="*70)
print("FINAL RULE-BASED VALIDATION")
print("="*70)

# Load data
train_data = pd.read_csv('data/training/train.csv')
test_data = pd.read_csv('data/training/test.csv')

train_data['datetime'] = pd.to_datetime(train_data['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

# Calculate grid stress
train_data['grid_stress'] = (train_data['TSD'] - train_data['TSD'].min()) / (train_data['TSD'].max() - train_data['TSD'].min())
test_data['grid_stress'] = (test_data['TSD'] - test_data['TSD'].min()) / (test_data['TSD'].max() - test_data['TSD'].min())

print(f"\n✓ Train data: {len(train_data):,} rows")
print(f"✓ Test data:  {len(test_data):,} rows")

# Create agent (learn from training data)
print("\nLearning thresholds from training data...")
agent = OptimizedRuleBasedAgent(train_data['system_price'].values)

# Create test episodes
episodes = []
for date in test_data['datetime'].dt.date.unique():
    episode = test_data[test_data['datetime'].dt.date == date].copy()
    if len(episode) == 48:
        episodes.append(episode.reset_index(drop=True))

print(f"\n✓ Created {len(episodes)} test episodes")

# Run validation
env = BatteryEnvironment()

results = {
    'datetime': [], 'price': [], 'demand': [], 'grid_stress': [],
    'action': [], 'power': [], 'soc': [], 
    'total_reward': [], 'total_revenue': [], 'violated': []
}

total_reward = 0
total_revenue = 0
total_violations = 0

print(f"\n{'='*70}")
print("RUNNING VALIDATION")
print(f"{'='*70}\n")

for ep_idx, episode in enumerate(episodes):
    soc = env.reset()
    
    for t in range(len(episode)):
        row = episode.iloc[t]
        
        state = {
            'hour': row['hour'],
            'price': row['system_price'],
            'soc1': soc,
            'grid_stress': row['grid_stress']
        }
        
        action = agent.predict(state)
        soc, reward, violated, revenue = env.step(action, row['system_price'], row['grid_stress'])
        
        results['datetime'].append(row['datetime'])
        results['price'].append(row['system_price'])
        results['demand'].append(row['TSD'])
        results['grid_stress'].append(row['grid_stress'])
        results['action'].append(action)
        results['power'].append(action * env.power_limit)
        results['soc'].append(soc)
        results['total_reward'].append(reward)
        results['total_revenue'].append(revenue)
        results['violated'].append(violated)
        
        total_reward += reward
        total_revenue += revenue
        if violated:
            total_violations += 1
    
    if (ep_idx + 1) % 50 == 0:
        print(f"  Completed {ep_idx + 1}/{len(episodes)} episodes...")

results_df = pd.DataFrame(results)
num_days = len(episodes)

print(f"\n{'='*70}")
print("FINAL RESULTS - RULE-BASED AGENT")
print(f"{'='*70}")
print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Test period:          {num_days} days")
print(f"   Total reward:         £{total_reward:,.2f}")
print(f"   Total revenue:        £{total_revenue:,.2f}")
print(f"   Avg reward/day:       £{total_reward/num_days:.2f}")
print(f"   Avg revenue/day:      £{total_revenue/num_days:.2f}")
print(f"   Constraint violations: {total_violations} ({total_violations/(num_days*48)*100:.2f}%)")

print(f"\n⚡ BATTERY STATS:")
print(f"   Avg SOC:              {results_df['soc'].mean()*100:.1f}%")
print(f"   Min SOC:              {results_df['soc'].min()*100:.1f}%")
print(f"   Max SOC:              {results_df['soc'].max()*100:.1f}%")

print(f"\n📈 OPERATION STATS:")
total_energy = abs(results_df['power']).sum() * 0.5
print(f"   Total energy cycled:  {total_energy:,.1f} MWh")
print(f"   Avg daily cycling:    {total_energy/num_days:.1f} MWh/day")

# Save results
Path('results').mkdir(exist_ok=True)
output_file = f'results/validation_rule_based_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_df.to_csv(output_file, index=False)

metrics = {
    'method': 'rule_based_optimized',
    'total_reward': float(total_reward),
    'total_revenue': float(total_revenue),
    'avg_reward_per_day': float(total_reward / num_days),
    'avg_revenue_per_day': float(total_revenue / num_days),
    'violation_rate': float(total_violations / (num_days * 48)),
    'num_days': num_days,
    'num_violations': int(total_violations),
    'avg_soc': float(results_df['soc'].mean())
}

with open('results/metrics_rule_based_final.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print(f"✓ Metrics saved to: results/metrics_rule_based_final.json")

# Quick visualization
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

week_data = results_df.iloc[:48*7]

# Price and actions
ax1 = axes[0]
ax1_twin = ax1.twinx()
ax1.plot(week_data['datetime'], week_data['price'], 'b-', label='Price', linewidth=2)
ax1_twin.bar(week_data['datetime'], week_data['power'], alpha=0.3, color='green')
ax1.set_ylabel('Price (£/MWh)', color='b')
ax1_twin.set_ylabel('Power (MW)', color='g')
ax1.set_title('Rule-Based Dispatch: Sample Week', fontweight='bold')
ax1.grid(True, alpha=0.3)

# SOC
axes[1].plot(week_data['datetime'], week_data['soc']*100, 'purple', linewidth=2)
axes[1].axhline(y=10, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(y=90, color='r', linestyle='--', alpha=0.5)
axes[1].set_ylabel('SOC (%)')
axes[1].set_title('State of Charge', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Cumulative reward
axes[2].plot(results_df['datetime'], results_df['total_reward'].cumsum(), 'darkgreen', linewidth=2)
axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.3)
axes[2].set_ylabel('Cumulative Reward (£)')
axes[2].set_title('Cumulative Reward Over Time', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/rule_based_final_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: results/rule_based_final_visualization.png")

print(f"\n{'='*70}")
print("✓✓✓ VALIDATION COMPLETE!")
print(f"{'='*70}")