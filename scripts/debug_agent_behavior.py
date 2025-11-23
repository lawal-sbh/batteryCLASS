"""
scripts/debug_agent_behavior.py
Analyze agent behavior on sample data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn

# Copy the agent classes here instead of importing
class CommanderNetwork(nn.Sequential):
    def __init__(self, state_dim=5, action_dim=48, hidden_dim=256):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

class TacticianNetwork(nn.Sequential):
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=128):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

class HierarchicalBatteryAgent:
    def __init__(self, commander_path, tactician_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load Commander
        commander_checkpoint = torch.load(commander_path, map_location=self.device, weights_only=False)
        commander_state = commander_checkpoint['policy_state_dict'] if 'policy_state_dict' in commander_checkpoint else commander_checkpoint
        self.commander = CommanderNetwork().to(self.device)
        self.commander.load_state_dict(commander_state)
        self.commander.eval()
        
        # Load Tactician
        tactician_checkpoint = torch.load(tactician_path, map_location=self.device, weights_only=False)
        tactician_state = tactician_checkpoint['policy_state_dict'] if 'policy_state_dict' in tactician_checkpoint else tactician_checkpoint
        self.tactician = TacticianNetwork().to(self.device)
        self.tactician.load_state_dict(tactician_state)
        self.tactician.eval()
        
        self.commander_target = None
        self.steps_since_command = 0
        self.command_interval = 12
        
    def predict(self, state_dict):
        with torch.no_grad():
            state = torch.FloatTensor([
                state_dict['hour'] / 24,
                state_dict['price'] / 100,
                state_dict['soc1'],
                state_dict['soc2'],
                state_dict['grid_stress']
            ]).unsqueeze(0).to(self.device)
            
            if self.steps_since_command % self.command_interval == 0:
                self.commander_target = self.commander(state).cpu().numpy()[0]
            
            action = self.tactician(state).cpu().numpy()[0]
            self.steps_since_command += 1
            return action

# Load agent
print("Loading agent...")
agent = HierarchicalBatteryAgent(
    'models/commander/best_model.pth',
    'models/tactician/best_model.pth'
)
print("✓ Agent loaded\n")

# Load real data
data = pd.read_csv('data/uk_battery_dispatch_complete_data.csv')
data['grid_stress'] = (data['TSD'] - data['TSD'].min()) / (data['TSD'].max() - data['TSD'].min())

print("="*60)
print("AGENT BEHAVIOR ANALYSIS")
print("="*60)

# Test on various price scenarios
print("\nPRICE RESPONSE TEST")
print("-"*60)
test_cases = [
    {'name': 'Very low price', 'price': 3.0},
    {'name': 'Low price', 'price': 6.0},
    {'name': 'Medium price', 'price': 8.0},
    {'name': 'High price', 'price': 12.0},
    {'name': 'Very high price', 'price': 20.0},
]

for case in test_cases:
    state = {
        'hour': 12,
        'price': case['price'],
        'soc1': 0.5,
        'soc2': 0.5,
        'grid_stress': 0.5
    }
    
    actions = agent.predict(state)
    
    action1_str = 'CHARGE' if actions[0] > 0.1 else ('DISCHARGE' if actions[0] < -0.1 else 'HOLD')
    action2_str = 'CHARGE' if actions[1] > 0.1 else ('DISCHARGE' if actions[1] < -0.1 else 'HOLD')
    
    print(f"{case['name']:20s} £{case['price']:5.2f}/MWh:")
    print(f"  Battery 1: {actions[0]:+.3f} ({action1_str:10s}) | Battery 2: {actions[1]:+.3f} ({action2_str})")

# Test SOC sensitivity
print("\n" + "="*60)
print("SOC SENSITIVITY TEST (at medium price £8/MWh)")
print("-"*60)

soc_tests = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
for soc in soc_tests:
    state = {
        'hour': 12,
        'price': 8.0,
        'soc1': soc,
        'soc2': soc,
        'grid_stress': 0.5
    }
    actions = agent.predict(state)
    print(f"SOC {soc*100:5.1f}%: Bat1={actions[0]:+.3f}, Bat2={actions[1]:+.3f}")

# Test hour sensitivity
print("\n" + "="*60)
print("TIME OF DAY SENSITIVITY (at £8/MWh, SOC=50%)")
print("-"*60)

hour_tests = [0, 6, 12, 16, 18, 22]
for hour in hour_tests:
    state = {
        'hour': hour,
        'price': 8.0,
        'soc1': 0.5,
        'soc2': 0.5,
        'grid_stress': 0.5
    }
    actions = agent.predict(state)
    print(f"Hour {hour:2d}:00: Bat1={actions[0]:+.3f}, Bat2={actions[1]:+.3f}")

print("\n" + "="*60)
print("REAL DATA STATISTICS")
print("-"*60)
print(f"Price range:  £{data['system_price'].min():.2f} to £{data['system_price'].max():.2f}")
print(f"Price mean:   £{data['system_price'].mean():.2f}")
print(f"Price median: £{data['system_price'].median():.2f}")
print(f"Price std:    £{data['system_price'].std():.2f}")

# Count how often agent charges vs discharges on sample data
print("\n" + "="*60)
print("AGENT BEHAVIOR ON REAL DATA (First 100 periods)")
print("-"*60)

sample = data.head(100)
charge_count = 0
discharge_count = 0
hold_count = 0

for _, row in sample.iterrows():
    state = {
        'hour': row['hour'],
        'price': row['system_price'],
        'soc1': 0.5,
        'soc2': 0.5,
        'grid_stress': row['grid_stress']
    }
    actions = agent.predict(state)
    avg_action = (actions[0] + actions[1]) / 2
    
    if avg_action > 0.1:
        charge_count += 1
    elif avg_action < -0.1:
        discharge_count += 1
    else:
        hold_count += 1

print(f"Charge actions:    {charge_count:3d} ({charge_count/100*100:.1f}%)")
print(f"Discharge actions: {discharge_count:3d} ({discharge_count/100*100:.1f}%)")
print(f"Hold actions:      {hold_count:3d} ({hold_count/100*100:.1f}%)")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if discharge_count > charge_count * 2:
    print("⚠️  PROBLEM: Agent discharges much more than it charges!")
    print("    This explains why batteries drain to 2-3% SOC.")
    print("    Likely cause: Training/testing distribution mismatch.")
elif charge_count < 10:
    print("⚠️  PROBLEM: Agent rarely charges!")
    print("    Agent doesn't respond properly to price signals.")
else:
    print("✓  Agent shows some charging behavior.")
    print("   Problem might be elsewhere (normalization, rewards, etc.)")

print("\n" + "="*60)