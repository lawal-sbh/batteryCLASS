#!/usr/bin/env python3
"""
Simple battery testing script - no Jupyter needed
"""

import sys
import os

# Get the parent directory of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')

# Add src directory to Python path
sys.path.insert(0, src_dir)

print(f"Looking for battery_model in: {src_dir}")

# Now import
from battery_model import Battery

print("SUCCESS: Battery class imported!")

# Create different battery types
batteries = {
    "small": Battery("Small_Battery", 10, 20),      # 10MW, 20MWh
    "medium": Battery("Medium_Battery", 50, 100),   # 50MW, 100MWh  
    "large": Battery("Large_Battery", 100, 200)     # 100MW, 200MWh
}

# Show initial status
print("INITIAL STATUS:")
for name, battery in batteries.items():
    status = battery.get_status()
    print(f"{battery.name}: SOC={status['soc']:.2f}, Available={status['available_energy_mwh']} MWh")
print()

# Test charging
print("CHARGING TEST:")
batteries["medium"].charge(30, 2)  # Charge 30MW for 2 hours
status = batteries["medium"].get_status()
print(f"Medium battery after charging: SOC={status['soc']:.2f}")
print()

# Test trading strategy
print("TRADING STRATEGY TEST:")
def simple_trading_strategy(battery, current_price):
    if current_price < 40:
        battery.charge(battery.max_power, 1)
        print(f"  {battery.name}: CHARGING at £{current_price}/MWh → SOC: {battery.soc:.2f}")
    elif current_price > 80:
        battery.discharge(battery.max_power, 1)
        print(f"  {battery.name}: DISCHARGING at £{current_price}/MWh → SOC: {battery.soc:.2f}")
    else:
        print(f"  {battery.name}: HOLDING at £{current_price}/MWh → SOC: {battery.soc:.2f}")

# Test with different prices
test_prices = [35, 45, 85, 60, 25, 95]
for price in test_prices:
    print(f"Price: £{price}/MWh")
    simple_trading_strategy(batteries["small"], price)
print()

# Final status
print("FINAL STATUS:")
for name, battery in batteries.items():
    status = battery.get_status()
    print(f"{battery.name}: SOC={status['soc']:.2f}")