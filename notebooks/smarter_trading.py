import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from battery_model import Battery
from simple_data_loader import MarketData

print("=== SMARTER TRADING STRATEGY ===")

# Create market and battery
market = MarketData()
battery = Battery("Smart_Battery", 50, 100)

print("24-Hour Price Profile:")
market.print_prices()
print("\nStarting IMPROVED trading simulation...")
print()

# Track performance
revenue = 0
energy_traded = 0

def smarter_trading_strategy(battery, current_price, hour):
    """A strategy that preserves capacity for high-price periods"""
    global revenue, energy_traded
    
    # Price thresholds that adapt based on time of day
    if hour < 12:  # Morning/Overnight - be conservative
        charge_threshold = 35
        discharge_threshold = 85
        target_soc = 0.8  # Keep some room for unexpected opportunities
    else:  # Afternoon/Evening - be aggressive
        charge_threshold = 40
        discharge_threshold = 75
        target_soc = 0.9  # Prepare for peak hours
    
    # Trading logic
    if current_price < charge_threshold and battery.soc < target_soc:
        # Charge, but don't fill completely
        power = min(battery.max_power, (target_soc - battery.soc) * battery.capacity)
        actual_power = battery.charge(power, 1)
        cost = current_price * actual_power
        revenue -= cost
        energy_traded += actual_power
        return f"CHARGE {actual_power}MW"
    
    elif current_price > discharge_threshold and battery.soc > 0.2:
        # Discharge, but keep reserve
        power = min(battery.max_power, (battery.soc - 0.1) * battery.capacity)
        actual_power = battery.discharge(power, 1)
        earnings = current_price * actual_power
        revenue += earnings
        energy_traded += actual_power
        return f"DISCHARGE {actual_power}MW"
    
    else:
        return "HOLD"

# Run the simulation
for hour in range(24):
    current_price = market.get_price_at_hour(hour)
    action = smarter_trading_strategy(battery, current_price, hour)
    
    print(f"Hour {hour:2d}: £{current_price:3d}/MWh | {action:15} | SOC: {battery.soc:.2f}")

print(f"\n=== TRADING RESULTS ===")
print(f"Final SOC: {battery.soc:.2f}")
print(f"Energy Traded: {energy_traded} MWh")
print(f"Estimated Daily Revenue: £{revenue:,.0f}")
print(f"Revenue per MWh: £{revenue/energy_traded if energy_traded > 0 else 0:,.0f}")
print(f"Battery status: {battery.get_status()}")