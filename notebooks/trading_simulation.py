import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from battery_model import Battery
from battery_data_loader import MarketData

print("=== REAL-TIME TRADING SIMULATION ===")

# Create market and batteries
market = MarketData()
battery_portfolio = [
    Battery("Trading_Battery_1", 50, 100),  # 50MW, 100MWh
]

print("Starting 24-hour trading simulation...")
print()

# Simulate a full day of trading
for hour in range(24):
    current_price = market.get_price_at_hour(hour)
    battery = battery_portfolio[0]
    
    # Trading decision
    if current_price < 40 and battery.soc < 0.9:
        action = "CHARGE"
        battery.charge(battery.max_power, 1)
    elif current_price > 80 and battery.soc > 0.1:
        action = "DISCHARGE" 
        battery.discharge(battery.max_power, 1)
    else:
        action = "HOLD"
    
    print(f"Hour {hour:2d}: £{current_price:3d}/MWh | {action:10} | SOC: {battery.soc:.2f}")

print(f"\nFinal battery status: {battery.get_status()}")