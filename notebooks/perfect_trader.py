import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from battery_model import Battery
from simple_data_loader import MarketData

print("=== PERFECT TRADER ANALYSIS ===")

market = MarketData()
battery = Battery("Perfect_Battery", 50, 100)

print("Calculating optimal trades with perfect foresight...")
print()

# With perfect knowledge of future prices, we can maximize profit
trades = []

# Buy at the absolute lowest price (hour 5: £24/MWh)
battery.charge(50, 4)  # Charge for 4 hours at cheapest price
buy_cost = 24 * 50 * 4 * 0.95  # £24/MWh * 50MW * 4h * efficiency
trades.append(f"BUY: 200 MWh at £24/MWh for £{buy_cost:,.0f}")

# Sell at the absolute highest price (hour 19: £100/MWh)  
battery.discharge(50, 4)  # Discharge for 4 hours at peak price
sell_revenue = 100 * 50 * 4  # £100/MWh * 50MW * 4h
trades.append(f"SELL: 200 MWh at £100/MWh for £{sell_revenue:,.0f}")

perfect_profit = sell_revenue - buy_cost

print("OPTIMAL TRADES WITH PERFECT FORESIGHT:")
for trade in trades:
    print(f"  {trade}")

print(f"\nMAXIMUM THEORETICAL PROFIT: £{perfect_profit:,.0f}")
print(f"Your smarter strategy: £4,693")
print(f"Performance gap: £{perfect_profit - 4693:,.0f} ({(perfect_profit - 4693)/perfect_profit*100:.1f}% of potential)")