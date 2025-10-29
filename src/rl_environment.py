import sys
import os
sys.path.append(os.path.dirname(__file__))

from battery_model import Battery
from simple_data_loader import MarketData

class BatteryTradingEnv:
    """Reinforcement Learning environment for battery trading"""
    
    def __init__(self):
        self.battery = Battery("RL_Battery", 50, 100)
        self.market = MarketData()
        self.current_hour = 0
        self.max_hours = 24
        self.total_revenue = 0
        
    def reset(self):
        """Reset environment for new episode"""
        self.battery = Battery("RL_Battery", 50, 100)
        self.current_hour = 0
        self.total_revenue = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation for RL agent"""
        return [
            self.battery.soc,  # State of Charge (0-1)
            self.market.get_price_at_hour(self.current_hour),  # Current price
            self.current_hour / 24.0,  # Time of day (0-1)
        ]
    
    def step(self, action):
        """
        Execute trading action
        Action: 0=HOLD, 1=CHARGE, 2=DISCHARGE
        """
        current_price = self.market.get_price_at_hour(self.current_hour)
        reward = 0
        
        if action == 1:  # CHARGE
            power = self.battery.charge(self.battery.max_power, 1)
            cost = current_price * power
            self.total_revenue -= cost
            reward = -cost  # Negative reward for spending money
            
        elif action == 2:  # DISCHARGE
            power = self.battery.discharge(self.battery.max_power, 1)
            earnings = current_price * power
            self.total_revenue += earnings
            reward = earnings  # Positive reward for making money
        
        # Move to next hour
        self.current_hour += 1
        done = self.current_hour >= self.max_hours
        
        return self._get_state(), reward, done, {}
    
    def render(self):
        """Display current status"""
        status = self.battery.get_status()
        price = self.market.get_price_at_hour(self.current_hour)
        print(f"Hour {self.current_hour}: £{price}/MWh | SOC: {status['soc']:.2f} | Revenue: £{self.total_revenue:,.0f}")

# Test the environment
if __name__ == "__main__":
    print("Testing RL Environment...")
    env = BatteryTradingEnv()
    state = env.reset()
    
    print("Initial state:", state)
    
    # Test one step
    next_state, reward, done, info = env.step(1)  # Charge
    print("After charging:", next_state, "Reward:", reward)
    
    next_state, reward, done, info = env.step(2)  # Discharge
    print("After discharging:", next_state, "Reward:", reward)