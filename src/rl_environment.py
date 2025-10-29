import sys
import os
sys.path.append(os.path.dirname(__file__))

from battery_model import Battery
from simple_data_loader import MarketData
import numpy as np
import gymnasium as gym

class BatteryTradingEnv(gym.Env):
    """Proper Gymnasium environment for battery trading"""
    
    def __init__(self):
        super().__init__()
        self.battery = Battery("RL_Battery", 50, 100)
        self.market = MarketData()
        self.current_hour = 0
        self.max_hours = 24
        self.total_revenue = 0
        
        # Define action and observation space (REQUIRED for Gym)
        self.action_space = gym.spaces.Discrete(3)  # 0=HOLD, 1=CHARGE, 2=DISCHARGE
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),    # min SOC, min price, min hour
            high=np.array([1.0, 1.0, 1.0]),   # max SOC, max price, max hour  
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        self.battery = Battery("RL_Battery", 50, 100)
        self.current_hour = 0
        self.total_revenue = 0
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state representation"""
        return np.array([
            self.battery.soc,  # State of Charge (0-1)
            self.market.get_price_at_hour(self.current_hour) / 100.0,  # Normalized price
            self.current_hour / 24.0,  # Time of day (0-1)
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Execute trading action with energy conservation
        """
        current_price = self.market.get_price_at_hour(self.current_hour)
        reward = 0
        valid_action = True
        
        if action == 1:  # CHARGE
            if self.battery.soc < 0.95:  # Can't charge if nearly full
                power = self.battery.charge(self.battery.max_power, 1)
                cost = current_price * power
                self.total_revenue -= cost
                reward = -cost / 1000.0
            else:
                valid_action = False
                reward = -2  # Penalize invalid action
                
        elif action == 2:  # DISCHARGE
            if self.battery.soc > 0.05:  # Can't discharge if nearly empty
                power = self.battery.discharge(self.battery.max_power, 1)
                earnings = current_price * power
                self.total_revenue += earnings
                reward = earnings / 1000.0
            else:
                valid_action = False
                reward = -2  # Penalize invalid action
        
        # Small incentive for holding at medium prices
        elif action == 0:  # HOLD
            if 40 <= current_price <= 70:  # Hold during medium prices is good
                reward = 0.5
        else:
            reward = -0.5  # Small penalty for holding during extremes
        # Move to next hour
        self.current_hour += 1
        done = self.current_hour >= self.max_hours
        
        return self._get_state(), reward, done, False, {}
    
    def render(self):
        """Display current status"""
        status = self.battery.get_status()
        price = self.market.get_price_at_hour(self.current_hour)
        print(f"Hour {self.current_hour}: £{price}/MWh | SOC: {status['soc']:.2f} | Revenue: £{self.total_revenue:,.0f}")

# Test the environment
if __name__ == "__main__":
    print("Testing Gym Environment...")
    env = BatteryTradingEnv()
    state, info = env.reset()
    
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Initial state:", state)
    
    # Test one step
    next_state, reward, done, truncated, info = env.step(1)  # Charge
    print("After charging:", next_state, "Reward:", reward)