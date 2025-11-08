import sys
import os
sys.path.append(os.path.dirname(__file__))

from rl_environment import BatteryTradingEnv
import numpy as np
import gymnasium as gym

class MultiObjectiveBatteryEnv(BatteryTradingEnv):
    """Adds grid stability objectives to battery trading"""
    
    def __init__(self, profit_weight=1.0, frequency_weight=0.3, congestion_weight=0.2):
        super().__init__()
        self.weights = {
            'profit': profit_weight,
            'frequency': frequency_weight, 
            'congestion': congestion_weight
        }
        
    def _calculate_frequency_support(self, action):
        """Reward for providing frequency stability"""
        frequency_deviation = self.market.get_frequency_at_hour(self.current_hour)
        
        # Battery helps by charging when frequency high (excess generation)
        # and discharging when frequency low (excess demand)
        if action == 1 and frequency_deviation > 0:  # Charge when frequency high
            return 1.0 - frequency_deviation
        elif action == 2 and frequency_deviation < 0:  # Discharge when frequency low
            return 1.0 - abs(frequency_deviation)
        else:
            return 0.0
    
    def _calculate_congestion_relief(self, action):
        """Reward for reducing grid congestion"""
        congestion = self.market.get_congestion_at_hour(self.current_hour)
        
        # Positive congestion = export constraints (too much generation)
        # Negative congestion = import constraints (too much demand)
        if action == 1 and congestion > 0.5:  # Charge during export constraints
            return congestion
        elif action == 2 and congestion < -0.5:  # Discharge during import constraints
            return abs(congestion)
        else:
            return 0.0
    
    def step(self, action):
        """Multi-objective step function"""
        # Get base economic reward (this also advances time and updates battery)
        current_state = self._get_state()
        profit_reward = 0
        
        # Execute the action and get profit reward
        if action == 1:  # CHARGE
            if self.battery.soc < 0.95:
                power = self.battery.charge(self.battery.max_power, 1)
                cost = self.market.get_price_at_hour(self.current_hour) * power
                self.total_revenue -= cost
                profit_reward = -cost / 1000.0
            else:
                profit_reward = -2
                
        elif action == 2:  # DISCHARGE
            if self.battery.soc > 0.05:
                power = self.battery.discharge(self.battery.max_power, 1)
                earnings = self.market.get_price_at_hour(self.current_hour) * power
                self.total_revenue += earnings
                profit_reward = earnings / 1000.0
            else:
                profit_reward = -2
        else:  # HOLD
            profit_reward = -0.5
        
        # Add grid stability rewards
        frequency_reward = self._calculate_frequency_support(action)
        congestion_reward = self._calculate_congestion_relief(action)
        
        # Combine objectives
        total_reward = (
            profit_reward * self.weights['profit'] +
            frequency_reward * self.weights['frequency'] +
            congestion_reward * self.weights['congestion']
        )
        
        # Move to next hour
        self.current_hour += 1
        done = self.current_hour >= self.max_hours
        
        return self._get_state(), total_reward, done, False, {}

# Test the multi-objective environment
if __name__ == "__main__":
    env = MultiObjectiveBatteryEnv()
    state, info = env.reset()
    print("Testing multi-objective environment...")
    
    # Test grid conditions
    print(f"Initial grid state: Price=£{env.market.get_price_at_hour(0)}, "
          f"Frequency={50+env.market.get_frequency_at_hour(0):.2f}Hz, "
          f"Congestion={env.market.get_congestion_at_hour(0):.1f}")
    
    # Test one step
    next_state, reward, done, truncated, info = env.step(1)
    print(f"Multi-objective reward: {reward}")