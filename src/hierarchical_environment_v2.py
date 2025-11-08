import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add src to path for direct import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constraint_calculator import ConstraintStabilityCalculator

class HierarchicalBatteryEnv(gym.Env):
    def __init__(self, battery_locations=['SCOTLAND', 'LONDON']):
        super().__init__()
        
        # Initialize stability calculator
        self.stability_calc = ConstraintStabilityCalculator(battery_locations)
        self.battery_locations = battery_locations
        self.num_batteries = len(battery_locations)
        
        # Action space: power dispatch for each battery [-1, 1] normalized
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_batteries,), dtype=np.float32)
        
        # State space: [hour, price, soc_battery1, soc_battery2, commander_target_1, commander_target_2, grid_stress]
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.soc = [0.5, 0.5]  # Start at 50% SOC for both batteries
        self.commander_target = [0.5, 0.5]  # Commander's SOC target
        self.total_reward = 0
        self.economic_reward = 0
        self.stability_reward = 0
        
        observation = self._get_obs()
        info = {}
        return observation, info
    
    def _get_obs(self):
        # Mock data - replace with real feeds later
        price = self._get_mock_price()
        grid_stress = 0.8 if self.hour in [18, 19, 20] else 0.2  # Peak hour stress
        
        return np.array([
            self.hour / 24,  # Normalized hour
            price / 100,     # Normalized price  
            self.soc[0],     # SOC battery 1
            self.soc[1],     # SOC battery 2
            self.commander_target[0],  # Commander target 1
            self.commander_target[1],  # Commander target 2
            grid_stress      # Grid stress level
        ], dtype=np.float32)
    
    def _get_mock_price(self):
        # Simple diurnal price pattern
        if 6 <= self.hour < 16:  # Day
            return 45 + self.np_random.normal(0, 5)
        elif 16 <= self.hour < 22:  # Peak
            return 85 + self.np_random.normal(0, 15)
        else:  # Night
            return 35 + self.np_random.normal(0, 3)
    
    def step(self, action):
        # Denormalize actions to MW
        power_dispatch = action * 10  # ±10 MW max per battery
        
        # Calculate economic reward (simple arbitrage)
        price = self._get_mock_price()
        economic_reward = 0
        for i in range(self.num_batteries):
            economic_reward -= power_dispatch[i] * price * 0.5  # £ for 30-min
        
        # Calculate stability rewards using our new calculator
        grid_conditions = {
            'hour': self.hour,
            'frequency': 49.8 if self.hour in [18, 19] else 50.0,
            'rocof': 0.6 if self.hour in [18, 19] else 0.2
        }
        
        stability_rewards = 0
        for i, location in enumerate(self.battery_locations):
            battery_stability = self.stability_calc.calculate_stability_rewards(
                power_dispatch[i], location, grid_conditions
            )
            stability_rewards += battery_stability['total_stability']
        
        # Update SOC (simplified)
        for i in range(self.num_batteries):
            self.soc[i] += power_dispatch[i] * 0.05  # 10 MW * 0.5h = 5 MWh
            self.soc[i] = np.clip(self.soc[i], 0, 1)
        
        # Calculate plan deviation penalty
        plan_deviation_penalty = -10 * sum(abs(self.soc[i] - self.commander_target[i]) for i in range(self.num_batteries))
        
        # Total reward combines all components
        total_reward = economic_reward + stability_rewards + plan_deviation_penalty
        
        # Update state
        self.hour += 1
        self.total_reward += total_reward
        self.economic_reward += economic_reward
        self.stability_reward += stability_rewards
        
        done = self.hour >= 24
        
        info = {
            'economic': economic_reward,
            'stability': stability_rewards, 
            'plan_deviation': plan_deviation_penalty,
            'soc': self.soc.copy(),
            'battery_locations': self.battery_locations
        }
        
        return self._get_obs(), total_reward, done, False, info
    
    def set_commander_target(self, target_soc):
        """Called by Commander to set SOC targets"""
        self.commander_target = target_soc