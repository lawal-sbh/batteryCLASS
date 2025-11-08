# src/hierarchical_environment_v2.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Include the calculator directly to avoid import issues
class ConstraintStabilityCalculator:
    def __init__(self, locations):
        self.locations = locations
        self.tec_rates = self._load_tec_data()
        self.rocof_threshold = 0.5
        
    def _load_tec_data(self):
        return {'SCOTLAND': 12.5, 'LONDON': 2.1}
    
    def calculate_congestion_cost(self, power_mw, location, hour):
        tec_rate = self.tec_rates[location]
        return abs(power_mw) * tec_rate * 0.5
    
    def calculate_inertia_penalty(self, grid_frequency, rocof):
        if rocof > self.rocof_threshold:
            return -1000
        return 0
    
    def calculate_stability_rewards(self, power_dispatch, location, grid_conditions):
        congestion_cost = self.calculate_congestion_cost(power_dispatch, location, grid_conditions['hour'])
        inertia_penalty = self.calculate_inertia_penalty(grid_conditions['frequency'], grid_conditions['rocof'])
        return {
            'congestion_cost': -congestion_cost,
            'inertia_penalty': inertia_penalty,
            'total_stability': -congestion_cost + inertia_penalty
        }

class HierarchicalBatteryEnv(gym.Env):
    def __init__(self, battery_locations=['SCOTLAND', 'LONDON']):
        super().__init__()
        
        # Initialize stability calculator
        self.stability_calc = ConstraintStabilityCalculator(battery_locations)
        self.battery_locations = battery_locations
        self.num_batteries = len(battery_locations)
        
        # Action space: power dispatch for each battery [-1, 1] normalized
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_batteries,), dtype=np.float32)
        
        # State space: [hour, price, soc_battery1, soc_battery2, grid_stress]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Initialize state variables
        self.hour = 0
        self.soc = [0.5, 0.5]
        self.commander_target = [0.5, 0.5]
        self.total_reward = 0
        self.economic_reward = 0
        self.stability_reward = 0
    
    def reset(self, seed=None, options=None):
        # Initialize the random number generator
        super().reset(seed=seed)
        
        # Reset state
        self.hour = 0
        self.soc = [0.5, 0.5]
        self.commander_target = [0.5, 0.5]
        self.total_reward = 0
        self.economic_reward = 0
        self.stability_reward = 0
        
        observation = self._get_obs()
        info = {}
        return observation, info
    
    def _get_obs(self):
        # Mock data
        price = self._get_mock_price()
        grid_stress = 0.8 if self.hour in [18, 19, 20] else 0.2
        
        return np.array([
            self.hour / 24,           # Normalized hour (0-1)
            price / 100,              # Normalized price (0-1)
            self.soc[0],              # SOC battery 1 (0-1)
            self.soc[1],              # SOC battery 2 (0-1)  
            grid_stress               # Grid stress level (0-1)
        ], dtype=np.float32)
    
    def _get_mock_price(self):
        # Simple diurnal price pattern
        if 6 <= self.hour < 16:       # Day
            return 45 + np.random.normal(0, 5)
        elif 16 <= self.hour < 22:    # Peak
            return 85 + np.random.normal(0, 15)
        else:                         # Night
            return 35 + np.random.normal(0, 3)
    
    def step(self, action):
        """
        Execute one time step in the environment
        """
        # Denormalize actions to MW (-10 to +10 MW)
        power_dispatch = action * 10
        
        # Calculate economic reward (simple arbitrage)
        price = self._get_mock_price()
        economic_reward = 0
        for i in range(self.num_batteries):
            economic_reward -= power_dispatch[i] * price * 0.5  # £ for 30-min dispatch
        
        # Calculate stability rewards
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
        
        # Apply stability weight (0.01 from tuning)
        stability_weight = 0.01
        weighted_stability = stability_rewards * stability_weight
        
        # Update SOC (simplified physics)
        for i in range(self.num_batteries):
            self.soc[i] += power_dispatch[i] * 0.05  # 10 MW * 0.5h = 5 MWh
            self.soc[i] = np.clip(self.soc[i], 0, 1)
        
        # Calculate plan deviation penalty
        plan_deviation_penalty = -10 * sum(abs(self.soc[i] - self.commander_target[i]) 
                                         for i in range(self.num_batteries))
        
        # Total reward with balanced weights
        total_reward = economic_reward + weighted_stability + plan_deviation_penalty
        
        # Update episode totals
        self.hour += 1
        self.total_reward += total_reward
        self.economic_reward += economic_reward
        self.stability_reward += stability_rewards
        
        # Check if episode is done
        done = self.hour >= 24
        
        info = {
            'economic': economic_reward,
            'stability': stability_rewards,
            'weighted_stability': weighted_stability,
            'plan_deviation': plan_deviation_penalty,
            'soc': self.soc.copy(),
            'battery_locations': self.battery_locations
        }
        
        return self._get_obs(), total_reward, done, False, info
    
    def set_commander_target(self, target_soc):
        """Set the commander's SOC targets for both batteries"""
        self.commander_target = target_soc
    
    def render(self):
        """Simple text rendering"""
        print(f"Hour: {self.hour}, SOC: {[f'{s:.2f}' for s in self.soc]}, "
              f"Total Reward: {self.total_reward:.2f}")
    
    def close(self):
        pass