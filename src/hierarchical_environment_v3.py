# src/hierarchical_environment_v3.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.uk_market_data_proxy_real import UKMarketDataProxyReal

class ImprovedConstraintStabilityCalculator:
    def __init__(self, locations):
        self.locations = locations
        self.tec_rates = self._load_tec_data()
        self.rocof_threshold = 0.5
        
    def _load_tec_data(self):
        return {'SCOTLAND': 15.2, 'LONDON': 3.1}
    
    def calculate_congestion_cost(self, power_mw, location, hour):
        tec_rate = self.tec_rates[location]
        return abs(power_mw) * tec_rate * 0.5
    
    def calculate_inertia_penalty(self, grid_frequency, rocof):
        if rocof > self.rocof_threshold:
            return -1000
        return 0
    
    def calculate_stability_bonus(self, power_dispatch, location, grid_conditions):
        """ADDED: Positive rewards for good grid behavior"""
        bonus = 0
        
        # Bonus for reducing congestion during peak hours
        if 16 <= grid_conditions['hour'] < 20 and power_dispatch < 0:  # Discharging during peak
            bonus += 50 * abs(power_dispatch)  # £50/MW bonus
            
        # Bonus for providing inertia during stress
        if grid_conditions['rocof'] > 0.4 and abs(power_dispatch) > 2:
            bonus += 30  # £30 bonus for quick response
            
        return bonus
    
    def calculate_stability_rewards(self, power_dispatch, location, grid_conditions):
        congestion_cost = self.calculate_congestion_cost(power_dispatch, location, grid_conditions['hour'])
        inertia_penalty = self.calculate_inertia_penalty(grid_conditions['frequency'], grid_conditions['rocof'])
        stability_bonus = self.calculate_stability_bonus(power_dispatch, location, grid_conditions)
        
        return {
            'congestion_cost': -congestion_cost,
            'inertia_penalty': inertia_penalty,
            'stability_bonus': stability_bonus,
            'total_stability': -congestion_cost + inertia_penalty + stability_bonus
        }

class HierarchicalBatteryEnvV3(gym.Env):
    def __init__(self, battery_locations=['SCOTLAND', 'LONDON'], use_calibrated_data=True):
        super().__init__()
        
        # Initialize with REAL data proxy
        self.data_proxy = UKMarketDataProxyReal() if use_calibrated_data else None
        self.use_calibrated_data = use_calibrated_data
        
        # Improved stability calculator with bonuses
        self.stability_calc = ImprovedConstraintStabilityCalculator(battery_locations)
        self.battery_locations = battery_locations
        self.num_batteries = len(battery_locations)
        
        # INCREASED stability weight
        self.stability_weight = 0.02  # Was 0.01
        
        # Action space: power dispatch for each battery [-1, 1] normalized
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_batteries,), dtype=np.float32)
        
        # State space: [hour, price, soc_battery1, soc_battery2, grid_stress]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.soc = [0.5, 0.5]  # Start at 50%
        self.commander_target = [0.5, 0.5]
        self.total_reward = 0
        self.economic_reward = 0
        self.stability_reward = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        price = self._get_price()
        grid_stress = 0.8 if self.hour in [18, 19, 20] else 0.2
        
        return np.array([
            self.hour / 24,
            price / 200,  # Normalize to £200/MWh max
            self.soc[0],
            self.soc[1], 
            grid_stress
        ], dtype=np.float32)
    
    def _get_price(self):
        if self.use_calibrated_data and self.data_proxy:
            return self.data_proxy.get_price_for_hour(self.hour)
        else:
            return self._get_mock_price()
    
    def _get_mock_price(self):
        if 6 <= self.hour < 16:
            return 45 + np.random.normal(0, 5)
        elif 16 <= self.hour < 22:
            return 85 + np.random.normal(0, 15)
        else:
            return 35 + np.random.normal(0, 3)
    
    def step(self, action):
        power_dispatch = action * 10
        
        # Calculate economic reward
        price = self._get_price()
        economic_reward = 0
        for i in range(self.num_batteries):
            economic_reward -= power_dispatch[i] * price * 0.5
        
        # Calculate stability rewards with bonuses
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
        
        # Apply INCREASED stability weight
        weighted_stability = stability_rewards * self.stability_weight
        
        # Update SOC
        for i in range(self.num_batteries):
            self.soc[i] += power_dispatch[i] * 0.05
            self.soc[i] = np.clip(self.soc[i], 0, 1)
        
        # ADDED: SOC penalty for extreme values
        soc_penalty = 0
        for soc in self.soc:
            if soc < 0.1 or soc > 0.9:
                soc_penalty -= 20  # Penalize extreme SOC
        
        # Calculate plan deviation penalty
        plan_deviation_penalty = -10 * sum(abs(self.soc[i] - self.commander_target[i]) for i in range(self.num_batteries))
        
        # Total reward with ALL improvements
        total_reward = economic_reward + weighted_stability + plan_deviation_penalty + soc_penalty
        
        # Update state
        self.hour += 1
        self.total_reward += total_reward
        self.economic_reward += economic_reward
        self.stability_reward += stability_rewards
        
        done = self.hour >= 24
        
        info = {
            'economic': economic_reward,
            'stability': stability_rewards,
            'weighted_stability': weighted_stability,
            'plan_deviation': plan_deviation_penalty,
            'soc_penalty': soc_penalty,
            'soc': self.soc.copy(),
            'battery_locations': self.battery_locations,
            'price': price,
            'data_source': 'CALIBRATED_UK_MARKET' if self.use_calibrated_data else 'SYNTHETIC',
            'stability_weight': self.stability_weight  # Track the weight used
        }
        
        return self._get_obs(), total_reward, done, False, info
    
    # ... keep other methods the same ...