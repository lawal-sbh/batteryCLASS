# src/hierarchical_environment_v2.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import the data proxy
from src.uk_market_data_proxy import UKMarketDataProxy

# Keep the ConstraintStabilityCalculator class
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
    def __init__(self, battery_locations=['SCOTLAND', 'LONDON'], use_calibrated_data=True):
        super().__init__()
        
        # Initialize market data proxy
        self.data_proxy = UKMarketDataProxy() if use_calibrated_data else None
        self.use_calibrated_data = use_calibrated_data
        
        # Initialize stability calculator
        self.stability_calc = ConstraintStabilityCalculator(battery_locations)
        self.battery_locations = battery_locations
        self.num_batteries = len(battery_locations)
        
        # Action space: power dispatch for each battery [-1, 1] normalized
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_batteries,), dtype=np.float32)
        
        # State space: [hour, price, soc_battery1, soc_battery2, grid_stress]
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.soc = [0.5, 0.5]
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
        """Get price - uses calibrated data if available, otherwise mock"""
        if self.use_calibrated_data and self.data_proxy:
            return self.data_proxy.get_price_for_hour(self.hour)
        else:
            return self._get_mock_price()
    
    def _get_mock_price(self):
        """Fallback mock prices (original implementation)"""
        if 6 <= self.hour < 16:
            return 45 + np.random.normal(0, 5)
        elif 16 <= self.hour < 22:
            return 85 + np.random.normal(0, 15)
        else:
            return 35 + np.random.normal(0, 3)
    
    def step(self, action):
        # Denormalize actions to MW
        power_dispatch = action * 10
        
        # Calculate economic reward using calibrated prices
        price = self._get_price()
        economic_reward = 0
        for i in range(self.num_batteries):
            economic_reward -= power_dispatch[i] * price * 0.5  # £ for 30-min
        
        # Calculate stability rewards using our calculator
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
        
        # Update SOC (simplified)
        for i in range(self.num_batteries):
            self.soc[i] += power_dispatch[i] * 0.05  # 10 MW * 0.5h = 5 MWh
            self.soc[i] = np.clip(self.soc[i], 0, 1)
        
        # Calculate plan deviation penalty
        plan_deviation_penalty = -10 * sum(abs(self.soc[i] - self.commander_target[i]) for i in range(self.num_batteries))
        
        # Total reward combines all components with balanced weights
        total_reward = economic_reward + weighted_stability + plan_deviation_penalty
        
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
            'soc': self.soc.copy(),
            'battery_locations': self.battery_locations,
            'price': price,  # Add price to info for analysis
            'data_source': 'CALIBRATED_UK_MARKET' if self.use_calibrated_data else 'SYNTHETIC'
        }
        
        return self._get_obs(), total_reward, done, False, info
    
    def get_calibration_report(self):
        """Report on data calibration for thesis"""
        if self.use_calibrated_data and self.data_proxy:
            return {
                'data_source': 'UK Market Data Proxy - Calibrated to Public Reports',
                'calibration_references': self.data_proxy.calibration_sources,
                'real_data_integration_ready': True,
                'compatible_apis': self.data_proxy.get_data_readiness_report()['compatible_apis'],
                'calibration_validation': self.data_proxy.validate_calibration()
            }
        return {'data_source': 'Synthetic data for development'}
    
    def set_commander_target(self, target_soc):
        """Set the commander's SOC targets for both batteries"""
        self.commander_target = target_soc
    
    def render(self):
        """Simple text rendering"""
        print(f"Hour: {self.hour}, SOC: {[f'{s:.2f}' for s in self.soc]}, Total Reward: {self.total_reward:.2f}")
    
    def close(self):
        pass