import sys
import os
sys.path.append(os.path.dirname(__file__))

from multi_objective_environment import MultiObjectiveBatteryEnv
import numpy as np
import gymnasium as gym

class HierarchicalBatteryEnv(MultiObjectiveBatteryEnv):
    """Two-level hierarchical control environment"""
    
    def __init__(self):
        super().__init__()
        # Day-ahead plan storage
        self.day_ahead_schedule = None
        self.day_ahead_planned_soc = [0.5] * 24  # Default: maintain 50% SOC
        
        # Expand observation space to include plan information
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),  # [SOC, price, hour, planned_SOC, plan_deviation]
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _get_state(self):
        """Enhanced state with hierarchical information"""
        base_state = super()._get_state()
        
        if self.day_ahead_schedule is not None and self.current_hour < len(self.day_ahead_planned_soc):
            planned_soc = self.day_ahead_planned_soc[self.current_hour]
            plan_deviation = self.battery.soc - planned_soc
        else:
            planned_soc = 0.5
            plan_deviation = 0.0
            
        return np.array([
            base_state[0],  # Current SOC
            base_state[1],  # Normalized price
            base_state[2],  # Normalized hour
            planned_soc,    # Planned SOC from day-ahead
            plan_deviation  # Deviation from plan
        ], dtype=np.float32)
    
    def set_day_ahead_plan(self, schedule):
        """Set the day-ahead energy schedule"""
        self.day_ahead_schedule = schedule
        # Convert schedule to planned SOC trajectory
        self._calculate_planned_soc(schedule)
    
    def _calculate_planned_soc(self, schedule):
        """Convert energy schedule to SOC trajectory"""
        soc = 0.5  # Start at 50%
        self.day_ahead_planned_soc = [soc]  # Start with initial SOC
        
        for hour, action in enumerate(schedule):
            if action == 1:  # Charge
                energy_added = 50 * 1  # 50MW for 1 hour = 50MWh
                soc = min(1.0, soc + (energy_added / self.battery.capacity))
            elif action == 2:  # Discharge
                energy_removed = 50 * 1  # 50MW for 1 hour = 50MWh  
                soc = max(0.0, soc - (energy_removed / self.battery.capacity))
            # Hold: SOC remains same
            
            # Store SOC for this hour (we already have initial SOC at index 0)
            if hour < len(schedule) - 1:  # Don't go beyond 24 hours
                self.day_ahead_planned_soc.append(soc)
        
        # Ensure we have exactly 24 planned SOC values
        while len(self.day_ahead_planned_soc) < 24:
            self.day_ahead_planned_soc.append(self.day_ahead_planned_soc[-1])
    
    def step(self, action):
        """Hierarchical step with plan adherence reward"""
        state, base_reward, done, truncated, info = super().step(action)
        
        # Add hierarchical reward for following day-ahead plan
        if self.day_ahead_schedule is not None:
            plan_reward = self._calculate_plan_adherence()
            hierarchical_reward = base_reward + plan_reward
        else:
            hierarchical_reward = base_reward
            
        return state, hierarchical_reward, done, truncated, info
    
    def _calculate_plan_adherence(self):
        """Reward for following day-ahead schedule"""
        if self.current_hour >= len(self.day_ahead_schedule):
            return 0.0
            
        planned_action = self.day_ahead_schedule[self.current_hour]
        current_soc = self.battery.soc
        planned_soc = self.day_ahead_planned_soc[self.current_hour]
        
        # Reward for being close to planned SOC
        soc_deviation = abs(current_soc - planned_soc)
        plan_adherence = 1.0 - soc_deviation
        
        return plan_adherence * 0.5  # Weight for plan following

# Day-ahead planner (Commander)
class DayAheadPlanner:
    """High-level planner that creates 24-hour schedules"""
    
    def __init__(self):
        self.forecast_horizon = 24
    
    def create_schedule(self, price_forecast, grid_conditions):
        """Create day-ahead schedule based on forecasts"""
        # Simple rule-based planner (could be replaced with RL)
        schedule = []
        
        for hour in range(24):
            price = price_forecast[hour]
            
            # Simple arbitrage strategy
            if price < 40:  # Cheap - charge
                schedule.append(1)
            elif price > 80:  # Expensive - discharge
                schedule.append(2)
            else:  # Medium - hold
                schedule.append(0)
                
        return schedule
    
    def create_conservative_schedule(self):
        """Conservative schedule that maintains capacity"""
        return [0] * 24  # Hold all day
    
    def create_aggressive_schedule(self):
        """Aggressive trading schedule"""
        return [1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]

if __name__ == "__main__":
    # Test hierarchical environment
    env = HierarchicalBatteryEnv()
    planner = DayAheadPlanner()
    
    # Create day-ahead plan
    schedule = planner.create_schedule(
        price_forecast=[35, 32, 30, 28, 25, 24, 26, 35, 45, 50, 55, 60, 
                       65, 70, 75, 80, 85, 90, 95, 100, 90, 75, 60, 45],
        grid_conditions={}
    )
    
    env.set_day_ahead_plan(schedule)
    state, info = env.reset()
    
    print("Testing Hierarchical Environment...")
    print(f"Day-ahead schedule: {schedule}")
    print(f"Initial state with plan: {state}")
    
    # Test one step
    next_state, reward, done, truncated, info = env.step(1)
    print(f"Hierarchical reward: {reward}")