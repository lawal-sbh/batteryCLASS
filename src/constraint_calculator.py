# src/constraint_calculator.py
class ConstraintStabilityCalculator:
    def __init__(self, locations):
        self.locations = locations  # ['SCOTLAND', 'LONDON']
        self.tec_rates = self._load_tec_data()
        self.rocof_threshold = 0.5  # Hz/s
        
    def _load_tec_data(self):
        # Mock TEC rates (£/MWh) - replace with real data later
        return {'SCOTLAND': 12.5, 'LONDON': 2.1}
    
    def calculate_congestion_cost(self, power_mw, location, hour):
        """Calculate congestion cost based on TEC rates"""
        tec_rate = self.tec_rates[location]
        return abs(power_mw) * tec_rate * 0.5  # £ cost for 30-min dispatch
    
    def calculate_inertia_penalty(self, grid_frequency, rocof):
        """Apply penalty for poor frequency stability"""
        if rocof > self.rocof_threshold:
            return -1000  # £ penalty for RoCoF breach
        return 0
    
    def calculate_stability_rewards(self, power_dispatch, location, grid_conditions):
        """Main method called by environment"""
        congestion_cost = self.calculate_congestion_cost(power_dispatch, location, grid_conditions['hour'])
        inertia_penalty = self.calculate_inertia_penalty(grid_conditions['frequency'], grid_conditions['rocof'])
        
        return {
            'congestion_cost': -congestion_cost,  # Negative cost = reward
            'inertia_penalty': inertia_penalty,
            'total_stability': -congestion_cost + inertia_penalty
        }

# Only add this for testing if running this file directly
if __name__ == "__main__":
    # Quick test
    calc = ConstraintStabilityCalculator(['SCOTLAND', 'LONDON'])
    result = calc.calculate_stability_rewards(
        power_dispatch=10,
        location='SCOTLAND', 
        grid_conditions={'hour': 18, 'frequency': 49.8, 'rocof': 0.6}
    )
    print("Direct test:", result)