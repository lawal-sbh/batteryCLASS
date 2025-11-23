"""
src/baselines/rule_based.py
Simple rule-based battery dispatch for baseline comparison
"""

import numpy as np

class RuleBasedDispatcher:
    """
    Threshold-based battery dispatch
    - Charge when price < 33rd percentile and SOC < 90%
    - Discharge when price > 67th percentile and SOC > 10%
    """
    def __init__(self, price_low_pct=33, price_high_pct=67):
        self.price_low = None
        self.price_high = None
        self.price_low_pct = price_low_pct
        self.price_high_pct = price_high_pct
        
    def fit(self, prices):
        """Learn thresholds from training data"""
        self.price_low = np.percentile(prices, self.price_low_pct)
        self.price_high = np.percentile(prices, self.price_high_pct)
        print(f"Rule-based thresholds: Low=£{self.price_low:.2f}, High=£{self.price_high:.2f}")
        
    def predict(self, state_dict):
        """
        Args:
            state_dict: {'price', 'soc1', 'soc2', ...}
        Returns:
            actions: [battery1_action, battery2_action]
        """
        price = state_dict['price']
        soc1 = state_dict['soc1']
        soc2 = state_dict['soc2']
        
        actions = []
        for soc in [soc1, soc2]:
            if price < self.price_low and soc < 0.9:
                action = 1.0  # Full charge
            elif price > self.price_high and soc > 0.1:
                action = -1.0  # Full discharge
            else:
                action = 0.0  # Hold
            actions.append(action)
            
        return np.array(actions)

if __name__ == "__main__":
    # Test the baseline
    import pandas as pd
    
    data = pd.read_csv('data/uk_battery_dispatch_complete_data.csv')
    baseline = RuleBasedDispatcher()
    baseline.fit(data['system_price'].values)
    
    # Test prediction
    test_state = {
        'price': 50.0,
        'soc1': 0.5,
        'soc2': 0.5,
        'hour': 12,
        'grid_stress': 0.5
    }
    action = baseline.predict(test_state)
    print(f"Test action: {action}")