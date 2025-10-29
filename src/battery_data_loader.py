import pandas as pd
import os

class MarketData:
    def __init__(self):
        self.data = None
    
    def load_sample_data(self):
        """Create sample price data for testing"""
        # Realistic UK electricity prices (based on historical patterns)
        sample_data = {
            'hour': range(24),
            'price': [35, 32, 30, 28, 25, 24, 26, 35, 45, 50, 55, 60, 
                      65, 70, 75, 80, 85, 90, 95, 100, 90, 75, 60, 45]
        }
        self.data = pd.DataFrame(sample_data)
        return self.data
    
    def get_price_at_hour(self, hour):
        """Get price for a specific hour"""
        if self.data is None:
            self.load_sample_data()
        return self.data.loc[self.data['hour'] == hour, 'price'].values[0]

def main():
    # Test the data loader
    market = MarketData()
    data = market.load_sample_data()
    print("SAMPLE MARKET DATA:")
    print(data.head(10))
    
    # Test price lookup
    print(f"\nPrice at hour 12: £{market.get_price_at_hour(12)}/MWh")
    print(f"Price at hour 18: £{market.get_price_at_hour(18)}/MWh")

if __name__ == "__main__":
    main()