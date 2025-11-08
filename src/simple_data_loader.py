class MarketData:
    def __init__(self):
        # Existing price data
        self.prices = [35, 32, 30, 28, 25, 24, 26, 35, 45, 50, 55, 60, 
                       65, 70, 75, 80, 85, 90, 95, 100, 90, 75, 60, 45]
        
        # NEW: Grid stability metrics (based on UK typical patterns)
        # Frequency deviation from 50Hz (UK target)
        self.frequency_deviation = [0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.1,
                                   0.15, 0.2, 0.25, 0.3, 0.25, 0.2, 0.15, 0.1,
                                   0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02]
        
        # NEW: Grid congestion (1.0 = severe export constraint, -1.0 = severe import constraint)
        self.congestion = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6,
                          0.8, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3,
                          -0.5, -0.7, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0]
    
    def get_price_at_hour(self, hour):
        """Get price for a specific hour (0-23)"""
        return self.prices[hour % 24]
    
    # NEW METHODS FOR GRID STABILITY
    def get_frequency_at_hour(self, hour):
        """Get frequency deviation from 50Hz"""
        return self.frequency_deviation[hour % 24]
    
    def get_congestion_at_hour(self, hour):
        """Get grid congestion level"""
        return self.congestion[hour % 24]
    
    def print_grid_conditions(self):
        print("24-HOUR GRID CONDITIONS:")
        for hour in range(24):
            print(f"  Hour {hour:2d}: £{self.prices[hour]:3d}/MWh | "
                  f"Freq: {50+self.frequency_deviation[hour]:.2f}Hz | "
                  f"Congestion: {self.congestion[hour]:.1f}")

if __name__ == "__main__":
    # Test the enhanced data
    market = MarketData()
    market.print_grid_conditions()