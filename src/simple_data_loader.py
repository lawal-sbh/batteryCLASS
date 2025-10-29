class MarketData:
    def __init__(self):
        self.prices = [35, 32, 30, 28, 25, 24, 26, 35, 45, 50, 55, 60, 
                       65, 70, 75, 80, 85, 90, 95, 100, 90, 75, 60, 45]
    
    def get_price_at_hour(self, hour):
        """Get price for a specific hour (0-23)"""
        return self.prices[hour % 24]  # Handles hours beyond 24
    
    def print_prices(self):
        print("24-HOUR MARKET PRICES:")
        for hour, price in enumerate(self.prices):
            print(f"  Hour {hour:2d}: £{price}/MWh")

if __name__ == "__main__":
    # Test the simple data loader
    market = MarketData()
    market.print_prices()
    print(f"\nSample price at hour 18: £{market.get_price_at_hour(18)}/MWh")