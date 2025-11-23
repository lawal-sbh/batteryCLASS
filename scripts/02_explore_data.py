"""
scripts/02_explore_data.py
Exploratory analysis of UK grid validation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(data_path):
    """Quick EDA on preprocessed data"""
    data = pd.read_csv('C:/Users/LOGIN/Desktop/batteryCLASS/data/uk_battery_dispatch_complete_data.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    print("="*60)
    print("UK GRID DATA SUMMARY")
    print("="*60)
    print(f"\nShape: {data.shape}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"\nColumns: {list(data.columns)}")
    
    # Statistics
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(data[['TSD', 'system_price']].describe())
    
    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Demand time series
    week_data = data.iloc[:48*7]
    axes[0, 0].plot(week_data['datetime'], week_data['TSD'])
    axes[0, 0].set_title('Demand - Sample Week')
    axes[0, 0].set_ylabel('Demand (MW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Price time series
    axes[0, 1].plot(week_data['datetime'], week_data['system_price'], color='orange')
    axes[0, 1].set_title('Price - Sample Week')
    axes[0, 1].set_ylabel('Price (£/MWh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Price distribution
    axes[1, 0].hist(data['system_price'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Price Distribution')
    axes[1, 0].set_xlabel('Price (£/MWh)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Demand distribution
    axes[1, 1].hist(data['TSD'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_title('Demand Distribution')
    axes[1, 1].set_xlabel('Demand (MW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Price vs Hour
    hourly_price = data.groupby('hour')['system_price'].mean()
    axes[2, 0].bar(hourly_price.index, hourly_price.values, alpha=0.7, color='purple')
    axes[2, 0].set_title('Average Price by Hour')
    axes[2, 0].set_xlabel('Hour of Day')
    axes[2, 0].set_ylabel('Avg Price (£/MWh)')
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # 6. Demand vs Hour
    hourly_demand = data.groupby('hour')['TSD'].mean()
    axes[2, 1].bar(hourly_demand.index, hourly_demand.values, alpha=0.7, color='teal')
    axes[2, 1].set_title('Average Demand by Hour')
    axes[2, 1].set_xlabel('Hour of Day')
    axes[2, 1].set_ylabel('Avg Demand (MW)')
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data/figures/data_exploration.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: data/figures/data_exploration.png")
    plt.show()

if __name__ == "__main__":
    explore_data('data/uk_battery_dispatch_complete_data.csv')