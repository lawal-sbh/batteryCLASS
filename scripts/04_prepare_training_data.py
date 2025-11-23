"""
scripts/04_prepare_training_data.py
Prepare real UK data for hierarchical RL training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("="*70)
print("PREPARING TRAINING DATA FROM REAL UK GRID")
print("="*70)

# Load complete data
data = pd.read_csv('data/uk_battery_dispatch_complete_data.csv')
data['datetime'] = pd.to_datetime(data['datetime'])

print(f"\nTotal data: {len(data):,} rows")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

# Calculate grid stress if not present
if 'grid_stress' not in data.columns:
    data['grid_stress'] = (data['TSD'] - data['TSD'].min()) / (data['TSD'].max() - data['TSD'].min())

# Define splits for training
# Training: Jan 2023 - Dec 2023 (12 months) - LOTS OF DATA
# Validation: Jan 2024 - May 2024 (5 months) - For hyperparameter tuning
# Test: Jun 2024 onwards - For final evaluation

train_data = data[data['datetime'] < '2024-01-01']
val_data = data[(data['datetime'] >= '2024-01-01') & (data['datetime'] < '2024-06-01')]
test_data = data[data['datetime'] >= '2024-06-01']

print(f"\n{'='*70}")
print("DATA SPLIT:")
print(f"{'='*70}")
print(f"Training:   {len(train_data):7,} rows ({len(train_data)//48:4,} days) | {train_data['datetime'].min().date()} to {train_data['datetime'].max().date()}")
print(f"Validation: {len(val_data):7,} rows ({len(val_data)//48:4,} days) | {val_data['datetime'].min().date()} to {val_data['datetime'].max().date()}")
print(f"Test:       {len(test_data):7,} rows ({len(test_data)//48:4,} days) | {test_data['datetime'].min().date()} to {test_data['datetime'].max().date()}")

# Create episodes (daily)
def create_episodes(df):
    episodes = []
    for date in df['datetime'].dt.date.unique():
        episode = df[df['datetime'].dt.date == date].copy()
        if len(episode) == 48:  # Complete day
            episodes.append(episode.reset_index(drop=True))
    return episodes

train_episodes = create_episodes(train_data)
val_episodes = create_episodes(val_data)
test_episodes = create_episodes(test_data)

print(f"\n{'='*70}")
print("EPISODES CREATED:")
print(f"{'='*70}")
print(f"Training episodes:   {len(train_episodes):4,}")
print(f"Validation episodes: {len(val_episodes):4,}")
print(f"Test episodes:       {len(test_episodes):4,}")

# Analyze training data statistics
print(f"\n{'='*70}")
print("TRAINING DATA STATISTICS:")
print(f"{'='*70}")
print("\nPrice Distribution:")
print(f"  Min:    £{train_data['system_price'].min():.2f}/MWh")
print(f"  25%:    £{train_data['system_price'].quantile(0.25):.2f}/MWh")
print(f"  Median: £{train_data['system_price'].median():.2f}/MWh")
print(f"  75%:    £{train_data['system_price'].quantile(0.75):.2f}/MWh")
print(f"  Max:    £{train_data['system_price'].max():.2f}/MWh")
print(f"  Mean:   £{train_data['system_price'].mean():.2f}/MWh")
print(f"  Std:    £{train_data['system_price'].std():.2f}/MWh")

print("\nDemand Distribution:")
print(f"  Min:    {train_data['TSD'].min():,.0f} MW")
print(f"  Mean:   {train_data['TSD'].mean():,.0f} MW")
print(f"  Max:    {train_data['TSD'].max():,.0f} MW")

# Save splits
output_dir = Path('data/training')
output_dir.mkdir(parents=True, exist_ok=True)

train_data.to_csv(output_dir / 'train.csv', index=False)
val_data.to_csv(output_dir / 'val.csv', index=False)
test_data.to_csv(output_dir / 'test.csv', index=False)

print(f"\n{'='*70}")
print("✓ Splits saved to: data/training/")
print(f"{'='*70}")

# Calculate normalization parameters from TRAINING data only
norm_params = {
    'price_mean': float(train_data['system_price'].mean()),
    'price_std': float(train_data['system_price'].std()),
    'price_min': float(train_data['system_price'].min()),
    'price_max': float(train_data['system_price'].max()),
    'demand_mean': float(train_data['TSD'].mean()),
    'demand_std': float(train_data['TSD'].std()),
    'demand_min': float(train_data['TSD'].min()),
    'demand_max': float(train_data['TSD'].max()),
}

with open(output_dir / 'normalization_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)

print("\n✓ Normalization parameters saved")
print(f"\nKey parameters:")
print(f"  Price: mean=£{norm_params['price_mean']:.2f}, std=£{norm_params['price_std']:.2f}")
print(f"  Range for normalization: £{norm_params['price_min']:.2f} to £{norm_params['price_max']:.2f}")

print(f"\n{'='*70}")
print("✓✓✓ TRAINING DATA READY!")
print(f"{'='*70}")
print(f"\nNext step: Run training script")
print(f"  python src/training/train_hierarchical.py")