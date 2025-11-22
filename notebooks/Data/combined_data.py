import pandas as pd
import numpy as np

print("=== COMBINING ALL DATA ===\n")

# Step 1: Load all your demand files
print("Loading demand data...")
demand_files = [
    'C:/Users/LOGIN/Desktop/batteryCLASS/Data/Raw_data/demanddata_2023.csv',
    'C:/Users/LOGIN/Desktop/batteryCLASS/Data/Raw_data/demanddata_2024.csv', 
    'C:/Users/LOGIN/Desktop/batteryCLASS/Data/Raw_data/demanddata_2025.csv'
]

demand_list = []
for file in demand_files:
    df = pd.read_csv(file)
    print(f"  ✓ Loaded {file.split('/')[-1]}: {len(df)} rows")
    demand_list.append(df)

# Combine all demand data
demand = pd.concat(demand_list, ignore_index=True)
print(f"\n✓ Combined demand data: {len(demand)} total rows")

# Step 2: Create proper datetime column
print("\nProcessing timestamps...")
demand['SETTLEMENT_DATE'] = pd.to_datetime(demand['SETTLEMENT_DATE'], format='mixed', dayfirst=True)
demand['datetime'] = demand['SETTLEMENT_DATE'] + \
                     pd.to_timedelta((demand['SETTLEMENT_PERIOD'] - 1) * 30, unit='min')

# Step 3: Load price data from ONS Excel (CORRECT METHOD)
print("\nLoading price data from ONS Excel...")

# The data is in sheet '1.Daily SP Electricity' with header rows to skip
prices = pd.read_excel(
    'C:/Users/LOGIN/Desktop/batteryCLASS/Data/Raw_data/electricitypricesdataset201125.xlsx',
    sheet_name='1.Daily SP Electricity',  # The daily data sheet
    skiprows=7  # Skip the metadata rows (rows 0-6)
)

print(f"  ✓ Loaded prices: {len(prices)} rows")
print(f"  ✓ Price columns: {list(prices.columns)}")

# Check first few rows to understand structure
print("\nFirst few rows of price data:")
print(prices.head())

# The ONS format typically has columns like:
# 'Date', 'System Price (£/MWh)', '7-day rolling average'
# Let's find and rename them

# Find date column
date_col = None
for col in prices.columns:
    if 'date' in str(col).lower():
        date_col = col
        break
if date_col is None:
    date_col = prices.columns[0]  # First column is usually date

# Find price column (not the rolling average)
price_col = None
for col in prices.columns:
    col_lower = str(col).lower()
    if 'system price' in col_lower and 'rolling' not in col_lower and 'average' not in col_lower:
        price_col = col
        break
if price_col is None:
    price_col = prices.columns[1]  # Second column usually price

print(f"\n  ✓ Using date column: '{date_col}'")
print(f"  ✓ Using price column: '{price_col}'")

# Create clean dataframe with just date and price
prices_clean = pd.DataFrame({
    'date': pd.to_datetime(prices[date_col]),
    'system_price': pd.to_numeric(prices[price_col], errors='coerce')
})

# Remove any rows with NaN prices (from header remnants or empty rows)
prices_clean = prices_clean.dropna(subset=['system_price'])

print(f"\n  ✓ Clean price data: {len(prices_clean)} rows")
print(f"  ✓ Date range: {prices_clean['date'].min()} to {prices_clean['date'].max()}")
print(f"  ✓ Price range: £{prices_clean['system_price'].min():.2f} to £{prices_clean['system_price'].max():.2f} per MWh")

# Step 4: Merge demand and prices
print("\nMerging demand and price data...")

# Get date from datetime for merging
demand['date'] = demand['datetime'].dt.normalize()  # Normalize to midnight
prices_clean['date'] = prices_clean['date'].dt.normalize()

# Merge on date (each day gets the same daily price for all 48 periods)
data_halfhourly = demand.merge(prices_clean, on='date', how='left')

print(f"  ✓ Merged dataset: {len(data_halfhourly)} rows")

# Check how many rows have prices
rows_with_prices = data_halfhourly['system_price'].notna().sum()
rows_without_prices = data_halfhourly['system_price'].isna().sum()
print(f"  ✓ Rows with prices: {rows_with_prices:,}")
if rows_without_prices > 0:
    print(f"  ⚠ Rows without prices: {rows_without_prices:,} (will be removed)")

# Step 5: Create RL features
print("\nCreating RL features...")
data_halfhourly['hour'] = data_halfhourly['datetime'].dt.hour
data_halfhourly['minute'] = data_halfhourly['datetime'].dt.minute
data_halfhourly['day_of_week'] = data_halfhourly['datetime'].dt.dayofweek
data_halfhourly['month'] = data_halfhourly['datetime'].dt.month
data_halfhourly['day_of_year'] = data_halfhourly['datetime'].dt.dayofyear
data_halfhourly['is_weekend'] = (data_halfhourly['day_of_week'] >= 5).astype(int)
data_halfhourly['is_peak'] = ((data_halfhourly['hour'] >= 16) & 
                               (data_halfhourly['hour'] <= 20)).astype(int)

# Sort by datetime before calculating changes
data_halfhourly = data_halfhourly.sort_values('datetime').reset_index(drop=True)

# Calculate changes (for RL state)
data_halfhourly['demand_change'] = data_halfhourly['TSD'].diff()
data_halfhourly['price_change'] = data_halfhourly['system_price'].diff()

# Step 6: Select and clean final dataset
print("\nFinalizing dataset...")

# Select final columns
final_columns = [
    'datetime', 'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD',
    'TSD', 'ND', 'EMBEDDED_WIND_GENERATION', 'EMBEDDED_SOLAR_GENERATION',
    'system_price',
    'hour', 'minute', 'day_of_week', 'month', 'day_of_year',
    'is_weekend', 'is_peak',
    'demand_change', 'price_change'
]

# Keep only columns that exist
final_columns = [col for col in final_columns if col in data_halfhourly.columns]
data_final = data_halfhourly[final_columns].copy()

# Remove rows with missing prices
rows_before = len(data_final)
data_final = data_final.dropna(subset=['system_price'])
rows_after = len(data_final)

if rows_before > rows_after:
    print(f"  ✓ Removed {rows_before - rows_after:,} rows with missing prices")

# Remove first row if it has NaN from diff() calculations
if data_final.iloc[0]['demand_change'] != data_final.iloc[0]['demand_change']:  # Check for NaN
    data_final = data_final.iloc[1:].reset_index(drop=True)

# Save the complete dataset
output_file = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/uk_battery_dispatch_complete_data.csv'
data_final.to_csv(output_file, index=False)

# Print summary
print(f"\n{'='*60}")
print(f"{'✓✓✓ SUCCESS! ✓✓✓':^60}")
print(f"{'='*60}")
print(f"\n📁 OUTPUT FILE:")
print(f"   {output_file}")
print(f"\n📊 DATASET SUMMARY:")
print(f"   Total rows:        {len(data_final):,}")
print(f"   Total columns:     {len(data_final.columns)}")
print(f"   Date range:        {data_final['datetime'].min()} to {data_final['datetime'].max()}")
print(f"   Total days:        {data_final['datetime'].dt.date.nunique():,}")
print(f"   Settlement periods per day: {len(data_final) / data_final['datetime'].dt.date.nunique():.1f}")

print(f"\n📋 COLUMNS ({len(data_final.columns)}):")
for i, col in enumerate(data_final.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n📈 KEY STATISTICS:")
stats = data_final[['TSD', 'system_price']].describe()
print(f"\n   Demand (TSD):")
print(f"      Mean:  {stats.loc['mean', 'TSD']:,.0f} MW")
print(f"      Min:   {stats.loc['min', 'TSD']:,.0f} MW")
print(f"      Max:   {stats.loc['max', 'TSD']:,.0f} MW")
print(f"\n   System Price:")
print(f"      Mean:  £{stats.loc['mean', 'system_price']:,.2f} /MWh")
print(f"      Min:   £{stats.loc['min', 'system_price']:,.2f} /MWh")
print(f"      Max:   £{stats.loc['max', 'system_price']:,.2f} /MWh")

print(f"\n🔍 SAMPLE DATA (First 10 rows):")
print(f"{'='*60}")
print(data_final[['datetime', 'TSD', 'system_price', 'hour', 'is_peak']].head(10).to_string(index=False))

print(f"\n✅ DATA QUALITY:")
missing = data_final.isnull().sum()
if missing.sum() == 0:
    print(f"   ✓ No missing values!")
else:
    print(f"   ⚠ Missing values found:")
    for col, count in missing[missing > 0].items():
        print(f"      {col}: {count}")

print(f"\n{'='*60}")
print(f"{'✓✓✓ READY FOR RL AGENT VALIDATION! ✓✓✓':^60}")
print(f"{'='*60}")

# Save a quick reference file with date range info
info_file = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/dataset_info.txt'
with open(info_file, 'w') as f:
    f.write("UK Battery Dispatch Dataset Information\n")
    f.write("="*50 + "\n\n")
    f.write(f"Generated: {pd.Timestamp.now()}\n")
    f.write(f"Total Rows: {len(data_final):,}\n")
    f.write(f"Date Range: {data_final['datetime'].min()} to {data_final['datetime'].max()}\n")
    f.write(f"Total Days: {data_final['datetime'].dt.date.nunique():,}\n")
    f.write(f"\nColumns:\n")
    for i, col in enumerate(data_final.columns, 1):
        f.write(f"  {i:2d}. {col}\n")

print(f"\n📄 Info file saved: {info_file}")

