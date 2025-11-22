# Data/fixed_real_data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_real_data():
    print("=== PROCESSING REAL UK MARKET DATA ===\n")
    print("Sources: National Grid ESO (demand) & ONS (prices)\n")
    
    try:
        # 1. Load and validate demand data from NESO
        print("1. PROCESSING NESO DEMAND DATA...")
        demand_files = [
            'C:/Users/LOGIN/Desktop/batteryCLASS/Data/demanddata_2023.csv',
            'C:/Users/LOGIN/Desktop/batteryCLASS/Data/demanddata_2024.csv', 
            'C:/Users/LOGIN/Desktop/batteryCLASS/Data/demanddata_2025.csv'
        ]
        
        demand_dfs = []
        for file in demand_files:
            if os.path.exists(file):
                print(f"   Reading {os.path.basename(file)}...")
                df = pd.read_csv(file)
                
                # Validate NESO demand data structure
                required_cols = ['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']
                if all(col in df.columns for col in required_cols):
                    print(f"   ✓ Valid NESO format: {len(df)} rows")
                    demand_dfs.append(df)
                else:
                    print(f"   ⚠️  Missing required columns in {file}")
            else:
                print(f"   ❌ File not found: {file}")
        
        if not demand_dfs:
            raise ValueError("No valid NESO demand data found")
        
        # Combine demand data
        demand = pd.concat(demand_dfs, ignore_index=True)
        print(f"   ✓ Combined NESO demand: {len(demand)} total settlement periods")
        
        # Process NESO timestamps - handle different date formats
        print("   Processing NESO timestamps...")
        
        def parse_neso_date(date_str):
            """Parse NESO dates that have different formats across years"""
            try:
                # Try different date formats found in your data
                for fmt in ['%d-%b-%y', '%d-%b-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%B-%Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                # Fallback to pandas automatic parsing
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT
        
        demand['datetime'] = demand['SETTLEMENT_DATE'].apply(parse_neso_date) + \
                            pd.to_timedelta((demand['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
        
        # Remove invalid timestamps
        initial_count = len(demand)
        demand = demand.dropna(subset=['datetime'])
        print(f"   ✓ Valid timestamps: {len(demand)}/{initial_count} rows")
        
        # 2. Load and validate ONS price data from correct sheet
        print("\n2. PROCESSING ONS PRICE DATA...")
        price_file = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/electricitypricesdataset201125.xlsx'
        
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"ONS price file not found: {price_file}")
        
        print(f"   Reading sheet '1.Daily SP Electricity'...")
        
        # Read the correct sheet with daily prices
        prices = pd.read_excel(price_file, sheet_name='1.Daily SP Electricity')
        print(f"   ✓ Loaded ONS price data: {len(prices)} rows")
        
        # Analyze ONS price data structure
        print(f"   ONS data columns: {list(prices.columns)}")
        print("   First 5 rows of ONS data:")
        print(prices.head())
        
        # 3. Identify datetime and price columns in ONS data
        print("\n3. IDENTIFYING DATA COLUMNS...")
        
        # ONS data should have date and price columns
        # Let's find them automatically
        date_col = None
        price_col = None
        
        for col in prices.columns:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['date', 'time', 'day', 'period']):
                date_col = col
            elif any(x in col_lower for x in ['price', '£', 'gbp', 'value', 'pound']):
                price_col = col
        
        if not date_col or not price_col:
            # If automatic detection fails, use the first two columns
            print("   ⚠️  Could not auto-detect columns, using first two columns")
            date_col = prices.columns[0]
            price_col = prices.columns[1] if len(prices.columns) > 1 else prices.columns[0]
        
        print(f"   Using '{date_col}' as date, '{price_col}' as price")
        
        # Convert ONS datetime and prices
        prices_clean = prices[[date_col, price_col]].copy()
        prices_clean['datetime'] = pd.to_datetime(prices_clean[date_col], errors='coerce')
        prices_clean['price'] = pd.to_numeric(prices_clean[price_col], errors='coerce')
        
        # Remove rows where we couldn't parse date or price
        prices_clean = prices_clean.dropna(subset=['datetime', 'price'])
        print(f"   ✓ Cleaned ONS prices: {len(prices_clean)} valid rows")
        
        # 4. Merge NESO demand with ONS prices
        print("\n4. MERGING NESO & ONS DATA...")
        
        # Prepare demand data for merge - use ND (National Demand) column
        demand_clean = demand[['datetime', 'ND']].copy()
        demand_clean = demand_clean.rename(columns={'ND': 'demand'})
        demand_clean['demand'] = pd.to_numeric(demand_clean['demand'], errors='coerce')
        demand_clean = demand_clean.dropna()
        
        print(f"   Cleaned NESO demand: {len(demand_clean)} rows")
        
        # For ONS daily prices, we need to match with daily demand
        # Let's aggregate demand to daily average for price matching
        demand_daily = demand_clean.copy()
        demand_daily['date'] = demand_daily['datetime'].dt.date
        demand_daily = demand_daily.groupby('date').agg({
            'demand': 'mean'
        }).reset_index()
        demand_daily['datetime'] = pd.to_datetime(demand_daily['date'])
        
        # Prepare ONS prices (already daily)
        prices_daily = prices_clean[['datetime', 'price']].copy()
        prices_daily['date'] = prices_daily['datetime'].dt.date
        
        # Merge on date
        combined_data = pd.merge(demand_daily, prices_daily, on='date', how='inner')
        print(f"   ✓ Merged daily dataset: {len(combined_data)} days with both demand and price")
        
        # 5. Data quality analysis
        print("\n5. DATA QUALITY ANALYSIS:")
        print(f"   Date range: {combined_data['datetime_x'].min()} to {combined_data['datetime_x'].max()}")
        print(f"   Demand range: {combined_data['demand'].min():.0f} - {combined_data['demand'].max():.0f} MW")
        print(f"   Price range: £{combined_data['price'].min():.1f} - £{combined_data['price'].max():.1f}/MWh")
        print(f"   Complete daily records: {len(combined_data)}")
        
        # Check for typical UK price patterns
        price_stats = combined_data['price'].describe()
        print(f"   Price statistics:")
        print(f"     Mean: £{price_stats['mean']:.1f}/MWh")
        print(f"     Std:  £{price_stats['std']:.1f}/MWh")
        print(f"     Max:  £{price_stats['max']:.1f}/MWh")
        
        # 6. Save processed real data
        print("\n6. SAVING PROCESSED REAL DATA...")
        
        # Clean up the combined data
        final_data = combined_data[['datetime_x', 'demand', 'price']].rename(columns={'datetime_x': 'datetime'})
        
        output_file = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/neso_ons_daily_combined.csv'
        final_data.to_csv(output_file, index=False)
        print(f"   ✓ Saved daily combined data: {output_file}")
        
        # Also save the half-hourly demand data for RL environment
        demand_output = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/neso_half_hourly_demand.csv'
        demand_clean.to_csv(demand_output, index=False)
        print(f"   ✓ Saved half-hourly demand data: {demand_output}")
        
        print(f"\n🎉 REAL DATA PROCESSING COMPLETE!")
        print(f"   Daily records: {len(final_data)} days")
        print(f"   Half-hourly demand records: {len(demand_clean)} periods")
        
        return final_data, demand_clean
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Install openpyxl: pip install openpyxl")
        print("2. Check Excel file is not corrupted")
        print("3. Verify sheet names in Excel file")
        
        import traceback
        traceback.print_exc()
        return None, None

def create_rl_training_data(demand_half_hourly, daily_prices):
    """Create training data suitable for RL environment"""
    print("\n7. CREATING RL TRAINING DATA...")
    
    # For RL, we need half-hourly prices. Since ONS gives daily prices,
    # we'll create half-hourly prices using typical UK patterns
    print("   Creating half-hourly price patterns from daily averages...")
    
    # Typical UK half-hourly price pattern (shape)
    typical_pattern = {
        0: 0.8,  1: 0.7,  2: 0.6,  3: 0.6,  4: 0.6,  5: 0.7,    # Night
        6: 0.8,  7: 1.0,  8: 1.2,  9: 1.1,  10: 1.0, 11: 1.0,   # Morning
        12: 1.0, 13: 1.0, 14: 1.0, 15: 1.1, 16: 1.3, 17: 1.5,   # Afternoon
        18: 1.8, 19: 1.6, 20: 1.3, 21: 1.1, 22: 1.0, 23: 0.9    # Evening
    }
    
    # Apply pattern to create half-hourly prices
    rl_data = demand_half_hourly.copy()
    rl_data['date'] = rl_data['datetime'].dt.date
    rl_data['hour'] = rl_data['datetime'].dt.hour
    
    # Merge with daily prices
    daily_prices['date'] = daily_prices['datetime'].dt.date
    rl_data = rl_data.merge(daily_prices[['date', 'price']], on='date', how='left')
    
    # Apply hourly pattern to daily average
    rl_data['price_half_hourly'] = rl_data['price'] * rl_data['hour'].map(typical_pattern)
    
    # Fill any missing prices with reasonable values
    if rl_data['price_half_hourly'].isna().any():
        avg_price = rl_data['price_half_hourly'].mean()
        if pd.isna(avg_price):
            avg_price = 50  # Default £50/MWh
        rl_data['price_half_hourly'] = rl_data['price_half_hourly'].fillna(avg_price)
    
    rl_output = 'C:/Users/LOGIN/Desktop/batteryCLASS/Data/rl_training_data.csv'
    rl_data[['datetime', 'demand', 'price_half_hourly']].to_csv(rl_output, index=False)
    print(f"   ✓ Created RL training data: {rl_output}")
    print(f"   ✓ Half-hourly records: {len(rl_data)}")
    
    return rl_data

if __name__ == "__main__":
    # First install openpyxl if needed
    try:
        import openpyxl
    except ImportError:
        print("Installing openpyxl...")
        os.system("pip install openpyxl")
    
    daily_data, half_hourly_demand = load_real_data()
    
    if daily_data is not None:
        rl_data = create_rl_training_data(half_hourly_demand, daily_data)
        
        print("\n📊 REAL DATA READY FOR:")
        print("1. RL environment training (rl_training_data.csv)")
        print("2. UKMarketDataProxy calibration") 
        print("3. Thesis results validation")
        print("4. Real market behavior analysis")