import pandas as pd

# See all sheets in the file
excel_file = pd.ExcelFile('C:/Users/LOGIN/Desktop/batteryCLASS/Data/Raw_data/electricitypricesdataset201125.xlsx')
print("Available sheets:", excel_file.sheet_names)

# Load the first sheet to see structure
df = pd.read_excel(excel_file, sheet_name=excel_file.sheet_names[0])
print("\nColumns:", list(df.columns))
print("\nFirst few rows:")
print(df.head())
