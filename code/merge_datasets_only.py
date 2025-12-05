import pandas as pd
import numpy as np

# Load master dataset (stablecoin adoption)
df_master = pd.read_excel("complete_crypto_remittance_data.xlsx", 
                          sheet_name="MASTER DATASET")
print(f"✓ Loaded Master Dataset: {df_master.shape[0]} rows × {df_master.shape[1]} columns")

# Load World Bank macro data
df_wb = pd.read_csv("worldbank_economic_data_2015_2024.csv")
print(f"✓ Loaded World Bank Data: {df_wb.shape[0]} rows × {df_wb.shape[1]} columns")

# Load Chainalysis (reference, not merged)
df_chain = pd.read_excel("chainalysis_extracted_data.xlsx", 
                         sheet_name="Remittance Costs")
print(f"✓ Loaded Chainalysis Cost Data: {df_chain.shape[0]} rows × {df_chain.shape[1]} columns")

print()


# Select only needed columns
df_master_clean = df_master[['Country', 'Year', 'Stablecoin_Share_Pct', 
                              'Total_Remittances_USD_millions']].copy()

# Drop rows where Stablecoin_Share_Pct is missing (needed as DV)
df_master_clean = df_master_clean.dropna(subset=['Stablecoin_Share_Pct'])

print(f"After cleaning: {df_master_clean.shape[0]} rows")
print(f"Years: {sorted(df_master_clean['Year'].unique())}")
print(f"Countries: {sorted(df_master_clean['Country'].unique())}")

print("\nMaster Dataset (cleaned):")
print(df_master_clean.to_string())

print()


# Standardize column names for merge
df_wb_clean = df_wb.rename(columns={'country': 'Country', 'year': 'Year'}).copy()

# Select only the columns you need for regression
needed_wb_cols = ['Country', 'Year', 'inflation_rate', 'exchange_rate_lcu_per_usd',
                  'internet_users_pct', 'mobile_subscriptions_per100', 'gdp_growth_pct',
                  'remittances_pct_gdp']

df_wb_clean = df_wb_clean[needed_wb_cols].copy()

# Filter to 2021-2024 (when stablecoin data exists)
df_wb_clean = df_wb_clean[(df_wb_clean['Year'] >= 2021) & (df_wb_clean['Year'] <= 2024)]

print(f"World Bank data (2021-2024): {df_wb_clean.shape[0]} rows")
print(f"Countries available: {sorted(df_wb_clean['Country'].unique())}")
print(f"Years available: {sorted(df_wb_clean['Year'].unique())}")

print("\nWorld Bank Data (2021-2024, sample):")
print(df_wb_clean.head(10).to_string())

print()


# Merge Master Dataset with World Bank on Country and Year
df_merged = df_master_clean.merge(df_wb_clean, 
                                  on=['Country', 'Year'], 
                                  how='left')

print(f"Merged dataset: {df_merged.shape[0]} rows × {df_merged.shape[1]} columns")

print("\nMerged dataset (full view):")
print(df_merged.to_string())

print()



print("\nMissing values by column:")
missing_counts = df_merged.isnull().sum()
for col in missing_counts.index:
    if missing_counts[col] > 0:
        pct = (missing_counts[col] / len(df_merged)) * 100
        print(f"  {col}: {missing_counts[col]} missing ({pct:.1f}%)")

if missing_counts.sum() == 0:
    print("  ✓ No missing values!")



# Save to CSV
output_file = "merged_dataset_for_regression.csv"
df_merged.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

print("\nMerged dataset structure:")
print(f"  Rows: {df_merged.shape[0]} (observations)")
print(f"  Columns: {df_merged.shape[1]}")
print(f"  Countries: {df_merged['Country'].nunique()}")
print(f"  Years: {len(df_merged['Year'].unique())}")