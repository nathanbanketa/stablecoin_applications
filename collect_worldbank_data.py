"""
Script to collect economic data from World Bank for stablecoin research
Covers: GDP, inflation, remittances, exchange rates, financial inclusion, internet
Period: 2015-2024
Countries: Argentina, Nigeria, Ethiopia, Philippines, Turkey, Vietnam
"""

import wbdata
import pandas as pd
from datetime import datetime

def collect_worldbank_data():
    """Collect all necessary economic indicators from World Bank"""
    
    # Country codes
    countries = ['AR', 'NG', 'ET', 'PH', 'TR', 'VN']
    
    # Define indicators
    indicators = {
        # GDP metrics
        'NY.GDP.MKTP.CD': 'gdp_usd',
        'NY.GDP.PCAP.CD': 'gdp_per_capita_usd',
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth_pct',
        
        # Inflation
        'FP.CPI.TOTL.ZG': 'inflation_rate',
        
        # Remittances
        'BX.TRF.PWKR.DT.GD.ZS': 'remittances_pct_gdp',
        'BX.TRF.PWKR.CD.DT': 'remittances_received_usd',
        
        # Exchange rates
        'PA.NUS.FCRF': 'exchange_rate_lcu_per_usd',
        
        # Financial inclusion & technology
        'IT.NET.USER.ZS': 'internet_users_pct',
        'IT.CEL.SETS.P2': 'mobile_subscriptions_per100',
        
        # Remittance costs
        'SI.RMT.COST.OB.ZS': 'remittance_cost_pct_sending_200usd'
    }
    
    print("Fetching data from World Bank API...")
    print(f"Countries: {countries}")
    print(f"Indicators: {len(indicators)}")
    print(f"Date range: 2015-2024\n")
    
    # Fetch data
    try:
        data = wbdata.get_dataframe(
            indicators,
            country=countries
        )
        
        # Reset index to get country and date as columns
        data = data.reset_index()
        
        # Rename 'date' to 'year' for clarity
        if 'date' in data.columns:
            data = data.rename(columns={'date': 'year'})
        
        # Convert year to integer if it's a date
        if data['year'].dtype == 'object' or 'datetime' in str(data['year'].dtype):
            data['year'] = pd.to_datetime(data['year']).dt.year
        
        # Add country names
        country_map = {
            'AR': 'Argentina',
            'NG': 'Nigeria',
            'ET': 'Ethiopia',
            'PH': 'Philippines',
            'TR': 'Turkey',
            'VN': 'Vietnam'
        }
        
        data['country_name'] = data['country'].map(country_map)
        
        # Reorder columns
        cols = ['country_name', 'country', 'year'] + [col for col in data.columns if col not in ['country_name', 'country', 'year']]
        data = data[cols]
        
        # Sort by country and year
        data = data.sort_values(['country_name', 'year'])
        
        print(f"✓ Data fetched successfully!")
        print(f"  Shape: {data.shape}")
        print(f"  Years: {data['year'].min()} to {data['year'].max()}")
        print(f"  Countries: {data['country_name'].unique()}\n")
        
        return data
        
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None

def analyze_data_completeness(df):
    """Analyze and report data completeness"""
    
    print("="*70)
    print("DATA COMPLETENESS ANALYSIS")
    print("="*70)
    
    for country in df['country_name'].unique():
        country_data = df[df['country_name'] == country]
        print(f"\n{country}:")
        print(f"  Years covered: {country_data['year'].min()} - {country_data['year'].max()}")
        print(f"  Total rows: {len(country_data)}")
        
        # Check completeness of key variables
        key_vars = [
            'gdp_usd', 'inflation_rate', 'remittances_pct_gdp',
            'exchange_rate_lcu_per_usd', 'internet_users_pct'
        ]
        
        for var in key_vars:
            if var in country_data.columns:
                available = country_data[var].notna().sum()
                pct = (available / len(country_data)) * 100
                print(f"    {var}: {available}/{len(country_data)} ({pct:.1f}%)")
    
    print("\n" + "="*70)
    print("OVERALL MISSING DATA")
    print("="*70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in df.columns:
        if col not in ['country_name', 'country', 'year']:
            print(f"{col:40s}: {missing[col]:3d} missing ({missing_pct[col]:5.1f}%)")

def save_data(df, filename='worldbank_economic_data_2015_2024.csv'):
    """Save data to CSV"""
    df.to_csv(filename, index=False)
    print(f"\n✓ Data saved to: {filename}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

if __name__ == '__main__':
    # Collect data
    df = collect_worldbank_data()
    
    if df is not None:
        # Analyze completeness
        analyze_data_completeness(df)
        
        # Save to CSV
        save_data(df)
        
        # Display sample
        print("\n" + "="*70)
        print("SAMPLE DATA (First 10 rows)")
        print("="*70)
        print(df.head(10).to_string())
        
        print("\n✓ Script completed successfully!")
        print("\nNext steps:")
        print("1. Review the CSV file")
        print("2. Check data completeness for your target countries")
        print("3. Collect crypto adoption data from Chainalysis")
        print("4. Merge datasets for analysis")
