import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')



# Load your merged dataset
df = pd.read_csv("merged_dataset_for_regression.csv")

print(f"✓ Loaded merged dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nData:")
print(df.to_string())
print()



countries = df['Country'].unique()
print(f"\nCountries available: {sorted(countries)}")
print()

# Create a dictionary to store time series for each country
time_series_data = {}

for country in sorted(countries):
    country_data = df[df['Country'] == country].sort_values('Year')
    
    # Create time series (year as index, adoption as values)
    ts = country_data.set_index('Year')['Stablecoin_Share_Pct'].dropna()
    
    time_series_data[country] = ts
    
    print(f"{country}:")
    print(f"  Years: {ts.index.tolist()}")
    print(f"  Values: {ts.values}")
    print(f"  Observations: {len(ts)}")
    print()


adf_results = {}

for country, ts in time_series_data.items():
    print(f"{country} (n={len(ts)} observations):")
    
    if len(ts) >= 4:  # ADF test requires at least 4 observations
        try:
            result = adfuller(ts, autolag='AIC')
            adf_results[country] = result
            
            print(f"  ADF Statistic: {result[0]:.4f}")
            print(f"  P-value: {result[1]:.4f}")
            print(f"  Critical Values: {result[4]}")
            
            if result[1] < 0.05:
                print(f"  ✓ STATIONARY (reject null hypothesis)")
                print(f"    → Use ARIMA with d=0")
            else:
                print(f"  ✗ NON-STATIONARY (fail to reject null hypothesis)")
                print(f"    → Use ARIMA with d=1 (first difference)")
        except Exception as e:
            print(f"  ⚠️  ADF test failed: {str(e)}")
            print(f"  → Using default d=1 (first differencing)")
            adf_results[country] = None
    else:
        print(f"  ⚠️  Too few observations for ADF test (need ≥4, have {len(ts)})")
        print(f"  → Using default d=1 (first differencing)")
        adf_results[country] = None
    
    print()

# Only create plots if we have data to plot
countries_with_data = [c for c in time_series_data.keys() if len(time_series_data[c]) >= 2]

if len(countries_with_data) > 0:
    fig, axes = plt.subplots(len(countries_with_data), 2, figsize=(14, 4*len(countries_with_data)))

    if len(countries_with_data) == 1:
        axes = axes.reshape(1, -1)

    for idx, country in enumerate(sorted(countries_with_data)):
        ts = time_series_data[country]
        
        if len(ts) >= 2:
            # Determine max lags based on data size
            max_lags = min(3, len(ts) - 1)
            
            try:
                # ACF plot
                plot_acf(ts, lags=max_lags, ax=axes[idx, 0])
                axes[idx, 0].set_title(f'{country} - ACF (n={len(ts)})')
            except Exception as e:
                axes[idx, 0].text(0.5, 0.5, f'ACF plot failed\n{str(e)[:50]}', 
                                 ha='center', va='center')
                axes[idx, 0].set_title(f'{country} - ACF (n={len(ts)})')
            
            try:
                # PACF plot
                if len(ts) >= 2:
                    plot_pacf(ts, lags=max_lags, ax=axes[idx, 1], method='ywm')
                else:
                    axes[idx, 1].text(0.5, 0.5, 'Not enough data for PACF', 
                                     ha='center', va='center')
                axes[idx, 1].set_title(f'{country} - PACF (n={len(ts)})')
            except Exception as e:
                axes[idx, 1].text(0.5, 0.5, f'PACF plot failed\n{str(e)[:50]}', 
                                 ha='center', va='center')
                axes[idx, 1].set_title(f'{country} - PACF (n={len(ts)})')

    plt.tight_layout()
    plt.savefig('acf_pacf_plots.png', dpi=300)
    print("✓ Saved: acf_pacf_plots.png")
else:
    print("⚠️  No data available for ACF/PACF plots")

print()


arima_models = {}
arima_results = {}

for country, ts in time_series_data.items():
    if len(ts) >= 2:
        print(f"Fitting ARIMA for {country} (n={len(ts)} observations)...")
        
        # For very small datasets, use simple ARIMA
        # p=1, d=0 or 1, q=1 are good defaults
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fitted = model.fit()
            arima_models[country] = model
            arima_results[country] = fitted
            
            print(f"  ✓ ARIMA(1,1,1) fitted successfully")
            print(f"    AIC: {fitted.aic:.2f}")
            print(f"    BIC: {fitted.bic:.2f}")
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Trying ARIMA(0,1,1)...")
            try:
                model = ARIMA(ts, order=(0, 1, 1))
                fitted = model.fit()
                arima_models[country] = model
                arima_results[country] = fitted
                print(f"  ✓ ARIMA(0,1,1) fitted successfully")
                print(f"    AIC: {fitted.aic:.2f}")
                print()
            except:
                print(f"  ✗ Could not fit ARIMA for {country}")
                print()


for country, fitted in arima_results.items():
    print(f"{country}:")
    print(f"\n{fitted.summary()}")
    print()
    
    # Try to create diagnostic plots (may fail with very small datasets)
    try:
        fig = fitted.plot_diagnostics(figsize=(12, 8))
        if hasattr(fig, 'suptitle'):
            fig.suptitle(f'{country} - ARIMA Diagnostics', fontsize=14, fontweight='bold')
        else:
            plt.suptitle(f'{country} - ARIMA Diagnostics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'diagnostics_{country}.png', dpi=300)
        print(f"✓ Saved: diagnostics_{country}.png")
        plt.close()
    except (ValueError, Exception) as e:
        print(f"⚠️  Diagnostic plots skipped (insufficient data for {country})")
        print(f"    This is normal for datasets with < 4 observations")
    
    print()


forecast_years = [2025, 2026, 2027]
forecast_results = {}

for country, fitted in arima_results.items():
    print(f"{country}:")
    
    # Forecast 3 steps ahead
    forecast = fitted.get_forecast(steps=3)
    forecast_df = forecast.summary_frame()
    
    print(f"  Point Forecast:")
    for idx, year in enumerate(forecast_years):
        mean = forecast_df['mean'].iloc[idx]
        ci_lower = forecast_df['mean_ci_lower'].iloc[idx]
        ci_upper = forecast_df['mean_ci_upper'].iloc[idx]
        
        print(f"    {year}: {mean:.1f}% (95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])")
    
    forecast_results[country] = forecast_df
    print()



fig, axes = plt.subplots(len(time_series_data), 1, figsize=(12, 4*len(time_series_data)))

if len(time_series_data) == 1:
    axes = [axes]

for idx, (country, ts) in enumerate(sorted(time_series_data.items())):
    ax = axes[idx]
    
    # Plot historical data
    ax.plot(ts.index, ts.values, 'bo-', linewidth=2, markersize=8, label='Observed')
    
    # Plot forecast
    if country in forecast_results:
        forecast_df = forecast_results[country]
        forecast_years_list = [max(ts.index) + i + 1 for i in range(len(forecast_df))]
        
        # Point forecast
        ax.plot(forecast_years_list, forecast_df['mean'], 'r^--', linewidth=2, 
               markersize=8, label='Forecast')
        
        # Confidence interval
        ax.fill_between(forecast_years_list,
                        forecast_df['mean_ci_lower'],
                        forecast_df['mean_ci_upper'],
                        alpha=0.3, color='red', label='95% CI')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Stablecoin Share (%)', fontsize=12)
    ax.set_title(f'{country} - Stablecoin Adoption: Historical & Forecast', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('time_series_forecasts.png', dpi=300)
print("✓ Saved: time_series_forecasts.png")


summary_data = []

for country in sorted(time_series_data.keys()):
    ts = time_series_data[country]
    
    # Current data
    current_year = ts.index[-1]
    current_value = ts.values[-1]
    
    # Forecast
    if country in forecast_results:
        forecast_2027 = forecast_results[country]['mean'].iloc[2]  # 3rd year (2027)
        forecast_2027_ci_lower = forecast_results[country]['mean_ci_lower'].iloc[2]
        forecast_2027_ci_upper = forecast_results[country]['mean_ci_upper'].iloc[2]
        
        change = forecast_2027 - current_value
        pct_change = (change / current_value) * 100 if current_value != 0 else 0
    else:
        forecast_2027 = np.nan
        forecast_2027_ci_lower = np.nan
        forecast_2027_ci_upper = np.nan
        change = np.nan
        pct_change = np.nan
    
    summary_data.append({
        'Country': country,
        'Current Year': int(current_year),
        'Current Value (%)': f"{current_value:.1f}",
        'Forecast 2027 (%)': f"{forecast_2027:.1f}",
        '95% CI Lower': f"{forecast_2027_ci_lower:.1f}",
        '95% CI Upper': f"{forecast_2027_ci_upper:.1f}",
        'Change (pp)': f"{change:.1f}",
        'Change (%)': f"{pct_change:.1f}%"
    })

summary_df = pd.DataFrame(summary_data)
print("\nFORECAST SUMMARY TABLE:")
print(summary_df.to_string(index=False))
print()

# Save summary
summary_df.to_csv('forecast_summary.csv', index=False)
print("✓ Saved: forecast_summary.csv")
print()
