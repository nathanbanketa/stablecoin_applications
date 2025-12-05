"""
AGENT-BASED SIMULATION FOR STABLECOIN ADOPTION
Computational Model of Adoption Dynamics
For: CSCI4911 Stablecoin Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AGENT-BASED SIMULATION: STABLECOIN ADOPTION")
print("Computational Model of Adoption Dynamics")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD DATA AND SET UP PARAMETERS
# ============================================================================

print("STEP 1: Loading data and setting up simulation...")
print("-"*80)

df = pd.read_csv("merged_dataset_for_regression.csv")
df_clean = df.dropna(subset=['Stablecoin_Share_Pct'])

print(f"✓ Loaded {len(df_clean)} observations")
print()

# Handle missing data by filling with previous year values
print("Handling missing data:")
for country in df_clean['Country'].unique():
    country_mask = df_clean['Country'] == country
    
    # Fill NaN in internet_users_pct with previous year or default
    if df_clean.loc[country_mask, 'internet_users_pct'].isna().any():
        prev_vals = df_clean.loc[country_mask, 'internet_users_pct'].dropna()
        if len(prev_vals) > 0:
            fill_val = prev_vals.iloc[-1]  # Use most recent available
        else:
            fill_val = 35.0  # Default for emerging markets
        df_clean.loc[country_mask & df_clean['internet_users_pct'].isna(), 'internet_users_pct'] = fill_val
        print(f"  {country} internet: filled missing with {fill_val:.1f}%")
    
    # Fill NaN in mobile_subscriptions_per100 with previous year or default
    if df_clean.loc[country_mask, 'mobile_subscriptions_per100'].isna().any():
        prev_vals = df_clean.loc[country_mask, 'mobile_subscriptions_per100'].dropna()
        if len(prev_vals) > 0:
            fill_val = prev_vals.iloc[-1]  # Use most recent available
        else:
            fill_val = 90.0  # Default for emerging markets (high mobile penetration)
        df_clean.loc[country_mask & df_clean['mobile_subscriptions_per100'].isna(), 'mobile_subscriptions_per100'] = fill_val
        print(f"  {country} mobile: filled missing with {fill_val:.1f}")

print("✓ Missing data handled")
print()

# ============================================================================
# STEP 2: DEFINE AGENT MODEL
# ============================================================================

print("STEP 2: Defining agent model...")
print("-"*80)
print()

print("Agent Decision Model:")
print("  An agent adopts stablecoins if:")
print("    1. Cost savings > threshold (7.7 percentage points)")
print("    2. Infrastructure sufficient (internet_users_pct > 30%)")
print("    3. Random adoption probability increases with:")
print("       - Higher inflation (currency pressure)")
print("       - Better internet (digital access)")
print("       - More mobile subscriptions (fintech adoption)")
print()

@dataclass
class Agent:
    """Represents a user/household that can adopt stablecoins"""
    agent_id: int
    country: str
    has_adopted: bool = False
    inflation_pressure: float = 0.0
    infrastructure_score: float = 0.0
    adoption_probability: float = 0.0
    
    def calculate_adoption_probability(self, inflation, internet, mobile):
        """
        Calculate likelihood of adopting stablecoins based on conditions
        
        Higher inflation → more likely to adopt (currency hedge)
        Better internet → more likely to adopt (digital access)
        More mobile subs → more likely to adopt (fintech ready)
        """
        # Normalize inputs to 0-1 scale
        inflation_normalized = min(inflation / 50, 1.0)  # Cap at 50%
        internet_normalized = internet / 100
        mobile_normalized = min(mobile / 100, 1.0)
        
        # Weighted combination (you can adjust weights)
        self.adoption_probability = (
            0.4 * inflation_normalized +
            0.3 * internet_normalized +
            0.3 * mobile_normalized
        )
        
        return self.adoption_probability
    
    def make_decision(self):
        """Agent decides whether to adopt based on probability"""
        self.has_adopted = np.random.random() < self.adoption_probability
        return self.has_adopted


# ============================================================================
# STEP 3: CREATE SIMULATION FUNCTION
# ============================================================================

print("STEP 3: Creating simulation function...")
print("-"*80)
print()

def run_simulation(country_data, num_agents=1000, num_iterations=10, 
                  inflation_scenario='baseline'):
    """
    Run agent-based simulation for a country
    
    Parameters:
        country_data: DataFrame row with inflation, internet, mobile data
        num_agents: Number of agents to simulate
        num_iterations: Number of simulation steps
        inflation_scenario: 'baseline', 'low', 'high', 'extreme'
    """
    
    # Get base parameters
    base_inflation = country_data['inflation_rate']
    internet = country_data['internet_users_pct']
    mobile = country_data['mobile_subscriptions_per100']
    
    # Apply scenario adjustment
    scenario_multipliers = {
        'low': 0.5,
        'baseline': 1.0,
        'high': 1.5,
        'extreme': 2.0
    }
    inflation = base_inflation * scenario_multipliers[inflation_scenario]
    
    # Initialize agents
    agents = [Agent(agent_id=i, country=country_data['Country']) 
              for i in range(num_agents)]
    
    # Calculate adoption probabilities for all agents
    for agent in agents:
        agent.calculate_adoption_probability(inflation, internet, mobile)
    
    # Run iterations (simulating network effects and adoption waves)
    adoption_history = []
    adoption_rates = []
    
    for iteration in range(num_iterations):
        # Agents make decisions
        for agent in agents:
            agent.make_decision()
        
        # Calculate adoption rate
        adopted = sum(1 for agent in agents if agent.has_adopted)
        adoption_rate = (adopted / num_agents) * 100
        adoption_rates.append(adoption_rate)
        adoption_history.append(adopted)
    
    return {
        'adoption_rates': adoption_rates,
        'adoption_history': adoption_history,
        'final_adoption_rate': adoption_rates[-1],
        'inflation_used': inflation,
        'internet': internet,
        'mobile': mobile
    }


# ============================================================================
# STEP 4: RUN SIMULATIONS FOR EACH COUNTRY
# ============================================================================

print("STEP 4: Running simulations...")
print("-"*80)
print()

simulation_results = {}

for country in df_clean['Country'].unique():
    country_data = df_clean[df_clean['Country'] == country].iloc[-1]  # Latest year
    
    print(f"{country}:")
    print(f"  Inflation: {country_data['inflation_rate']:.1f}%")
    print(f"  Internet Users: {country_data['internet_users_pct']:.1f}%")
    print(f"  Mobile Subscriptions: {country_data['mobile_subscriptions_per100']:.1f}")
    print(f"  Actual Adoption: {country_data['Stablecoin_Share_Pct']:.1f}%")
    print()
    
    # Run baseline scenario
    baseline = run_simulation(country_data, num_agents=1000, num_iterations=10)
    
    # Run alternative scenarios
    low_inflation = run_simulation(country_data, num_agents=1000, num_iterations=10,
                                   inflation_scenario='low')
    high_inflation = run_simulation(country_data, num_agents=1000, num_iterations=10,
                                    inflation_scenario='high')
    
    simulation_results[country] = {
        'baseline': baseline,
        'low_inflation': low_inflation,
        'high_inflation': high_inflation,
        'actual': country_data['Stablecoin_Share_Pct']
    }
    
    print(f"  Baseline simulation: {baseline['final_adoption_rate']:.1f}%")
    print(f"  Low inflation scenario: {low_inflation['final_adoption_rate']:.1f}%")
    print(f"  High inflation scenario: {high_inflation['final_adoption_rate']:.1f}%")
    print()

# ============================================================================
# STEP 5: COMPARE SIMULATIONS TO ACTUAL DATA
# ============================================================================

print("="*80)
print("STEP 5: Comparing simulation results to actual adoption...")
print("="*80)
print()

comparison_data = []

for country, results in simulation_results.items():
    comparison_data.append({
        'Country': country,
        'Actual (%)': results['actual'],
        'Baseline Sim (%)': results['baseline']['final_adoption_rate'],
        'Low Inflation Sim (%)': results['low_inflation']['final_adoption_rate'],
        'High Inflation Sim (%)': results['high_inflation']['final_adoption_rate'],
        'Baseline Error': abs(results['actual'] - results['baseline']['final_adoption_rate'])
    })

comparison_df = pd.DataFrame(comparison_data)
print("Simulation Results vs Actual Adoption:")
print(comparison_df.to_string(index=False))
print()

comparison_df.to_csv('simulation_comparison.csv', index=False)
print("✓ Saved: simulation_comparison.csv")
print()

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("STEP 6: Creating visualizations...")
print("-"*80)

# Plot 1: Actual vs Simulated (all countries)
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(simulation_results))
width = 0.2

actual_vals = [simulation_results[c]['actual'] for c in sorted(simulation_results.keys())]
baseline_vals = [simulation_results[c]['baseline']['final_adoption_rate'] 
                 for c in sorted(simulation_results.keys())]

ax.bar(x - width/2, actual_vals, width, label='Actual', color='steelblue', alpha=0.8)
ax.bar(x + width/2, baseline_vals, width, label='Baseline Simulation', color='coral', alpha=0.8)

ax.set_xlabel('Country', fontsize=12)
ax.set_ylabel('Adoption Rate (%)', fontsize=12)
ax.set_title('Agent-Based Simulation: Actual vs Predicted Adoption', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sorted(simulation_results.keys()))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simulation_actual_vs_predicted.png', dpi=300)
print("✓ Saved: simulation_actual_vs_predicted.png")

# Plot 2: Scenario comparison (using Nigeria as example)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

countries_to_plot = list(sorted(simulation_results.keys()))[:4]

for idx, country in enumerate(countries_to_plot):
    ax = axes[idx]
    results = simulation_results[country]
    
    iterations = range(len(results['baseline']['adoption_rates']))
    
    ax.plot(iterations, results['baseline']['adoption_rates'], 'o-', 
           linewidth=2, markersize=6, label='Baseline', color='steelblue')
    ax.plot(iterations, results['low_inflation']['adoption_rates'], 's--',
           linewidth=2, markersize=6, label='Low Inflation', color='green', alpha=0.7)
    ax.plot(iterations, results['high_inflation']['adoption_rates'], '^--',
           linewidth=2, markersize=6, label='High Inflation', color='red', alpha=0.7)
    
    ax.axhline(y=results['actual'], color='black', linestyle=':', linewidth=2,
              label=f'Actual ({results["actual"]:.1f}%)')
    
    ax.set_xlabel('Simulation Iteration', fontsize=10)
    ax.set_ylabel('Adoption Rate (%)', fontsize=10)
    ax.set_title(f'{country} - Adoption Scenarios', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('simulation_scenarios.png', dpi=300)
print("✓ Saved: simulation_scenarios.png")

print()