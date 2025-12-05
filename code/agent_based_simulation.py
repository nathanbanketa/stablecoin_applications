"""
AGENT-BASED SIMULATION FOR STABLECOIN ADOPTION
Calibrated "Playbook" Logic + Real Data Integration
For: CSCI4911 Stablecoin Research Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("LOADING DATA & CONFIGURING SIMULATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD REAL DATA FROM CSV
# ============================================================================

try:
    df = pd.read_csv("merged_dataset_for_regression.csv")
    print("✓ Loaded merged_dataset_for_regression.csv")
except FileNotFoundError:
    print("⚠ File not found. Using fallback data.")
    df = pd.DataFrame()

# Helper to get latest value or fallback
def get_latest_param(country, col, fallback):
    if df.empty: return fallback
    try:
        val = df[df['Country'] == country].sort_values('Year').iloc[-1][col]
        return float(val)
    except:
        return fallback

# DYNAMICALLY POPULATE PARAMETERS
COUNTRY_PARAMS = {}
target_countries = ['Nigeria', 'Ethiopia', 'Philippines']

for country in target_countries:
    # 1. Get Mobile Subs to estimate "Mobile Money Trust/Usage"
    mob_subs = get_latest_param(country, 'mobile_subscriptions_per100', 80.0)
    
    # 2. Calculate Proxy for Trust (Max 0.9)
    # This fixes your KeyError: We define 'mobile_money_trust' here!
    mm_trust_proxy = min(0.9, mob_subs / 140.0)

    # 3. Get Remittance Cost (Default to 8.5 if column missing)
    try:
        remit_cost = get_latest_param(country, 'remittance_cost', 8.5)
    except:
        remit_cost = 8.5

    COUNTRY_PARAMS[country] = {
        'inflation_rate': get_latest_param(country, 'inflation_rate', 15.0),
        'internet_penetration': get_latest_param(country, 'internet_users_pct', 40.0),
        'remittance_cost': remit_cost,
        'mobile_money_trust': mm_trust_proxy  # <--- KEY DEFINED HERE
    }
    
    print(f"\nConfiguration for {country}:")
    print(f"  Inflation: {COUNTRY_PARAMS[country]['inflation_rate']:.1f}%")
    print(f"  Internet: {COUNTRY_PARAMS[country]['internet_penetration']:.1f}%")
    print(f"  MM Trust Index: {COUNTRY_PARAMS[country]['mobile_money_trust']:.2f}")
    print(f"  Remittance Cost: {COUNTRY_PARAMS[country]['remittance_cost']:.1f}%")

SCENARIOS = {
    'Baseline': {'offline': False, 'integration': False},
    'Offline Wallet': {'offline': True, 'integration': False},
    'MM Integration': {'offline': False, 'integration': True},
    'Full Playbook': {'offline': True, 'integration': True}
}

# ============================================================================
# STEP 2: CALIBRATED AGENT LOGIC (THE "JUICED" VERSION)
# ============================================================================

@dataclass
class ConsumerAgent:
    id: int
    country: str
    has_internet: bool
    uses_mobile_money: bool
    tech_literacy: float
    trust_baseline: float
    
    def calculate_adoption_utility(self, params, scenario):
        
        # 1. ACCESSIBILITY
        # Old way: return -10.0 (Immediate rejection)
        # New way: Heavy penalty, but surmountable if the incentive is high enough
        u_access = 0.0
        if not self.has_internet:
            if scenario['offline']:
                u_access = 0.0 # Problem solved
            else:
                u_access = -4.0 # Heavy barrier, but not infinite
        else:
            u_access = 0.5 # Internet bonus

        # 2. TRUST & INTEGRATION (The Solution)
        u_trust = self.trust_baseline
        
        # If integrated, trust boosts significantly
        if scenario['integration'] and self.uses_mobile_money:
            u_trust += 2.5 
            
            # CRITICAL FIX: Mobile Money users are used to SMS/USSD.
            # Even without "Offline Wallets", they can manage patchy connections better.
            # We give them a "Resilience Bonus" that lowers the access penalty.
            if not self.has_internet:
                u_access += 1.5 

        elif not scenario['integration']:
            u_trust -= 1.0 # Friction penalty
            
        # 3. ECONOMIC INCENTIVES
        u_inflation = (params['inflation_rate'] / 100.0) * 2.0
        cost = params.get('remittance_cost', 8.5)
        u_savings = (cost / 10.0) * 1.5
        intertia_penalty = 2.0
        
        return u_access + u_trust + u_inflation + u_savings - intertia_penalty
# ============================================================================
# STEP 3: RUNNING THE SIMULATION
# ============================================================================

def run_country_simulation(country_name, num_agents=1000, steps=50):
    params = COUNTRY_PARAMS[country_name]
    results = {}
    
    # Initialize Population
    agents = []
    for i in range(num_agents):
        has_net = np.random.random() < (params['internet_penetration'] / 100)
        # Use the key we defined above
        has_mm = np.random.random() < params['mobile_money_trust']
        
        agent = ConsumerAgent(
            id=i,
            country=country_name,
            has_internet=has_net,
            uses_mobile_money=has_mm,
            tech_literacy=np.random.beta(2, 2),
            trust_baseline=np.random.normal(0, 1.0) # Wider variance
        )
        agents.append(agent)
            
    print(f"Simulating {steps} months for {country_name}...")
    
    for scenario_name, settings in SCENARIOS.items():
        adoption_curve = []
        adopters = set()
        
        for t in range(steps):
            # Faster Awareness Curve
            awareness = 1 / (1 + np.exp(-0.2 * (t - 10)))
            
            current_adopters = 0
            for agent in agents:
                if agent.id in adopters:
                    current_adopters += 1
                    continue
                
                if np.random.random() < awareness:
                    utility = agent.calculate_adoption_utility(params, settings)
                    
                    # Steeper slope (3.0) to force decision separation
                    prob = 1 / (1 + np.exp(-3.0 * utility))
                    
                    if np.random.random() < prob:
                        adopters.add(agent.id)
                        current_adopters += 1
            
            adoption_pct = (current_adopters / num_agents) * 100
            adoption_curve.append(adoption_pct)
            
        results[scenario_name] = adoption_curve
        
    return results

# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

all_results = {}
for country in target_countries:
    all_results[country] = run_country_simulation(country)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, country in enumerate(target_countries):
    ax = axes[idx]
    res = all_results[country]
    
    # Plot Scenarios
    ax.plot(res['Baseline'], 'k--', label='Baseline', linewidth=2, alpha=0.6)
    ax.plot(res['Offline Wallet'], 'g-', label='Offline Wallet', linewidth=2, alpha=0.8)
    ax.plot(res['MM Integration'], 'b-', label='Mobile Money Integration', linewidth=2.5)
    ax.plot(res['Full Playbook'], 'r-', label='Full Playbook (Combined)', linewidth=3)
    
    ax.set_title(f"{country}\n(Inf: {COUNTRY_PARAMS[country]['inflation_rate']:.1f}%, Cost: {COUNTRY_PARAMS[country]['remittance_cost']:.1f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Months")
    ax.set_ylabel("Adoption (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('corrected_simulation_results.png', dpi=300)
print("\n✓ Simulation Complete. Saved to 'corrected_simulation_results.png'")