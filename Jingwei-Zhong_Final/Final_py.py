import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. GLOBAL PATH & PARAMETER SETUP
DIR_DATA = "/Users/watson/Desktop/Final"

print("STEP 1: LOAD AND CLEAN PORTFOLIO HOLDINGS")

# Load Portfolio 1
df1 = pd.read_csv(os.path.join(DIR_DATA, "portfolio_1.csv"))
df1.columns = df1.columns.str.strip().str.lower()
df1['portfolio_id'] = 'Portfolio_1'

# Load Portfolio 2
df2 = pd.read_csv(os.path.join(DIR_DATA, "portfolio_2.csv"))
df2.columns = df2.columns.str.strip().str.lower()
df2['portfolio_id'] = 'Portfolio_2'

# Combine Portfolio 1 and Portfolio 2
df_portfolios = pd.concat([df1, df2], ignore_index=True)

# Convert entity codes
df_portfolios['entity_code'] = df_portfolios['entity_code'].astype(str).str.strip().str.upper()

print("STEP 2: LOAD AND CLEAN FIRM CHARACTERISTICS")

# Load entities information file
df_entities = pd.read_csv(os.path.join(DIR_DATA, "entities_info.csv"))
df_entities.columns = df_entities.columns.str.strip().str.lower()
df_entities['entity_code'] = df_entities['entity_code'].astype(str).str.strip().str.upper()

# Standardize messy industry labels to uniform names
df_entities['industry'] = df_entities['industry'].str.strip().str.upper()
df_entities['industry'] = df_entities['industry'].replace({
    'IND. A': 'INDUSTRY_A', 'IND. B': 'INDUSTRY_B',
    'INDUSTRYA': 'INDUSTRY_A', 'INDUSTRYB': 'INDUSTRY_B', 'INDUSTRY': 'INDUSTRY_A'
})
df_entities['industry'] = df_entities['industry'].fillna('INDUSTRY_A')

# Fill missing LGD
df_entities['lgd'] = df_entities['lgd'].fillna(df_entities['lgd'].median())

print("STEP 3: PROCESS STOCK RETURNS AND ANNUALIZE METRICS")

# Load historical daily stock returns data
df_stock = pd.read_csv(os.path.join(DIR_DATA, "stock_returns.csv"))
df_stock.columns = df_stock.columns.str.strip().str.lower()
df_stock['entity_code'] = df_stock['entity_code'].astype(str).str.strip().str.upper()

# Drop duplicate entries for the same company on the same date
df_stock = df_stock.drop_duplicates(subset=['date', 'entity_code'])

# compute daily average return and daily volatility
df_stock_stats = df_stock.groupby('entity_code')['stock_return'].agg(['mean', 'std']).reset_index()
df_stock_stats.columns = ['entity_code', 'mean_return', 'volatility']

# assuming 252 trading days per year
df_stock_stats['mean_return'] = df_stock_stats['mean_return'] * 252
df_stock_stats['volatility'] = df_stock_stats['volatility'] * np.sqrt(252)

print("STEP 4: CALCULATE BASELINE SECTOR DEFAULT RATES")

# Load 20-year historical default frequency data
df_default = pd.read_csv(os.path.join(DIR_DATA, "default_history_20y.csv"))
df_default.columns = df_default.columns.str.strip().str.upper()
if 'YEAR' in df_default.columns:
    df_default = df_default.set_index('YEAR')

# Compute long-term average default rate
df_industry_pd = df_default.mean(numeric_only=True).reset_index()
df_industry_pd.columns = ['industry', 'baseline_pd']
df_industry_pd['industry'] = df_industry_pd['industry'].replace({
    'IND. A': 'INDUSTRY_A', 'IND. B': 'INDUSTRY_B', 'INDUSTRY': 'INDUSTRY_A'
})

# BUG Fix
for idx in df_industry_pd.index:
    if df_industry_pd.loc[idx, 'baseline_pd'] > 1.0:
        df_industry_pd.loc[idx, 'baseline_pd'] = df_industry_pd.loc[idx, 'baseline_pd'] / 100.0

print("STEP 5: MERGE ALL CLEAN COMPONENTS INTO MASTER TABLE")

# Progressively join all tables together
master_df = pd.merge(df_portfolios, df_entities, on='entity_code', how='left')
master_df = pd.merge(master_df, df_stock_stats, on='entity_code', how='left')
master_df = pd.merge(master_df, df_industry_pd, on='industry', how='left')

# Global fallback imputation
master_df['ead'] = master_df['ead'].fillna(master_df['ead'].median())
master_df['lgd'] = master_df['lgd'].fillna(0.45)
master_df['baseline_pd'] = master_df['baseline_pd'].fillna(0.0295)

# Replace missing strings in non-modeling note columns
text_cols = ['entity_name', 'region', 'internal_comment', 'size_bucket', 'analyst_note', 'region_note']
for col in text_cols:
    if col in master_df.columns:
        master_df[col] = master_df[col].fillna('N/A')

# Create results folder and save the dataset
os.makedirs(os.path.join(DIR_DATA, "results"), exist_ok=True)
master_df.to_csv(os.path.join(DIR_DATA, "results", "cleaned_modeling_master_table.csv"), index=False)
print("SUCCESS!")

print("\nSTEP 6: RUN 10,000 MONTE CARLO RISK SIMULATIONS")

# Model assumptions based on standard academic parameters
rho_industry = 0.25  # Correlation between industry factor A and industry factor B
beta = 0.40  # Firm asset sensitivity to systematic sector shocks
np.random.seed(42)# Set random seed to make sure simulation results are exactly reproducible
final_results = {}
loss_storage = {}

# Simulate losses separately for Portfolio 1 and Portfolio 2
for p_id in ['Portfolio_1', 'Portfolio_2']:
    # Filter the positions belonging
    p_df = master_df[master_df['portfolio_id'] == p_id]
    n_assets = len(p_df)

    # Extract columns into simple arrays for fast loop access
    ead_list = p_df['ead'].values
    lgd_list = p_df['lgd'].values
    pd_list = p_df['baseline_pd'].values
    ind_list = p_df['industry'].values

    # Map probability of default to standard normal critical threshold
    z_crit = norm.ppf(pd_list)

    losses_m1 = []
    losses_m2 = []

    # Main loop for 10,000 independent stochastic simulation paths
    for s in range(10000):

        # MODEL 1: SINGLE-FACTOR MERTON FRAMEWORK
        Y_global = np.random.normal(0, 1)  # Draw global economy factor
        epsilon = np.random.normal(0, 1, n_assets)  # Draw idiosyncratic shock for each firm

        loss_s1 = 0
        for i in range(n_assets):
            # Calculate latent asset return R using Merton's single-factor formula
            R = beta * Y_global + np.sqrt(1 - beta ** 2) * epsilon[i]
            # If asset return falls below default boundary, trigger credit default and add loss
            if R < z_crit[i]:
                loss_s1 += ead_list[i] * lgd_list[i]
        losses_m1.append(loss_s1)

        # MODEL 2: MULTI-FACTOR INDUSTRY MIXTURE FRAMEWORK
        # Generate two correlated systematic risk factors for Industry A and Industry B
        cov_matrix = [[1.0, rho_industry], [rho_industry, 1.0]]
        Y_A, Y_B = np.random.multivariate_normal([0, 0], cov_matrix)

        loss_s2 = 0
        # Dynamically assign systematic factor based on firm specific sector label
        for i in range(n_assets):
            if ind_list[i] == 'INDUSTRY_B':
                Y_system = Y_B
            else:
                Y_system = Y_A

            R = beta * Y_system + np.sqrt(1 - beta ** 2) * epsilon[i]
            if R < z_crit[i]:
                loss_s2 += ead_list[i] * lgd_list[i]
        losses_m2.append(loss_s2)

    # Calculate portfolio credit risk metrics
    el_m1 = np.mean(losses_m1)
    var95_m1 = np.percentile(losses_m1, 95)
    var99_m1 = np.percentile(losses_m1, 99)
    # Expected Shortfall: average of all losses that >= 99% VaR boundary
    tail_losses_m1 = [l for l in losses_m1 if l >= var99_m1]
    es99_m1 = np.mean(tail_losses_m1) if len(tail_losses_m1) > 0 else var99_m1

    el_m2 = np.mean(losses_m2)
    var95_m2 = np.percentile(losses_m2, 95)
    var99_m2 = np.percentile(losses_m2, 99)
    tail_losses_m2 = [l for l in losses_m2 if l >= var99_m2]
    es99_m2 = np.mean(tail_losses_m2) if len(tail_losses_m2) > 0 else var99_m2

    # Store results matrix inside final output dictionary
    final_results[f"{p_id} (Model 1: Single-Factor)"] = [el_m1, var95_m1, var99_m1, es99_m1]
    final_results[f"{p_id} (Model 2: Multi-Factor)"] = [el_m2, var95_m2, var99_m2, es99_m2]

    loss_storage[f"{p_id}_M1"] = losses_m1
    loss_storage[f"{p_id}_M2"] = losses_m2

# Export final metrics table to a summary CSV file
df_report = pd.DataFrame(final_results,
                         index=['Expected Loss (EL)', 'VaR (95%)', 'VaR (99%)', 'Expected Shortfall (ES 99%)']).T
df_report.to_csv(os.path.join(DIR_DATA, "results", "model_risk_metrics_comparison.csv"))

print("\n Table 1: Portfolio Aggregate Characteristics:")
summary = master_df.groupby('portfolio_id').agg({
    'entity_code': 'count',
    'ead': 'sum',
    'lgd': 'mean',
    'baseline_pd': 'mean'
})
print(summary)

print("\n Table 2: Non-Zero Credit Risk Metrics Cross-Comparison:")
df_report_formatted = df_report.copy()
for col in df_report_formatted.columns:
    df_report_formatted[col] = df_report_formatted[col].apply(lambda x: f"${x:,.2f}")
print(df_report_formatted)

print("\n STEP 7: AUTOMATED DENSITY DISTRIBUTION PLOTTING")
plt.figure(figsize=(14, 6))

# Plot Empirical Loss Profile for Portfolio 1
plt.subplot(1, 2, 1)
plt.hist(loss_storage['Portfolio_1_M1'], bins=80, density=True, alpha=0.5, color='blue', label='Model 1: Single-Factor')
plt.hist(loss_storage['Portfolio_1_M2'], bins=80, density=True, alpha=0.4, color='red', label='Model 2: Multi-Factor')
plt.axvline(df_report.loc['Portfolio_1 (Model 1: Single-Factor)', 'VaR (99%)'], color='blue', linestyle='--')
plt.axvline(df_report.loc['Portfolio_1 (Model 2: Multi-Factor)', 'VaR (99%)'], color='red', linestyle=':')
plt.title('Portfolio 1 Loss Distribution Profile')
plt.xlabel('Loss Amount ($)')
plt.ylabel('Probability Density Frequency')
plt.legend()

# Plot Empirical Loss Profile for Portfolio 2
plt.subplot(1, 2, 2)
plt.hist(loss_storage['Portfolio_2_M1'], bins=80, density=True, alpha=0.5, color='blue', label='Model 1: Single-Factor')
plt.hist(loss_storage['Portfolio_2_M2'], bins=80, density=True, alpha=0.4, color='red', label='Model 2: Multi-Factor')
plt.axvline(df_report.loc['Portfolio_2 (Model 1: Single-Factor)', 'VaR (99%)'], color='blue', linestyle='--')
plt.axvline(df_report.loc['Portfolio_2 (Model 2: Multi-Factor)', 'VaR (99%)'], color='red', linestyle=':')
plt.title('Portfolio 2 Loss Distribution Profile')
plt.xlabel('Loss Amount ($)')
plt.ylabel('Probability Density Frequency')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(DIR_DATA, "results", "portfolio_loss_distributions.png"), dpi=200)
print("SUCCESS!")