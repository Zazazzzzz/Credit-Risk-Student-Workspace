# ================================================================
# File:        02_parameter_estimation.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: Part 2 — Model Setup and Parameter Estimation
#
# MODELS CHOSEN:
#   1. Gaussian One-Factor Model (Vasicek 1987)
#      X_i = rho_i*F + sqrt(1-rho_i^2)*epsilon_i
#      Default if X_i <= Phi^{-1}(PD_i)
#      Industry standard under Basel III IRB approach.
#
#   2. Beta-Bernoulli Model (course Task 6)
#      P ~ Beta(alpha, beta), Yi|P ~ Bernoulli(P)
#      Captures parameter uncertainty in PD.
#
# PARAMETERS ESTIMATED:
#   PD  — from base_pd_hint (primary), validated by default history
#   LGD — from entities_info, missing filled with median
#   EAD — from portfolio files
#   rho — correlation of firm stock return with industry index
#
# INPUT:  results/step1_cleaned.pkl
# OUTPUT: results/parameters.csv          → report Table 2
#         results/rho_distribution.png    → report Figure 1
#         results/pd_validation.png       → report Figure 2
#         results/step2_parameters.pkl    → 03_simulation.py
# ================================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results'
)

print("=" * 65)
print("   PART 2: MODEL SETUP AND PARAMETER ESTIMATION")
print("=" * 65)

# ── Load cleaned data from Part 1 ────────────────────────────
data = pickle.load(open(
    os.path.join(DIR_RESULTS, 'step1_cleaned.pkl'), 'rb'))
table1 = data['table1']
table2 = data['table2']
returns = data['returns']
idx_ret = data['idx_ret']
defaults = data['defaults']
entities = data['entities']
print(f"\nLoaded: P1={len(table1)} obligors, P2={len(table2)} obligors")

# ================================================================
# STEP 2.1: REVIEW PD AND LGD
# ================================================================
# PD: from base_pd_hint — analyst estimate of 1-year default prob
# LGD: fraction of EAD lost if firm defaults
# EL  = PD x LGD x EAD already calculated in table1/table2

print("\n" + "-" * 65)
print("STEP 2.1: PD and LGD review")
print("-" * 65)

for name, t in [("Portfolio 1", table1), ("Portfolio 2", table2)]:
    print(f"\n  {name}:")
    print(f"    PD  mean={t['PD'].mean() * 100:.3f}%  "
          f"min={t['PD'].min() * 100:.3f}%  "
          f"max={t['PD'].max() * 100:.3f}%")
    print(f"    LGD mean={t['LGD'].mean() * 100:.1f}%  "
          f"min={t['LGD'].min() * 100:.1f}%  "
          f"max={t['LGD'].max() * 100:.1f}%")
    print(f"    Total EL = EUR {t['EL'].sum() / 1e6:.4f}M")

# ================================================================
# STEP 2.2: ESTIMATE RHO FROM STOCK RETURNS
# ================================================================
# rho = systematic sensitivity in the Gaussian One-Factor Model
#
# METHOD:
#   rho_i = Corr(firm_i daily return, industry_i index return)
#
# INTUITION:
#   High rho means the firm moves closely with its industry.
#   In bad economic times (low F), high-rho firms default more.
#   This creates correlated defaults within the same industry.
#
# CONSTRAINT: rho clipped to [0.1, 0.9]
#   Must be positive (firms positively exposed to economy)
#   Must be below 1 (some firm-specific risk always remains)
#
# FALLBACK: firms with fewer than 30 observations get median rho

print("\n" + "-" * 65)
print("STEP 2.2: Estimating rho")
print("-" * 65)
print("  Method: rho_i = Corr(firm return, industry index return)")

# Merge returns with industry labels
returns_with_ind = returns.merge(
    entities[['entity_code', 'industry']],
    on='entity_code', how='left')

# Merge with industry index returns
returns_full = returns_with_ind.merge(
    idx_ret[['date', 'industry', 'index_return']],
    on=['date', 'industry'], how='left')


def estimate_rho(group):
    """
    Estimate rho for one firm.
    Requires >= 30 observations. Clips result to [0.1, 0.9].
    Returns NaN if insufficient data.
    """
    clean = group.dropna(subset=['stock_return', 'index_return'])
    if len(clean) < 30:
        return np.nan
    corr = clean['stock_return'].corr(clean['index_return'])
    return float(np.clip(corr, 0.1, 0.9))


rho_df = (returns_full
          .groupby('entity_code')
          .apply(estimate_rho)
          .reset_index()
          .rename(columns={0: 'rho'}))

rho_median = rho_df['rho'].median()
n_missing = rho_df['rho'].isna().sum()
rho_df['rho'] = rho_df['rho'].fillna(rho_median)

print(f"\n  Rho estimated for {rho_df['rho'].notna().sum()} firms")
print(f"  Firms using median fallback: {n_missing}")
print(f"  Median rho: {rho_median:.3f}")
print(f"  Min rho:    {rho_df['rho'].min():.3f}")
print(f"  Max rho:    {rho_df['rho'].max():.3f}")

# Add rho to portfolio tables
table1 = table1.merge(rho_df, on='entity_code', how='left')
table2 = table2.merge(rho_df, on='entity_code', how='left')
table1['rho'] = table1['rho'].fillna(rho_median)
table2['rho'] = table2['rho'].fillna(rho_median)

print(f"\n  P1 mean rho: {table1['rho'].mean():.3f}")
print(f"  P2 mean rho: {table2['rho'].mean():.3f}")

# ================================================================
# STEP 2.3: VALIDATE PD USING DEFAULT HISTORY
# ================================================================
# Two PD sources available:
#   base_pd_hint: analyst estimate (PRIMARY — better coverage)
#   empirical PD: n_defaults / n_years from 20-year history
#
# PROBLEM with empirical PD:
#   Many firms have 0 historical defaults over 20 years.
#   This gives empirical PD = 0% which underestimates true risk.
#
# VALIDATION:
#   Check correlation between both sources.
#   Positive correlation confirms base_pd_hint is consistent
#   with historical defaults.

print("\n" + "-" * 65)
print("STEP 2.3: Validating PD from default history")
print("-" * 65)

emp_pd = (defaults.groupby('entity_code')['default_event']
          .agg(n_defaults='sum', n_years='count').reset_index())
emp_pd['empirical_pd'] = emp_pd['n_defaults'] / emp_pd['n_years']

print(f"  Firms with history:   {len(emp_pd)}")
print(f"  Total defaults:       {emp_pd['n_defaults'].sum():.0f}")
print(f"  Firms with 0 defaults:{(emp_pd['n_defaults'] == 0).sum()}")
print(f"  Mean empirical PD:    {emp_pd['empirical_pd'].mean() * 100:.3f}%")

comparison = entities[['entity_code', 'base_pd_hint']].merge(
    emp_pd, on='entity_code', how='inner')
corr = comparison['base_pd_hint'].corr(comparison['empirical_pd'])
print(f"\n  Corr(base_pd_hint, empirical_pd) = {corr:.3f}")
print(f"  Decision: use base_pd_hint as PRIMARY PD estimate")
print(f"  Reason: better coverage, avoids zero-default bias")

# ================================================================
# STEP 2.4: BETA-BERNOULLI PARAMETERS
# ================================================================
# For the Beta-Bernoulli model we need alpha and beta.
# Calibrated using method of moments:
#   E[P]   = p_mean  (mean portfolio PD)
#   Var[P] = rho_bb * p_mean * (1-p_mean)
#   rho_bb = 0.005 (0.5% default correlation, standard assumption)

print("\n" + "-" * 65)
print("STEP 2.4: Beta-Bernoulli parameters")
print("-" * 65)


def compute_beta_params(p_mean, default_corr=0.005):
    """
    Compute Beta(alpha, beta) parameters using method of moments.
    E[P] = p_mean, Var[P] = default_corr * p_mean * (1-p_mean)
    """
    var_p = default_corr * p_mean * (1 - p_mean)
    common = p_mean * (1 - p_mean) / var_p - 1
    alpha = p_mean * common
    beta = (1 - p_mean) * common
    return alpha, beta


for name, table in [("Portfolio 1", table1), ("Portfolio 2", table2)]:
    pm = table['PD'].mean()
    a, b = compute_beta_params(pm)
    print(f"\n  {name}: mean PD={pm * 100:.3f}%  "
          f"alpha={a:.4f}  beta={b:.4f}  "
          f"E[P]={a / (a + b) * 100:.3f}%")

# ================================================================
# STEP 2.5: PARAMETER SUMMARY AND SAVE
# ================================================================
print("\n" + "=" * 65)
print("PARAMETER SUMMARY")
print("=" * 65)
print(f"\n  {'Parameter':<22} {'Portfolio 1':>15} {'Portfolio 2':>15}")
print(f"  {'-' * 54}")
print(f"  {'Obligors':<22} {len(table1):>15} {len(table2):>15}")
print(f"  {'Total EAD (EUR M)':<22} "
      f"{table1['EAD'].sum() / 1e6:>15.2f} "
      f"{table2['EAD'].sum() / 1e6:>15.2f}")
print(f"  {'Mean PD':<22} "
      f"{table1['PD'].mean() * 100:>14.3f}% "
      f"{table2['PD'].mean() * 100:>14.3f}%")
print(f"  {'Mean LGD':<22} "
      f"{table1['LGD'].mean() * 100:>14.1f}% "
      f"{table2['LGD'].mean() * 100:>14.1f}%")
print(f"  {'Mean rho':<22} "
      f"{table1['rho'].mean():>15.3f} "
      f"{table2['rho'].mean():>15.3f}")
print(f"  {'Total EL (EUR M)':<22} "
      f"{table1['EL'].sum() / 1e6:>15.4f} "
      f"{table2['EL'].sum() / 1e6:>15.4f}")

# Save CSV → report Table 2
params = pd.concat([
    table1[['entity_code', 'entity_name', 'industry',
            'region', 'PD', 'LGD', 'EAD', 'rho', 'EL'
            ]].assign(portfolio='Portfolio 1'),
    table2[['entity_code', 'entity_name', 'industry',
            'region', 'PD', 'LGD', 'EAD', 'rho', 'EL'
            ]].assign(portfolio='Portfolio 2'),
])
params.to_csv(os.path.join(DIR_RESULTS, 'parameters.csv'), index=False)
print(f"\n  Saved: parameters.csv → report Table 2")

# Plot rho distribution → report Figure 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Estimated Systematic Sensitivity (rho)',
             fontsize=12, fontweight='bold')
for ax, table, name, color in [
    (ax1, table1, 'Portfolio 1', 'steelblue'),
    (ax2, table2, 'Portfolio 2', 'coral'),
]:
    ax.hist(table['rho'], bins=12, color=color,
            edgecolor='black', alpha=0.8)
    ax.axvline(table['rho'].mean(), color='red', linestyle='--',
               lw=2, label=f"Mean={table['rho'].mean():.3f}")
    ax.set_title(name)
    ax.set_xlabel('rho')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTS, 'rho_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: rho_distribution.png → report Figure 1")

# Plot PD validation scatter → report Figure 2
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(comparison['empirical_pd'] * 100,
           comparison['base_pd_hint'] * 100,
           alpha=0.6, color='steelblue', edgecolor='black')
ax.set_xlabel('Empirical PD (%)')
ax.set_ylabel('base_pd_hint (%)')
ax.set_title(f'PD Validation (correlation={corr:.3f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTS, 'pd_validation.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: pd_validation.png → report Figure 2")

# Save for next step
pickle.dump({
    'table1': table1,
    'table2': table2,
    'rho_median': rho_median,
}, open(os.path.join(DIR_RESULTS, 'step2_parameters.pkl'), 'wb'))
print(f"  Saved: step2_parameters.pkl → used by 03_simulation.py")

print("\n" + "=" * 65)
print("PART 2 COMPLETE — Next: run 03_simulation.py")
print("=" * 65)