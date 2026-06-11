# ================================================================
# File:        03_simulation.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: Part 3 — Simulation and Risk Measurement
#
# MODELS:
#   1. Gaussian One-Factor Model (Vasicek 1987)
#      X_i = rho_i*F + sqrt(1-rho_i^2)*epsilon_i
#      Default if X_i <= Phi^{-1}(PD_i)
#
#   2. Beta-Bernoulli Model
#      P ~ Beta(alpha, beta)
#      Yi | P ~ Bernoulli(P)
#
# RISK MEASURES:
#   EL = Expected Loss
#   Std = Standard deviation of losses
#   VaR 99% = Value at Risk at 99% confidence
#   ES 99%  = Expected Shortfall (average of worst 1%)
#
# INPUT:  results/step2_parameters.pkl
# OUTPUT: results/loss_distributions.png  → report Figure 3
#         results/risk_measures_table.csv → report Table 3
#         results/step3_results.pkl       → 04_analysis_plots.py
# ================================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

DIR_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results'
)

print("=" * 65)
print("   PART 3: SIMULATION AND RISK MEASUREMENT")
print("=" * 65)

# ── Load parameters ───────────────────────────────────────────
data = pickle.load(open(
    os.path.join(DIR_RESULTS, 'step2_parameters.pkl'), 'rb'))
table1 = data['table1']
table2 = data['table2']
print("\nDEBUG CHECK:")
print(f"  table1 shape: {table1.shape}")
print(f"  table1 columns: {table1.columns.tolist()}")
print(f"  table1 PD sample: {table1['PD'].head(3).tolist()}")
print(f"  table1 LGD sample: {table1['LGD'].head(3).tolist()}")
print(f"  table1 EAD sample: {table1['EAD'].head(3).tolist()}")
print(f"  table1 rho sample: {table1['rho'].head(3).tolist()}")
print(f"  Any NaN in PD?  {table1['PD'].isna().sum()}")
print(f"  Any NaN in LGD? {table1['LGD'].isna().sum()}")
print(f"  Any NaN in EAD? {table1['EAD'].isna().sum()}")
print(f"  Any NaN in rho? {table1['rho'].isna().sum()}")
# Define function directly here instead of loading from pickle
def compute_beta_params(p_mean, default_corr=0.005):
    """
    Compute Beta(alpha, beta) parameters using method of moments.
    E[P] = p_mean
    Var[P] = default_corr * p_mean * (1-p_mean)
    """
    var_p  = default_corr * p_mean * (1 - p_mean)
    common = p_mean * (1 - p_mean) / var_p - 1
    alpha  = p_mean * common
    beta   = (1 - p_mean) * common
    return alpha, beta

# ── Settings ──────────────────────────────────────────────────
N_SIM = 100_000  # Monte Carlo scenarios
ALPHA = 0.99  # confidence level (99% = Basel III standard)
np.random.seed(42)

print(f"\nSettings: {N_SIM:,} scenarios | {ALPHA * 100:.0f}% confidence")


# ================================================================
# SIMULATION FUNCTION
# ================================================================

def simulate_portfolio(table, name):
    """
    Simulate loss distribution using two models.

    MODEL 1 — Gaussian One-Factor (Vasicek 1987):
    Industry standard. Each firm exposed to common factor F
    and own idiosyncratic shock epsilon.
    X_i = rho_i*F + sqrt(1-rho_i^2)*epsilon_i
    Default if X_i <= Phi^{-1}(PD_i)

    MODEL 2 — Beta-Bernoulli (mixture model):
    Default probability itself is random: P ~ Beta(alpha,beta)
    All firms share same P in each scenario → dependence.
    Yi | P ~ Bernoulli(P)

    Parameters:
        table: DataFrame with PD, LGD, EAD, rho per firm
        name:  portfolio name string

    Returns: dict with loss arrays and all risk measures
    """
    print(f"\n{'─' * 65}")
    print(f"  {name}")
    print(f"{'─' * 65}")

    m = len(table)
    PDs = table['PD'].values
    LGDs = table['LGD'].values
    EADs = table['EAD'].values
    rhos = table['rho'].values

    # Normalize EAD: loss expressed as fraction of total portfolio
    w = EADs / EADs.sum()

    # ── MODEL 1: Gaussian One-Factor ──────────────────────────
    # Step 1: Default thresholds d = Phi^{-1}(PD)
    # Firm i defaults when X_i <= d_i
    d = norm.ppf(PDs)

    # Step 2: Common economic factor (one per scenario)
    F = np.random.normal(0, 1, N_SIM)

    # Step 3: Firm-specific shocks (one per firm per scenario)
    eps = np.random.normal(0, 1, (N_SIM, m))

    # Step 4: Creditworthiness for each firm in each scenario
    # Broadcasting: F[:, None] shape (N_SIM,1) × rhos shape (m,)
    X = rhos * F[:, None] + np.sqrt(1 - rhos ** 2) * eps

    # Step 5: Default indicators Y=1 if X<=d, else Y=0
    Y_gf = (X <= d).astype(float)

    # Step 6: Portfolio loss = weighted sum of defaults
    L_gf = Y_gf @ (w * LGDs)

    # Step 7: Analytical EL (exact formula, no simulation needed)
    EL_gf = (PDs * LGDs * w).sum()
    Std_gf = L_gf.std()
    VaR_gf = np.percentile(L_gf, ALPHA * 100)
    ES_gf = L_gf[L_gf > VaR_gf].mean()

    print(f"\n  Gaussian One-Factor Model:")
    print(f"    EL (analytical):  {EL_gf:.6f}")
    print(f"    EL (simulated):   {L_gf.mean():.6f}  ← should match!")
    print(f"    Std:              {Std_gf:.6f}")
    print(f"    VaR {ALPHA * 100:.0f}%:          {VaR_gf:.6f}")
    print(f"    ES  {ALPHA * 100:.0f}%:          {ES_gf:.6f}")

    # ── MODEL 2: Beta-Bernoulli ───────────────────────────────
    # Calibrate Beta parameters from mean PD and default corr
    p_mean = PDs.mean()
    a_bb, b_bb = compute_beta_params(p_mean, default_corr=0.005)

    L_bb = np.zeros(N_SIM)
    for s in range(N_SIM):
        # Draw shared random PD for this economic scenario
        P = np.random.beta(a_bb, b_bb)
        # Each firm defaults independently with probability P
        Y = np.random.binomial(1, P, m).astype(float)
        L_bb[s] = (Y * w * LGDs).sum()

    EL_bb = L_bb.mean()
    Std_bb = L_bb.std()
    VaR_bb = np.percentile(L_bb, ALPHA * 100)
    ES_bb = L_bb[L_bb > VaR_bb].mean()

    print(f"\n  Beta-Bernoulli Model:")
    print(f"    EL:    {EL_bb:.6f}")
    print(f"    Std:   {Std_bb:.6f}")
    print(f"    VaR {ALPHA * 100:.0f}%: {VaR_bb:.6f}")
    print(f"    ES  {ALPHA * 100:.0f}%: {ES_bb:.6f}")

    return {
        'name': name,
        'L_gf': L_gf, 'L_bb': L_bb,
        'EL_gf': EL_gf, 'EL_bb': EL_bb,
        'Std_gf': Std_gf, 'Std_bb': Std_bb,
        'VaR_gf': VaR_gf, 'VaR_bb': VaR_bb,
        'ES_gf': ES_gf, 'ES_bb': ES_bb,
    }


# ── Run both portfolios ───────────────────────────────────────
r1 = simulate_portfolio(table1, "Portfolio 1 — Industry A")
r2 = simulate_portfolio(table2, "Portfolio 2 — Industry B")

# ── Comparison table ──────────────────────────────────────────
print("\n" + "=" * 65)
print("RISK MEASURES COMPARISON")
print("=" * 65)
print(f"\n  {'Measure':<12} {'P1 Gauss':>12} {'P1 Beta-B':>12}"
      f" {'P2 Gauss':>12} {'P2 Beta-B':>12}")
print(f"  {'─' * 62}")

rows = [
    ('EL', r1['EL_gf'], r1['EL_bb'], r2['EL_gf'], r2['EL_bb']),
    ('Std', r1['Std_gf'], r1['Std_bb'], r2['Std_gf'], r2['Std_bb']),
    ('VaR 99%', r1['VaR_gf'], r1['VaR_bb'], r2['VaR_gf'], r2['VaR_bb']),
    ('ES 99%', r1['ES_gf'], r1['ES_bb'], r2['ES_gf'], r2['ES_bb']),
]
for label, *vals in rows:
    print(f"  {label:<12}" + "".join(f" {v:>12.6f}" for v in vals))

# Save → report Table 3
comp_df = pd.DataFrame(rows,
                       columns=['Metric', 'P1_Gaussian', 'P1_BetaBernoulli',
                                'P2_Gaussian', 'P2_BetaBernoulli'])
tbl_path = os.path.join(DIR_RESULTS, 'risk_measures_table.csv')
comp_df.to_csv(tbl_path, index=False)
print(f"\n  Saved: risk_measures_table.csv → report Table 3")

# ── Plot: loss distributions ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    'Simulated Portfolio Loss Distributions\n'
    'Gaussian One-Factor vs Beta-Bernoulli | '
    f'{N_SIM:,} scenarios | {ALPHA * 100:.0f}% confidence',
    fontsize=13, fontweight='bold')

configs = [
    (axes[0, 0], r1['L_gf'], r1['EL_gf'], r1['VaR_gf'], r1['ES_gf'],
     'Portfolio 1 — Gaussian One-Factor', 'steelblue'),
    (axes[0, 1], r1['L_bb'], r1['EL_bb'], r1['VaR_bb'], r1['ES_bb'],
     'Portfolio 1 — Beta-Bernoulli', 'steelblue'),
    (axes[1, 0], r2['L_gf'], r2['EL_gf'], r2['VaR_gf'], r2['ES_gf'],
     'Portfolio 2 — Gaussian One-Factor', 'coral'),
    (axes[1, 1], r2['L_bb'], r2['EL_bb'], r2['VaR_bb'], r2['ES_bb'],
     'Portfolio 2 — Beta-Bernoulli', 'coral'),
]

for ax, L, EL, VaR, ES, title, color in configs:
    ax.hist(L, bins=100, density=True,
            color=color, edgecolor='white', alpha=0.8)
    ax.axvline(EL, color='green', linestyle='--', lw=2,
               label=f'EL  = {EL:.5f}')
    ax.axvline(VaR, color='orange', linestyle='--', lw=2,
               label=f'VaR = {VaR:.5f}')
    ax.axvline(ES, color='red', linestyle='--', lw=2,
               label=f'ES  = {ES:.5f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Portfolio Loss (fraction of total EAD)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
loss_path = os.path.join(DIR_RESULTS, 'loss_distributions.png')
plt.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: loss_distributions.png → report Figure 3")

# ── Save for analysis step ────────────────────────────────────
pickle.dump({'r1': r1, 'r2': r2, 'ALPHA': ALPHA, 'N_SIM': N_SIM},
            open(os.path.join(DIR_RESULTS, 'step3_results.pkl'), 'wb'))

print("\n" + "=" * 65)
print("PART 3 COMPLETE — Next: run 04_analysis_plots.py")
print("=" * 65)