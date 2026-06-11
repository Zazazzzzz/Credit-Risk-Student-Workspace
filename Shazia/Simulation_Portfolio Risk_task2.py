# ---------------------------
# Script Name: Simulation_Portfolio Risk:task2.py
# Author: Shazia Ishaq
# Description: Simulation and Analytical credit Portfolio Risk
# Data : using two portfiolios 1 and 2

# -------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

print("=" * 65)
print("Credit Risk Task 2")
print("Simulation and Analytical Portfolio Risk")

 # ------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Credit Risk', 'portfolio.xlsx')

try:
    p1 = pd.read_excel(data_path, sheet_name='portfolio1')
    p2 = pd.read_excel(data_path, sheet_name='portfolio2')
    print(f"Loaded: {os.path.normpath(data_path)}")
except:
    # Fallback — exact copy of professor's data
    base = {
        'obligors': ['CO1', 'CO2', 'CO3', 'CO4', 'CO5',
                     'CO6', 'CO7', 'CO8', 'CO9', 'CO10'],
        'exposure': [0.10, 0.50, 0.01, 0.02, 0.05,
                     0.03, 0.07, 0.09, 0.06, 0.07],
        'LGD': [0.4] * 10,
        'rho': [0.30, 0.25, 0.25, 0.30, 0.27,
                0.27, 0.28, 0.29, 0.27, 0.27],
    }
    p1 = pd.DataFrame({**base,
                       'rating': ['AAA', 'A', 'BBB', 'BB', 'BBB',
                                  'AA', 'A', 'A', 'A', 'A']})
    p2 = pd.DataFrame({**base,
                       'rating': ['AAA', 'B', 'BBB', 'BB', 'BBB',
                                  'AA', 'A', 'A', 'A', 'A']})
    print("Using built-in data (same as professor's file)")

print("\n=== Portfolio 1 ===")
print(p1.to_string(index=False))
print("\n=== Portfolio 2 ===")
print(p2.to_string(index=False))
print("\nOnly difference: CO2 goes from A (Portfolio 1) to B (Portfolio 2)")
print("CO2 exposure = 0.50 — this is 50% of total portfolio!")

# ── Rating → PD mapping (Moody's 2023) ─────────────────────────
rating_to_pd = {
    'AAA': 0.0001, 'AA': 0.0002, 'A': 0.0006,
    'BBB': 0.0020, 'BB': 0.0150, 'B': 0.0500, 'CCC': 0.2000,
}


# ===============================================================
# CORE FUNCTION
# ===============================================================

def solve_portfolio(df, name, rho_override=None, n_sim=100_000):
    print(f"\n{'=' * 65}")
    print(f"   {name}")
    print(f"{'=' * 65}")

    np.random.seed(42)
    m = len(df)
    companies = df['obligors'].values
    exposures = df['exposure'].values
    LGDs = df['LGD'].values
    ratings = df['rating'].values
    rhos = np.array(rho_override) if rho_override else df['rho'].values
    PDs = np.array([rating_to_pd[r] for r in ratings])

    # ── STEP 1: Ratings → PD ───────────────────────────────────
    print(f"\nSTEP 1: Rating → PD mapping")
    print(f"{'Obligor':<7} {'Rating':>6} {'PD':>8} "
          f"{'Exposure':>9} {'LGD':>5} {'rho':>5}")
    print("-" * 42)
    for i in range(m):
        print(f"{companies[i]:<7} {ratings[i]:>6} {PDs[i] * 100:>7.3f}% "
              f"{exposures[i]:>9.2f} {LGDs[i]:>5.1f} {rhos[i]:>5.2f}")

    # ── STEP 2: PD → Threshold d = Phi^{-1}(PD) ───────────────
    thresholds = norm.ppf(PDs)
    print(f"\nSTEP 2: Thresholds d = Phi^(-1)(PD)")
    for i in range(m):
        print(f"  {companies[i]}: PD={PDs[i] * 100:.3f}% → "
              f"d={thresholds[i]:.4f}")

    # ── STEP 3: Simulate correlated defaults ───────────────────
    # X_i = rho_i * F + sqrt(1-rho_i^2) * epsilon_i
    # Default if X_i <= d_i
    print(f"\nSTEP 3: Simulating {n_sim:,} scenarios...")

    F = np.random.normal(0, 1, n_sim)
    epsilon = np.random.normal(0, 1, (n_sim, m))
    X = (rhos * F[:, np.newaxis]
         + np.sqrt(1 - rhos ** 2) * epsilon)
    defaults = (X <= thresholds).astype(float)

    # ── STEP 4: Portfolio loss L = sum(EAD * LGD * Y) ──────────
    loss_w = exposures * LGDs
    L = defaults @ loss_w
    EL_ana = (PDs * LGDs * exposures).sum()

    print(f"\nSTEP 4: Loss calculation")
    print(f"  Simulated EL:  {L.mean():.6f}")
    print(f"  Analytical EL: {EL_ana:.6f}")
    print(f"  Individual EL contributions:")
    ind_EL = PDs * LGDs * exposures
    for i in range(m):
        bar = '|' * int(ind_EL[i] / EL_ana * 40)
        print(f"    {companies[i]}: {ind_EL[i]:.6f} "
              f"({ind_EL[i] / EL_ana * 100:5.1f}%) {bar}")

    # ── STEP 5: Simulated VaR and ES ───────────────────────────
    alpha = 0.99
    VaR_sim = np.percentile(L, alpha * 100)
    ES_sim = L[L > VaR_sim].mean()

    print(f"\nSTEP 5: Simulated VaR and ES (alpha=99%)")
    print(f"  VaR 99% = {VaR_sim:.6f}")
    print(f"  ES  99% = {ES_sim:.6f}")

    # ── STEP 6: Analytical l(F) ─────────────────────────────────
    # l(F) = sum_i EAD_i * LGD_i * Phi((d_i - rho_i*F)/sqrt(1-rho_i^2))
    F_grid = np.linspace(-4, 4, 2000)
    l_F = np.array([
        (exposures * LGDs *
         norm.cdf((thresholds - rhos * f) / np.sqrt(1 - rhos ** 2))).sum()
        for f in F_grid
    ])

    print(f"\nSTEP 6: Analytical l(F)")
    print(f"  l(F=-4) = {l_F[0]:.6f}  (worst economy)")
    print(f"  l(F= 0) = {l_F[1000]:.6f}  (average economy)")
    print(f"  l(F=+4) = {l_F[-1]:.6f}  (best economy)")

    # ── STEP 7: Analytical VaR and ES ──────────────────────────
    # F_alpha = -Phi^{-1}(alpha) = threshold for worst 1%
    F_alpha = -norm.ppf(alpha)
    idx = np.argmin(np.abs(F_grid - F_alpha))
    VaR_ana = l_F[idx]

    mask = F_grid <= F_alpha
    ES_ana = (np.trapezoid(l_F[mask] * norm.pdf(F_grid[mask]), F_grid[mask])
              / norm.cdf(F_alpha))

    print(f"\nSTEP 7: Analytical VaR and ES")
    print(f"  F_alpha = -Phi^(-1)(0.99) = {F_alpha:.4f}")
    print(f"  VaR 99% = {VaR_ana:.6f}")
    print(f"  ES  99% = {ES_ana:.6f}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n  {'Measure':<18} {'Simulation':>12} {'Analytical':>12}")
    print(f"  {'-' * 44}")
    print(f"  {'Expected Loss':<18} {L.mean():>12.6f} {EL_ana:>12.6f}")
    print(f"  {'VaR 99%':<18} {VaR_sim:>12.6f} {VaR_ana:>12.6f}")
    print(f"  {'ES 99%':<18} {ES_sim:>12.6f} {ES_ana:>12.6f}")

    # ── STEP 8: Plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{name}', fontsize=12, fontweight='bold')

    # Chart 1: Loss distribution
    ax = axes[0]
    ax.hist(L, bins=150, density=True,
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(L.mean(), color='green', linestyle='--', lw=2,
               label=f'EL = {L.mean():.4f}')
    ax.axvline(VaR_sim, color='orange', linestyle='--', lw=2,
               label=f'VaR 99% = {VaR_sim:.4f}')
    ax.axvline(ES_sim, color='red', linestyle='--', lw=2,
               label=f'ES 99% = {ES_sim:.4f}')
    ax.set_title('Simulated Loss Distribution')
    ax.set_xlabel('Portfolio Loss L')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 2: Analytical l(F)
    ax = axes[1]
    ax.plot(F_grid, l_F, color='navy', lw=2, label='l(F)')
    ax.axvline(F_alpha, color='red', linestyle='--', lw=1.5,
               label=f'F_0.99 = {F_alpha:.3f}')
    ax.axhline(VaR_ana, color='orange', linestyle=':', lw=1.5,
               label=f'VaR = {VaR_ana:.4f}')
    ax.fill_between(F_grid, l_F, where=mask,
                    alpha=0.3, color='red', label='ES region')
    ax.set_title('Analytical Conditional Loss l(F)')
    ax.set_xlabel('Economic Factor F  (← bad economy)')
    ax.set_ylabel('l(F)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # Chart 3: Comparison
    ax = axes[2]
    labels = ['EL', 'VaR 99%', 'ES 99%']
    sv = [L.mean(), VaR_sim, ES_sim]
    av = [EL_ana, VaR_ana, ES_ana]
    x = np.arange(3)
    w = 0.35
    b1 = ax.bar(x - w / 2, sv, w, label='Simulation',
                color='steelblue', edgecolor='black', alpha=0.85)
    b2 = ax.bar(x + w / 2, av, w, label='Analytical',
                color='coral', edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Simulation vs Analytical')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    for b, v in zip(b1, sv):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                f'{v:.4f}', ha='center', fontsize=7)
    for b, v in zip(b2, av):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0001,
                f'{v:.4f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.show()

    return {'EL_sim': L.mean(), 'VaR_sim': VaR_sim, 'ES_sim': ES_sim,
            'EL_ana': EL_ana, 'VaR_ana': VaR_ana, 'ES_ana': ES_ana}


# ===============================================================
# RUN ALL 4 CASES
# ===============================================================

new_rho = [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7]

r1 = solve_portfolio(p1, "Portfolio 1 | CO2=A | original rho")
r2 = solve_portfolio(p2, "Portfolio 2 | CO2=B | original rho")
r1_new = solve_portfolio(p1, "Portfolio 1 | CO2=A | new rho",
                         rho_override=new_rho)
r2_new = solve_portfolio(p2, "Portfolio 2 | CO2=B | new rho",
                         rho_override=new_rho)

# ── Final comparison table ──────────────────────────────────────
print("\n" + "=" * 72)
print("   FINAL COMPARISON TABLE")
print("=" * 72)
print(f"{'Case':<38} {'EL':>9} {'VaR 99%':>9} {'ES 99%':>9}")
print("-" * 72)
for label, res in [
    ("P1: CO2=A, original rho", r1),
    ("P1: CO2=A, new rho", r1_new),
    ("P2: CO2=B, original rho", r2),
    ("P2: CO2=B, new rho", r2_new),
]:
    print(f"{label:<38} {res['EL_sim']:>9.6f} "
          f"{res['VaR_sim']:>9.6f} {res['ES_sim']:>9.6f}")

# ── Discussion ──────────────────────────────────────────────────
print("""
DISCUSSION:
───────────────────────────────────────────────────────────────
1. PD AND EXPECTED LOSS:
   CO2 has exposure=0.50 (50% of portfolio).
   P1: CO2=A (PD=0.06%) → CO2 contributes little to EL
   P2: CO2=B (PD=5.00%) → CO2 dominates EL completely!
   One large risky counterparty can drive the whole portfolio.

2. RHO AND TAIL RISK:
   Higher rho → companies default together in bad economy
   → VaR and ES increase significantly
   → EL stays almost the same (rho does not affect average!)

3. WHY RHO MATTERS FOR VaR BUT NOT EL:
   EL = PD × LGD × Exposure — no rho in this formula!
   Rho only changes clustering of losses around the average.
   Same mean, much bigger extremes = higher VaR and ES.

4. SIMULATION vs ANALYTICAL DIFFERENCES:
   Simulation has random noise (finite scenarios).
   The discrete spike when CO2 defaults (loss jumps by 0.20)
   is harder to capture analytically.
   More simulations → closer to analytical result.

5. CO2 CONCENTRATION RISK:
   CO2 = 50% of portfolio exposure.
   If CO2 defaults: instant loss of 0.50×0.40 = 0.20 (20%!)
   This single company dominates VaR and ES in Portfolio 2.
   Real solution: reduce CO2 exposure or require collateral.
""")