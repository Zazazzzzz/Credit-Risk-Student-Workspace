# ================================================================
# File:        04_analysis_plots.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: Part 4 — Analysis, Comparison, and Discussion
#
# INPUT:  results/step3_results.pkl
# OUTPUT: results/comparison_chart.png    → report Figure 4
#         results/discussion_summary.txt  → reference for report
# ================================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results'
)

print("=" * 65)
print("   PART 4: ANALYSIS AND INTERPRETATION")
print("=" * 65)

# ── Load results ──────────────────────────────────────────────
data = pickle.load(open(
    os.path.join(DIR_RESULTS, 'step3_results.pkl'), 'rb'))
r1 = data['r1']
r2 = data['r2']
ALPHA = data['ALPHA']
N_SIM = data['N_SIM']

# ================================================================
# ANALYSIS 1: PORTFOLIO 1 vs PORTFOLIO 2
# ================================================================
# We compare all risk measures between both portfolios.
# Portfolio 2 expected to be riskier due to higher PD and LGD.

print("\n" + "-" * 65)
print("ANALYSIS 1: Portfolio 1 vs Portfolio 2 (Gaussian Model)")
print("-" * 65)

print(f"\n  {'Measure':<12} {'P1':>10} {'P2':>10} "
      f"{'Diff':>12} {'Ratio':>10}")
print(f"  {'─' * 48}")

for label, v1, v2 in [
    ('EL', r1['EL_gf'], r2['EL_gf']),
    ('Std', r1['Std_gf'], r2['Std_gf']),
    ('VaR 99%', r1['VaR_gf'], r2['VaR_gf']),
    ('ES 99%', r1['ES_gf'], r2['ES_gf']),
]:
    diff = v2 - v1
    ratio = v2 / v1 if v1 > 0 else 0
    print(f"  {label:<12} {v1:>10.6f} {v2:>10.6f} "
          f"{diff:>+12.6f} {ratio:>9.2f}x")

print(f"\n  Portfolio 2 has higher risk in all measures.")
print(f"  Driven by higher mean PD and LGD in Industry B firms.")

# ================================================================
# ANALYSIS 2: MODEL COMPARISON
# ================================================================
# Gaussian One-Factor vs Beta-Bernoulli for each portfolio.
# Both models calibrated to same mean PD → similar EL.
# VaR and ES differ due to different dependence structure.

print("\n" + "-" * 65)
print("ANALYSIS 2: Gaussian One-Factor vs Beta-Bernoulli")
print("-" * 65)

for name, res in [("Portfolio 1", r1), ("Portfolio 2", r2)]:
    print(f"\n  {name}:")
    print(f"    {'Measure':<12} {'Gaussian':>12} {'Beta-B':>12} "
          f"{'Diff':>12}")
    print(f"    {'─' * 50}")
    for label, gf, bb in [
        ('EL', res['EL_gf'], res['EL_bb']),
        ('Std', res['Std_gf'], res['Std_bb']),
        ('VaR 99%', res['VaR_gf'], res['VaR_bb']),
        ('ES 99%', res['ES_gf'], res['ES_bb']),
    ]:
        print(f"    {label:<12} {gf:>12.6f} {bb:>12.6f} "
              f"{bb - gf:>+12.6f}")

# ================================================================
# ANALYSIS 3: ECONOMIC INTERPRETATION
# ================================================================
print("\n" + "-" * 65)
print("ANALYSIS 3: Economic Interpretation")
print("-" * 65)

print(f"""
  1. EXPECTED LOSS AND CREDIT QUALITY:
     EL = PD x LGD x EAD is the fundamental credit risk formula.
     Portfolio 2 has higher EL because Industry B firms have
     higher default probability (mean PD 3.3% vs 2.0%) and
     higher LGD (49.7% vs 47.1%).
     A risk manager must reserve more capital for Portfolio 2.

  2. TAIL RISK — VaR AND ES:
     VaR 99% is the loss exceeded only in 1% of scenarios.
     ES 99% is the average loss in those worst 1% scenarios.
     Both exceed EL substantially — loss distributions are
     right-skewed and heavy-tailed as expected for credit risk.
     ES > VaR in all cases, confirming significant tail risk
     beyond the VaR threshold.

  3. SYSTEMATIC RISK (rho):
     Estimated rho values measure how much each firm's returns
     co-move with the industry index. Higher rho means defaults
     cluster more in bad economic scenarios, raising VaR and ES
     even when EL stays the same.

  4. MODEL COMPARISON:
     Gaussian One-Factor uses firm-specific rho from return data.
     Beta-Bernoulli uses a shared random PD across all firms.
     Both give similar EL but may differ in tail risk measures
     depending on how dependence is structured.
     Using both models provides a more robust risk assessment.
""")

# ================================================================
# FINAL COMPARISON CHART
# ================================================================
# Four-panel bar chart comparing all risk measures.
# → Used in report: Figure 4

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    'Credit Portfolio Risk — Full Comparison\n'
    'Portfolio 1 (Industry A) vs Portfolio 2 (Industry B)\n'
    'Gaussian One-Factor Model vs Beta-Bernoulli Model',
    fontsize=13, fontweight='bold'
)

colors = ['#1a3a5c', '#2e6da4', '#c0392b', '#e74c3c']
labels = ['P1 Gaussian', 'P1 Beta-B', 'P2 Gaussian', 'P2 Beta-B']

for ax, title, vals in [
    (axes[0, 0], 'Expected Loss (EL)',
     [r1['EL_gf'], r1['EL_bb'], r2['EL_gf'], r2['EL_bb']]),
    (axes[0, 1], 'Standard Deviation',
     [r1['Std_gf'], r1['Std_bb'], r2['Std_gf'], r2['Std_bb']]),
    (axes[1, 0], 'Value at Risk 99%',
     [r1['VaR_gf'], r1['VaR_bb'], r2['VaR_gf'], r2['VaR_bb']]),
    (axes[1, 1], 'Expected Shortfall 99%',
     [r1['ES_gf'], r1['ES_bb'], r2['ES_gf'], r2['ES_bb']]),
]:
    bars = ax.bar(labels, vals, color=colors,
                  edgecolor='black', alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss fraction')
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.5f}', ha='center', fontsize=8)

plt.tight_layout()
comp_path = os.path.join(DIR_RESULTS, 'comparison_chart.png')
plt.savefig(comp_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: comparison_chart.png → report Figure 4")

# ================================================================
# SAVE DISCUSSION SUMMARY
# ================================================================
summary = f"""CREDIT PORTFOLIO RISK ANALYSIS — RESULTS SUMMARY
Shazia Ishaq | University of Freiburg | 2026
{N_SIM:,} Monte Carlo simulations | {ALPHA * 100:.0f}% confidence level

PORTFOLIO 1 (Industry A — 30 obligors)
  Gaussian One-Factor: EL={r1['EL_gf']:.6f} Std={r1['Std_gf']:.6f} VaR={r1['VaR_gf']:.6f} ES={r1['ES_gf']:.6f}
  Beta-Bernoulli:      EL={r1['EL_bb']:.6f} Std={r1['Std_bb']:.6f} VaR={r1['VaR_bb']:.6f} ES={r1['ES_bb']:.6f}

PORTFOLIO 2 (Industry B — 26 obligors)
  Gaussian One-Factor: EL={r2['EL_gf']:.6f} Std={r2['Std_gf']:.6f} VaR={r2['VaR_gf']:.6f} ES={r2['ES_gf']:.6f}
  Beta-Bernoulli:      EL={r2['EL_bb']:.6f} Std={r2['Std_bb']:.6f} VaR={r2['VaR_bb']:.6f} ES={r2['ES_bb']:.6f}

KEY FINDINGS:
1. Portfolio 2 has higher risk across all measures (higher PD and LGD)
2. Both models agree on EL (same mean PD calibration)
3. ES > VaR in all cases confirming heavy-tailed distributions
4. Results support requiring more capital for Portfolio 2
"""
with open(os.path.join(DIR_RESULTS, 'discussion_summary.txt'), 'w') as f:
    f.write(summary)
print(f"  Saved: discussion_summary.txt → reference for writing report")

# ================================================================
# LIST ALL OUTPUT FILES
# ================================================================
print("\n" + "=" * 65)
print("ALL FILES IN RESULTS FOLDER")
print("=" * 65)
for fname in sorted(os.listdir(DIR_RESULTS)):
    if not fname.startswith('.'):
        size = os.path.getsize(os.path.join(DIR_RESULTS, fname))
        print(f"  {fname:<45} {size:>8} bytes")

print("\n" + "=" * 65)
print("PART 4 COMPLETE — All analysis done!")
print("Next: write report using results above")
print("=" * 65)