# ---------------------------------------------------------------
# Script Name: Task 5 - Probability of k Defaults
# Author: Shazia Ishaq
# Description: Comparing the probability of exactly k companies
#              defaulting under two scenarios:
#              1. Independent defaults - calculated using the
#                 Binomial formula (exact math)
#              2. Correlated defaults - estimated from simulation
#                 by counting how often k defaults occurred
#
# This task shows WHY correlation is dangerous:
# The correlated model gives much higher probability of
# extreme losses (many defaults at once) compared to
# the independent model.
#
# Parameters:
#   N    = 2000   number of simulations
#   m    = 1000   number of companies
#   p    = 1%     default probability per company
#   rho  = 0.25   sensitivity to common economic factor
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# ── Parameters ────────────────────────────────────────────────
num_trials = 2000
num_obligors = 1000
prob_default = 0.01
correlation = 0.25
max_defaults = 30       # we look at k = 0, 1, 2, ... 30

non_default_color = 'silver'
default_color = 'thistle'

np.random.seed(42)


# ── Step 1: Simulate Correlated Defaults ──────────────────────
# Same function as Task 4 — we reuse it here
# correlation = 0.25 means companies share common economic risk

def simulate_correlated_defaults(num_trials, num_obligors,
                                  prob_default, correlation):
    # Common economic factor (same for all companies)
    F = np.random.normal(0, 1, num_trials)
    # Individual company shocks
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors))
    # Creditworthiness = economic factor + individual shock
    X = (correlation * F[:, np.newaxis]
         + np.sqrt(1 - correlation**2) * epsilon)
    # Default threshold: company defaults if X falls below this
    threshold = norm.ppf(prob_default)
    # Count defaults per trial
    defaults = (X < threshold).sum(axis=1)
    return defaults


correlated_defaults = simulate_correlated_defaults(
    num_trials, num_obligors, prob_default, correlation
)

# ── Step 2: Independent Probabilities (Binomial Formula) ──────
# For independent defaults we can use EXACT math
# No simulation needed — the Binomial formula gives us
# the exact probability of exactly k defaults:
#
# P(K = k) = C(1000, k) * (0.01)^k * (0.99)^(1000-k)
#
# Where:
# C(1000, k) = number of ways to choose k defaulters from 1000
# (0.01)^k   = probability that exactly k companies default
# (0.99)^(1000-k) = probability that the rest survive

independent_probs = [
    binom.pmf(k, num_obligors, prob_default)
    for k in range(max_defaults + 1)
]

print("=" * 55)
print("   INDEPENDENT DEFAULTS - BINOMIAL PROBABILITIES")
print("=" * 55)
print(f"{'k defaults':<15} {'P(K=k)':>15}")
print("-" * 30)
for k in range(16):
    print(f"{k:<15} {independent_probs[k]:>15.6f}")

# ── Step 3: Correlated Probabilities (Empirical Counting) ─────
# For correlated defaults we CANNOT use a simple formula
# Instead we estimate probabilities from our simulation:
#
# P(K = k) ≈ (number of trials where exactly k defaults) / 2000
#
# Example: if exactly 10 defaults happened in 150 out of 2000 trials
# then P(K=10) ≈ 150/2000 = 0.075
#
# np.mean(correlated_defaults == k) does exactly this:
# it counts how many trials had exactly k defaults
# and divides by total number of trials

correlated_probs = np.zeros(max_defaults + 1)
for k in range(max_defaults + 1):
    correlated_probs[k] = np.mean(correlated_defaults == k)

print("\n" + "=" * 55)
print("   CORRELATED DEFAULTS - EMPIRICAL PROBABILITIES")
print("=" * 55)
print(f"{'k defaults':<15} {'P(K=k)':>15}")
print("-" * 30)
for k in range(16):
    print(f"{k:<15} {correlated_probs[k]:>15.6f}")

# ── Step 4: Compare the Two Distributions ────────────────────
print("\n" + "=" * 55)
print("   COMPARISON - KEY DIFFERENCES")
print("=" * 55)
print(f"{'k defaults':<12} {'Independent':>14} {'Correlated':>14} {'Ratio':>10}")
print("-" * 52)
for k in [0, 5, 10, 15, 20, 25, 30]:
    ind = independent_probs[k]
    cor = correlated_probs[k]
    ratio = cor/ind if ind > 0 else float('inf')
    print(f"{k:<12} {ind:>14.6f} {cor:>14.6f} {ratio:>10.2f}x")

print("\nKey Observation:")
print("For small k (few defaults): independent > correlated")
print("For large k (many defaults): correlated > independent")
print("Correlated model has FATTER TAILS!")
print("This means extreme losses are much more likely")
print("when companies share a common economic factor.")

# ── Step 5: Plot the Probabilities ───────────────────────────
plt.figure(figsize=(12, 6))

# Independent: exact binomial probabilities (circles)
plt.plot(range(max_defaults + 1),
         independent_probs,
         label='Independent Defaults (Binomial formula)',
         color=default_color,
         marker='o',
         linewidth=2,
         markersize=6)

# Correlated: empirical probabilities from simulation (crosses)
plt.plot(range(max_defaults + 1),
         correlated_probs,
         label=f'Correlated Defaults (rho={correlation}, empirical)',
         color=non_default_color,
         marker='x',
         linewidth=2,
         markersize=8)

plt.title('Probability of k Defaults: Independent vs Correlated\n'
          f'{num_obligors} obligors, PD={prob_default*100}%, '
          f'{num_trials} simulations',
          fontsize=12, fontweight='bold')
plt.xlabel('Number of Defaults (k)')
plt.ylabel('Probability P(K = k)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()