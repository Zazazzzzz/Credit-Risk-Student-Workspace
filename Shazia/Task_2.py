# ---------------------------------------------------------------
# Script Name: Task 2 - Portfolio Default Simulation
# Author: Shazia Ishaq
# Description: Simulation of default outcomes in a homogeneous
#              credit portfolio under independence assumption.
#
# Here each obligor defaults independently with the
# same probability p. This is the simplest portfolio credit
# risk model — the binomial model.

# Key Question: If 100 firms each have 1% default probability,
# how many defaults should we expect? And how much uncertainty
# is there around that expectation?

# Under independence, the number of defaults K follows:
#   K ~ Binomial(m, p)
#   E[K] = m * p = 100 * 0.01 = 1
#   Var[K] = m * p * (1-p) = 100 * 0.01 * 0.99 = 0.99
#
# Parameters:
#   N = 1000  number of simulation trials
#   p = 0.01  default probability per obligor
#   m = 100   number of obligors in portfolio
# ---------------------------------------------------------------

# Library
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ── Parameters ────────────────────────────────────────────────
N = 1000    # number of simulation trials
p = 0.01    # marginal default probability for each obligor
m = 100     # number of obligors in the portfolio

np.random.seed(42)

# ── Step 1: Simulate Independent Defaults ─────────────────────
# For each trial (row) and each obligor (column):
# Draw Yi ~ Bernoulli(p) — 1 means default, 0 means survival
# np.random.binomial(1, p, size=(N, m)) generates an N x m
# matrix of independent Bernoulli random variables
defaults = np.random.binomial(1, p, size=(N, m))

# ── Step 2: Count Defaults Per Trial ──────────────────────────
# Sum across columns (axis=1) to get total defaults per trial
# Result: vector of length N, each entry = number of defaults
# in that simulation
default_counts = defaults.sum(axis=1)

# ── Step 3: Analytical Benchmarks ────────────────────────────
# Under Binomial(m, p):
expected_defaults = m * p
variance_defaults = m * p * (1 - p)
std_defaults = np.sqrt(variance_defaults)

# ── Step 4: Print Results ─────────────────────────────────────
print("=" * 55)
print("   PORTFOLIO DEFAULT SIMULATION - RESULTS")
print("=" * 55)
print(f"Portfolio size:                  {m} obligors")
print(f"Default probability per obligor: {p*100:.1f}%")
print(f"Number of simulations:           {N}")
print("-" * 55)
print(f"Analytical expected defaults:    {expected_defaults:.4f}")
print(f"Simulated mean defaults:         {default_counts.mean():.4f}")
print(f"Analytical std of defaults:      {std_defaults:.4f}")
print(f"Simulated std of defaults:       {default_counts.std():.4f}")
print(f"Maximum defaults in one trial:   {default_counts.max()}")
print(f"Minimum defaults in one trial:   {default_counts.min()}")
print("-" * 55)

# ── Step 5: Probability of Zero Defaults ─────────────────────
# Very important in credit risk: how often does nothing happen?
prob_zero = binom.pmf(0, m, p)
prob_zero_sim = np.mean(default_counts == 0)
print(f"Analytical P(0 defaults):        {prob_zero:.4f}")
print(f"Simulated  P(0 defaults):        {prob_zero_sim:.4f}")

# Probability of more than 5 defaults (tail risk)
prob_tail = 1 - binom.cdf(5, m, p)
prob_tail_sim = np.mean(default_counts > 5)
print(f"Analytical P(>5 defaults):       {prob_tail:.6f}")
print(f"Simulated  P(>5 defaults):       {prob_tail_sim:.6f}")
print("=" * 55)

# ── Step 6: Interpretation ────────────────────────────────────
print("\nInterpretation:")
print(f"On average {expected_defaults:.1f} out of {m} firms default per year.")
print(f"In {prob_zero*100:.1f}% of scenarios no defaults occur at all.")
print(f"Extreme losses (>5 defaults) occur in only "
      f"{prob_tail*100:.4f}% of scenarios.")
print("This reflects the illusion of stability mentioned in the lecture.")
print("Under independence, tail risk is very small.")
print("Correlation between defaults changes this dramatically (Task 4).")

# ── Step 7: Plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Simulated histogram
ax.hist(default_counts, bins=range(0, 16),
        density=True, edgecolor='black',
        color='thistle', alpha=0.7,
        label='Simulated distribution')

# Overlay theoretical binomial PMF
k_values = range(0, 16)
binomial_probs = [binom.pmf(k, m, p) for k in k_values]
ax.plot(k_values, binomial_probs, 'bo-',
        markersize=5, linewidth=1.5,
        label='Theoretical Binomial PMF')

# Mark expected value
ax.axvline(x=expected_defaults, color='red',
           linestyle='--', linewidth=1.5,
           label=f'Expected defaults = {expected_defaults:.1f}')

ax.set_title("Portfolio Default Simulation - Independent Defaults\n"
             f"{m} obligors, PD={p*100:.0f}%, {N} simulations",
             fontsize=12)
ax.set_xlabel("Number of Defaults")
ax.set_ylabel("Density / Probability")
ax.legend()
ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.5)
plt.tight_layout()
plt.show()