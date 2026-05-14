# ---------------------------------------------------------------
# Script Name: Task 6 - Beta-Bernoulli Defaults
# Author: Shazia Ishaq
# Description: Simulating defaults when the default probability
#              itself is random and shared across all companies.
#
# In Tasks 4 and 5 we used a fixed default probability of 1%.
# In this task the default probability is NOT fixed.
# Instead it changes randomly in each simulation trial.
#
# Think of it like this:
# In a GOOD economy: default probability might be 0.5%
# In a NORMAL economy: default probability might be 1%
# In a BAD economy: default probability might be 3%
#
# We do not know which economy we are in, so we let
# the probability P itself be random: P ~ Beta(alpha, beta)
# Then given P, each company defaults independently: Yi|P ~ Bernoulli(P)
#
# This creates DEPENDENCE between companies because they all
# share the same random economy (same P in each trial).
#
# Parameters:
#   N    = 2000   number of simulations
#   m    = 1000   number of companies
#   p    = 1%     unconditional default probability (average)
#   rho  = 0.5%   default correlation between companies
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# ── Parameters ────────────────────────────────────────────────
num_trials = 2000       # how many times we repeat the experiment
num_obligors = 1000     # number of companies in portfolio
p = 0.01                # unconditional default probability = 1%
rho = 0.005             # default correlation = 0.5%

np.random.seed(42)

# ── Step 1: Calculate Beta Distribution Parameters ───────────
# We need to find alpha and beta so that:
# E[P] = p = 0.01  (average default probability is 1%)
# Var[P] = rho * p * (1-p)  (variance matches our correlation)
#
# Using method of moments formulas:
# First calculate the variance we want
variance = rho * p * (1 - p)

# Then find the common factor
# Think of it as: how spread out is the Beta distribution?
common = p * (1 - p) / variance - 1

# Calculate alpha and beta parameters
alpha = p * common
beta_param = (1 - p) * common

print("=" * 55)
print("   TASK 6: BETA-BERNOULLI DEFAULT MODEL")
print("=" * 55)
print(f"\nModel Setup:")
print(f"Unconditional PD:      {p*100:.1f}%")
print(f"Default correlation:   {rho*100:.1f}%")
print(f"\nBeta Distribution Parameters:")
print(f"alpha = {alpha:.4f}")
print(f"beta  = {beta_param:.4f}")
print(f"\nVerification:")
print(f"E[P] = alpha/(alpha+beta) = "
      f"{alpha/(alpha+beta_param):.4f} (should be {p})")

# ── Step 2: Understand the Beta Distribution ──────────────────
# Beta(alpha, beta) generates random probabilities between 0 and 1
# In our case it generates the default probability for each trial
#
# Small alpha, large beta → probability stays close to 0
# alpha = beta → symmetric around 0.5
# Our parameters give: most values close to 0.01 (1%)
# but occasionally much higher (bad economy!)

print(f"\nBeta Distribution Preview:")
sample_probs = np.random.beta(alpha, beta_param, size=10)
print(f"10 sample default probabilities:")
for i, prob in enumerate(sample_probs):
    print(f"  Trial {i+1}: P = {prob:.4f} ({prob*100:.2f}%)")

# ── Step 3: Run the Simulation ────────────────────────────────
# For each trial:
# 1. Draw a random default probability P ~ Beta(alpha, beta)
#    This represents the state of the economy in this scenario
# 2. Each company defaults independently with probability P
#    Yi | P ~ Bernoulli(P)
# 3. Count how many companies defaulted

print(f"\nRunning {num_trials} simulations...")

# Draw all random default probabilities at once
# Shape: (2000,) one probability per trial
P_sim = np.random.beta(alpha, beta_param, size=num_trials)

print(f"\nSimulated default probability statistics:")
print(f"Mean P:    {P_sim.mean():.4f} (should be close to {p})")
print(f"Min P:     {P_sim.min():.4f}")
print(f"Max P:     {P_sim.max():.4f}")
print(f"Std P:     {P_sim.std():.4f}")

# Count defaults for each trial
default_counts = np.zeros(num_trials, dtype=int)

for i in range(num_trials):
    # In this trial the economy has default probability P_sim[i]
    # Each of the 1000 companies defaults with this probability
    defaults = np.random.binomial(1, P_sim[i], size=num_obligors)
    default_counts[i] = defaults.sum()

# ── Step 4: Analyze the Results ───────────────────────────────
print("\n" + "=" * 55)
print("   SIMULATION RESULTS")
print("=" * 55)
print(f"Number of simulations:        {num_trials}")
print(f"Number of companies:          {num_obligors}")
print(f"Unconditional PD:             {p*100:.1f}%")
print(f"Default correlation:          {rho*100:.1f}%")
print(f"\nExpected defaults per trial:  {num_obligors * p:.1f}")
print(f"Simulated mean defaults:      {default_counts.mean():.4f}")
print(f"Simulated std of defaults:    {default_counts.std():.4f}")
print(f"Maximum defaults in one trial:{default_counts.max()}")
print(f"Minimum defaults in one trial:{default_counts.min()}")

# ── Step 5: Interpretation ────────────────────────────────────
print("\n" + "=" * 55)
print("   INTERPRETATION")
print("=" * 55)
print(f"The Beta-Bernoulli model creates dependence because")
print(f"all companies share the SAME random probability P.")
print(f"When P is high (bad economy) → many defaults together")
print(f"When P is low  (good economy) → few or no defaults")
print(f"\nThis is different from Task 4 where we used a")
print(f"Gaussian factor. Here the mixing is done through")
print(f"the Beta distribution on the probability itself.")
print(f"\nBoth approaches create CLUSTERING of defaults")
print(f"which is what we observe in real financial crises.")

# ── Step 6: Plot Histogram ────────────────────────────────────
plt.figure(figsize=(10, 6))

plt.hist(default_counts,
         bins=range(0, default_counts.max() + 2),
         density=True,
         edgecolor='black',
         color='thistle',
         alpha=0.85)

# Mark the expected number of defaults
plt.axvline(x=num_obligors * p,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label=f'Expected defaults = {num_obligors*p:.0f}')

# Mark the mean from simulation
plt.axvline(x=default_counts.mean(),
            color='blue',
            linestyle='--',
            linewidth=1.5,
            label=f'Simulated mean = {default_counts.mean():.1f}')

plt.title("Beta-Bernoulli Default Model\n"
          f"Portfolio: {num_obligors} obligors, "
          f"PD={p*100:.0f}%, "
          f"Default Correlation={rho*100:.1f}%, "
          f"{num_trials} simulations",
          fontsize=12, fontweight='bold')
plt.xlabel("Number of Defaults")
plt.ylabel("Estimated Probability")
plt.legend()
plt.grid(True, axis='y', linestyle='--',
         color='gray', linewidth=0.5)
plt.tight_layout()
plt.show()
