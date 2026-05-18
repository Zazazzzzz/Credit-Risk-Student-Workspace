# ---------------------------------------------------------------
# Script Name: Portfolio Default Distributions
# Author: Shazia Ishaq
# Description: Comparing how many companies default in a portfolio
#              under two very different assumptions:
#              1. Each company defaults independently (no connection)
#              2. Companies are connected through a shared economy
#                 (Gaussian one-factor model)
#
# WHY DOES THIS MATTER?
# ---------------------
# Banks hold loans to many companies at once (a portfolio).
# The big question is: how many companies default AT THE SAME TIME?
#
# If defaults are INDEPENDENT:
# -> Companies default one by one randomly
# -> Very unlikely that many default together
# -> Bank can predict losses quite accurately
#
# If defaults are CORRELATED:
# -> When economy is bad ALL companies struggle together
# -> Many can default AT THE SAME TIME
# -> Bank faces much larger unexpected losses!
# -> This is what happened in 2008 financial crisis!
#
# THE GAUSSIAN ONE-FACTOR MODEL:
# Each company i has a creditworthiness variable X_i:
#
#   X_i = rho * F + sqrt(1 - rho^2) * epsilon_i
#
# Where:
#   F         = common economic factor (same for ALL companies)
#   epsilon_i = company-specific shock (different for each company)
#   rho       = how much the economy affects each company
#
# Company i defaults if X_i falls below the default threshold:
#   threshold = Phi^(-1)(p)  (inverse normal of default probability)
#
# When rho=0: companies are independent (no common factor)
# When rho>0: companies share economic risk (correlated defaults)
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# Color choices for plots
non_default_color = 'silver'
default_color = 'thistle'

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

num_trials = 2000
# How many times we repeat the whole experiment
# More trials = more accurate probability estimates

num_obligors = 1000
# Number of companies in our portfolio
# A large portfolio of 1000 companies

prob_default = 0.01
# Each company has 1% chance of defaulting
# This is the MARGINAL default probability
# Same for all companies (homogeneous portfolio)

correlation = 0.25
# How strongly companies are connected to the common factor
# rho = 0.25 means 25% of each company risk comes from economy
# The remaining 75% is company-specific risk
# Higher rho = more clustering of defaults = more dangerous!

# ---------------------------------------------------------------
# THE CORE FUNCTION: SIMULATE DEFAULTS
# ---------------------------------------------------------------
# This function works for BOTH independent and correlated cases:
# When correlation=0  -> pure independent defaults
# When correlation>0  -> correlated defaults (shared factor)

def simulate_correlated_defaults(num_trials, num_obligors,
                                  prob_default, correlation):
    # STEP A: Generate the common economic factor F
    # F represents the state of the economy in each simulation
    # F > 0 = good economy (companies do well)
    # F < 0 = bad economy (companies struggle)
    # Shape: (num_trials,) = one economy state per simulation
    F = np.random.normal(0, 1, num_trials)

    # STEP B: Generate company-specific random shocks
    # epsilon_i is the unique risk of each individual company
    # Independent of the economy and of other companies
    # Shape: (num_trials, num_obligors) = one shock per company per trial
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors))

    # STEP C: Compute creditworthiness variable X_i for each company
    # X_i = rho * F + sqrt(1-rho^2) * epsilon_i
    #
    # rho * F              = systematic (economy) component
    # sqrt(1-rho^2)*epsilon = idiosyncratic (company-specific) component
    #
    # The weights rho and sqrt(1-rho^2) are chosen so that:
    # Var(X_i) = rho^2 + (1-rho^2) = 1  (standard normal variance)
    #
    # F[:, np.newaxis] reshapes F from (2000,) to (2000,1)
    # This allows broadcasting: adding (2000,1) to (2000,1000)
    # NumPy automatically copies F across all 1000 companies
    # Result X has shape (2000, 1000)
    X = (correlation * F[:, np.newaxis]
         + np.sqrt(1 - correlation**2) * epsilon)

    # STEP D: Calculate default threshold
    # A company defaults if its creditworthiness X falls below this
    # norm.ppf(0.01) = -2.326
    # This means: 1% of standard normal distribution is below -2.326
    # So if X < -2.326 the company defaults (probability = 1%)
    threshold = norm.ppf(prob_default)

    # STEP E: Count defaults in each trial
    # X < threshold gives True/False for each company (shape 2000x1000)
    # .sum(axis=1) counts True values across companies for each trial
    # Result shape: (2000,) = number of defaults per simulation
    defaults = (X < threshold).sum(axis=1)

    return defaults


# ---------------------------------------------------------------
# RUN SIMULATIONS
# ---------------------------------------------------------------

np.random.seed(42)

# SCENARIO 1: Independent defaults (rho = 0)
# No common factor - each company only has its own risk
# This is the SIMPLE case - like flipping 1000 biased coins
independent_defaults = simulate_correlated_defaults(
    num_trials, num_obligors, prob_default, 0
)

# SCENARIO 2: Correlated defaults (rho = 0.25)
# All companies share the same economic factor F
# When economy is bad (F low) -> many companies default together!
correlated_defaults = simulate_correlated_defaults(
    num_trials, num_obligors, prob_default, correlation
)

# ---------------------------------------------------------------
# COMPARE THE TWO SCENARIOS
# ---------------------------------------------------------------

print("=" * 60)
print("   PORTFOLIO DEFAULT DISTRIBUTIONS - COMPARISON")
print("=" * 60)
print(f"Portfolio: {num_obligors} obligors, PD={prob_default*100}%")
print(f"Simulations: {num_trials}")
print("-" * 60)
print(f"{'Metric':<35} {'Independent':>12} {'Correlated':>12}")
print("-" * 60)
print(f"{'Mean defaults':<35} "
      f"{independent_defaults.mean():>12.2f} "
      f"{correlated_defaults.mean():>12.2f}")
print(f"{'Std of defaults':<35} "
      f"{independent_defaults.std():>12.2f} "
      f"{correlated_defaults.std():>12.2f}")
print(f"{'Max defaults in one trial':<35} "
      f"{independent_defaults.max():>12} "
      f"{correlated_defaults.max():>12}")
print(f"{'Min defaults in one trial':<35} "
      f"{independent_defaults.min():>12} "
      f"{correlated_defaults.min():>12}")
print("=" * 60)

print("\nKEY INSIGHT:")
print(f"Both scenarios have similar MEAN defaults")
print(f"({independent_defaults.mean():.1f} vs "
      f"{correlated_defaults.mean():.1f})")
print(f"But correlated case has MUCH HIGHER standard deviation")
print(f"({independent_defaults.std():.2f} vs "
      f"{correlated_defaults.std():.2f})")
print(f"This means extreme losses are much more likely!")
print(f"In bad economic times all companies can default together.")

# ---------------------------------------------------------------
# PLOT 1: HISTOGRAMS SIDE BY SIDE
# ---------------------------------------------------------------
# Left plot: independent defaults - narrow bell curve
# Right plot: correlated defaults - wide flat distribution with fat tails
#
# Using same y-axis scale to make comparison fair!

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: independent defaults histogram
ax1.hist(independent_defaults,
         bins=range(0, 31),
         density=True,
         color=default_color,
         edgecolor=non_default_color,
         label='Independent')
ax1.set_title('Independent Defaults\n'
              'Each company flips its own coin')
ax1.set_xlabel('Number of Defaults')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.5)

# Right: correlated defaults histogram
ax2.hist(correlated_defaults,
         bins=range(0, 31),
         density=True,
         color=non_default_color,
         edgecolor=default_color,
         label=f'Correlated (rho={correlation})')
ax2.set_title('Correlated Defaults\n'
              'All companies share common economic factor')
ax2.set_xlabel('Number of Defaults')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.5)

# Use same y-axis for fair comparison
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

fig.suptitle('Default Distribution: Independent vs Correlated\n'
             f'{num_obligors} obligors, PD={prob_default*100}%, '
             f'{num_trials} simulations',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# PLOT 2: PROBABILITY OF K DEFAULTS
# ---------------------------------------------------------------
# Instead of showing counts we show PROBABILITIES
# How likely is it that EXACTLY k companies default?
#
# INDEPENDENT: use exact Binomial formula
# P(K=k) = C(1000,k) * (0.01)^k * (0.99)^(1000-k)
# This is mathematically exact - no simulation needed!
#
# CORRELATED: estimate from simulation
# P(K=k) = (number of trials with exactly k defaults) / 2000
# np.mean(correlated_defaults == k) does this efficiently

max_defaults = 30

# Independent: exact binomial probabilities
independent_probs = [
    binom.pmf(k, num_obligors, prob_default)
    for k in range(max_defaults + 1)
]

# Correlated: empirical probabilities from simulation
correlated_probs = np.zeros(max_defaults + 1)
for k in range(max_defaults + 1):
    correlated_probs[k] = np.mean(correlated_defaults == k)

print("\nProbability Comparison:")
print(f"{'k':<5} {'P(K=k) Independent':>22} {'P(K=k) Correlated':>20}")
print("-" * 50)
for k in range(16):
    print(f"{k:<5} {independent_probs[k]:>22.6f} "
          f"{correlated_probs[k]:>20.6f}")

print("\nKEY OBSERVATION:")
print("For small k: Independent > Correlated")
print("For large k: Correlated > Independent")
print("Correlation creates FAT TAILS in loss distribution!")
print("Extreme losses are much more likely with correlation.")

plt.figure(figsize=(12, 6))
plt.plot(range(max_defaults + 1),
         independent_probs,
         label='Independent (Binomial formula)',
         color=default_color,
         marker='o',
         linewidth=2,
         markersize=5)
plt.plot(range(max_defaults + 1),
         correlated_probs,
         label=f'Correlated rho={correlation} (empirical)',
         color=non_default_color,
         marker='x',
         linewidth=2,
         markersize=7)
plt.title('Probability of Exactly k Defaults\n'
          'Independent (Binomial) vs Correlated (Simulation)',
          fontsize=12, fontweight='bold')
plt.xlabel('Number of Defaults k')
plt.ylabel('Probability P(K = k)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()