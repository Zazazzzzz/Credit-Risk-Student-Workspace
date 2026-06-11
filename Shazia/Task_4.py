# ---------------------------------------------------------------
# Script Name: Task 4 - Independent vs Correlated Defaults
# Author: Shazia Ishaq
# Description: Comparing what happens when 1000 companies default
#              independently vs when they are all affected by the
#              same economic factor (like a recession)
#
# Think of it like this:
# INDEPENDENT: each company flips its own coin to decide default
# CORRELATED:  all companies are affected by the SAME bad weather
#              (systematic factor) PLUS their own individual luck
#
# Parameters:
#   N    = 2000   number of simulations
#   m    = 1000   number of companies in portfolio
#   p    = 1%     default probability per company
#   rho  = 0.25   sensitivity to the common economic factor
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────
num_trials = 2000       # how many times we repeat the experiment
num_obligors = 1000     # number of companies in our portfolio
prob_default = 0.01     # each company has 1% chance of default
correlation = 0.25      # how much companies share common risk

non_default_color = 'silver'
default_color = 'thistle'

np.random.seed(42)


# ── The Core Function ─────────────────────────────────────────
# This function simulates how many companies default in each trial
# When correlation = 0 → independent defaults
# When correlation > 0 → correlated defaults (shared factor)

def simulate_correlated_defaults(num_trials, num_obligors,
                                  prob_default, correlation):
    # Step A: Generate the common economic factor F
    # F represents the state of the economy in each simulation
    # F > 0 means good economy, F < 0 means bad economy
    # Shape: (2000,) one value per simulation trial
    F = np.random.normal(0, 1, num_trials)

    # Step B: Generate company-specific shocks (epsilon)
    # Each company has its own individual risk on top of economy
    # Shape: (2000, 1000) one value per company per trial
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors))

    # Step C: Compute creditworthiness variable X for each company
    # X = rho * F + sqrt(1 - rho^2) * epsilon
    # rho * F        → how much economy affects each company
    # sqrt(1-rho^2)  → how much individual risk remains
    # When rho=0: X = epsilon (purely individual, independent)
    # When rho=1: X = F (all companies move together perfectly)
    #
    # F[:, np.newaxis] reshapes F from (2000,) to (2000,1)
    # so it can be added to epsilon of shape (2000,1000)
    # This is called broadcasting in numpy
    X = (correlation * F[:, np.newaxis]
         + np.sqrt(1 - correlation**2) * epsilon)

    # Step D: Calculate the default threshold
    # A company defaults if X falls below this threshold
    # norm.ppf(0.01) gives the value below which 1% of the
    # standard normal distribution lies
    threshold = norm.ppf(prob_default)

    # Step E: Count defaults in each trial
    # X < threshold gives True/False for each company
    # .sum(axis=1) counts how many companies defaulted
    defaults = (X < threshold).sum(axis=1)

    return defaults


# ── Run Both Simulations ──────────────────────────────────────

# Scenario 1: Independent defaults (correlation = 0)
# Each company only has its own individual risk
# No common economic factor affects them
independent_defaults = simulate_correlated_defaults(
    num_trials, num_obligors, prob_default, 0
)

# Scenario 2: Correlated defaults (correlation = 0.25)
# All companies are partly affected by the same economy
# Bad economy → many defaults at once
correlated_defaults = simulate_correlated_defaults(
    num_trials, num_obligors, prob_default, correlation
)

# ── Print Summary Statistics ──────────────────────────────────
print("=" * 55)
print("   TASK 4: INDEPENDENT vs CORRELATED DEFAULTS")
print("=" * 55)
print(f"Portfolio: {num_obligors} companies, PD={prob_default*100}%")
print(f"Simulations: {num_trials}")
print("-" * 55)
print(f"{'Metric':<30} {'Independent':>12} {'Correlated':>12}")
print("-" * 55)
print(f"{'Mean defaults':<30} "
      f"{independent_defaults.mean():>12.2f} "
      f"{correlated_defaults.mean():>12.2f}")
print(f"{'Std of defaults':<30} "
      f"{independent_defaults.std():>12.2f} "
      f"{correlated_defaults.std():>12.2f}")
print(f"{'Max defaults in one trial':<30} "
      f"{independent_defaults.max():>12} "
      f"{correlated_defaults.max():>12}")
print(f"{'Min defaults in one trial':<30} "
      f"{independent_defaults.min():>12} "
      f"{correlated_defaults.min():>12}")
print("=" * 55)

print("\nKey Observation:")
print(f"Both scenarios have similar MEAN defaults")
print(f"({independent_defaults.mean():.1f} vs "
      f"{correlated_defaults.mean():.1f})")
print(f"But correlated defaults have much higher STD")
print(f"({independent_defaults.std():.2f} vs "
      f"{correlated_defaults.std():.2f})")
print(f"This means extreme losses happen much more often")
print(f"when companies are correlated!")

# ── Plot Side by Side Histograms ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Independent defaults
# Should look like a tight bell curve around the mean
ax1.hist(independent_defaults,
         bins=range(0, 31),
         density=True,
         color=default_color,
         edgecolor=non_default_color,
         label='Independent')
ax1.set_title('Independent Defaults\n'
              '(no common factor, each company on its own)')
ax1.set_xlabel('Number of Defaults')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--',
         color='gray', linewidth=0.5)

# Right plot: Correlated defaults
# Should look wider and flatter with a longer tail
# This shows that extreme losses are much more likely!
ax2.hist(correlated_defaults,
         bins=range(0, 31),
         density=True,
         color=non_default_color,
         edgecolor=default_color,
         label=f'Correlated Defaults (rho={correlation})')
ax2.set_title('Correlated Defaults\n'
              '(common economic factor affects all companies)')
ax2.set_xlabel('Number of Defaults')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, axis='y', linestyle='--',
         color='gray', linewidth=0.5)

# Use same y-axis scale for fair comparison
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

fig.suptitle('Default Distribution: Independent vs Correlated\n'
             f'{num_obligors} obligors, PD={prob_default*100}%, '
             f'{num_trials} simulations',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()