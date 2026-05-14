# ---------------------------------------------------------------
# Script Name: Default Number Distributions under Independent and
#              Correlated Defaults
# Author: Hongyi Shen
#
# Description:
# This script studies the distribution of the number of defaults
# in a homogeneous credit portfolio under two settings:
#
# 1. Independent defaults:
#    Each obligor defaults independently with the same default
#    probability.
#
# 2. Correlated defaults:
#    Defaults are generated under a Gaussian one-factor model,
#    where each obligor's latent credit variable depends on a
#    common systematic factor and an idiosyncratic component.
#
# The script first simulates default counts and visualizes their
# distributions using histograms. It then compares the probability
# of observing different numbers of defaults:
# - theoretically for the independent case using the binomial PMF,
# - empirically for the correlated case using simulation output.
#
# The purpose is to illustrate how dependence between obligors
# changes the shape of the default distribution and increases the
# likelihood of extreme default outcomes.
# ---------------------------------------------------------------

# simulation of loss distributions of independent and dependent defaults in a homogeneous portfolio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

non_default_color = 'silver'
default_color = 'thistle'

# Parameters
num_trials = 2000
num_obligors = 1000
prob_default = 0.01
correlation = 0.25

# Function to simulate defaults with correlation
def simulate_correlated_defaults(num_trials, num_obligors, prob_default, correlation):
    # Generate common factor F
    F = np.random.normal(0, 1, num_trials) # Shape: (2000,)
    # Generate idiosyncratic factors epsilon_i
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors)) # Shape: (2000,1000)
    # Compute latent variables X_i
    X = correlation * F[:, np.newaxis] + np.sqrt(1 - correlation ** 2) * epsilon
    # F[:, np.newaxis] transforms F into a 2D column vector of shape (n, 1).
    # np.newaxis is used to add a new dimension, essentially reshaping F for broadcasting.

    # F[:, np.newaxis] reshapes F from (2000,) to (2000, 1)
    # Now it's a column vector.
    # F[:, np.newaxis] has shape (2000, 1)
    # epsilon has shape (2000, 1000)
    # Through broadcasting:
    # The common factor F is copied across all 1000 obligors in each of the 2000 trials.
    # Each entry in X gets a contribution from the common factor and its own idiosyncratic noise.
    # So X ends up with shape: (2000, 1000)

    # Broadcasting is the behind-the-scenes mechanism that makes the shapes align so that * can be applied.
    # the * operator itself is element-wise multiplication
    # If two arrays are of shape:
    # (A, 1) and (A, B)
    # Then NumPy will broadcast the (A, 1) array across the columns to match (A, B)

    # Determine defaults
    threshold = norm.ppf(prob_default)
    defaults = (X < threshold).sum(axis=1)
    # X < threshold	(2000, 1000) Boolean default indicator
    # defaults	(2000,)	Number of defaults per trial

    return defaults

np.random.seed(42)
# Simulate independent defaults
independent_defaults = simulate_correlated_defaults(num_trials, num_obligors, prob_default, 0)

# Simulate correlated defaults
correlated_defaults = simulate_correlated_defaults(num_trials, num_obligors, prob_default, correlation)

# Plot the density of the number of defaults on the same plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# plt.subplots(nrows, ncols): Creates a figure with a grid of subplots.
# nrows=1: There is one row.
# ncols=2: There are two columns → two subplots placed side by side.
# fig: Represents the overall figure (the entire plotting area).
# (ax1, ax2): These are two subplot axes objects:
# ax1: The first subplot (left).
# ax2: The second subplot (right).

# Plot independent defaults
ax1.hist(independent_defaults, bins=range(0, 31), density=True, color=default_color, edgecolor=non_default_color,
         label='Independent')
ax1.set_title('Independent')
ax1.set_xlabel('Number of Defaults')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.5)

# Plot correlated defaults
ax2.hist(correlated_defaults, bins=range(0, 31), density=True, color=non_default_color, edgecolor=default_color,
         label='Asset Correlation(0.25)')
ax2.set_title('Asset Correlation(0.25)')
ax2.set_xlabel('Number of Defaults')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.5)

# Set identical y-axis limits for both subplots
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])  # Get the maximum y-limit from both subplots
ax1.set_ylim(0, y_max) # can also use ax1.set_yticks(range(0, y_max, gap)) to set physical space (gap) between tick mark
ax2.set_ylim(0, y_max)

# Add labels and title
plt.tight_layout()
plt.show()


# Probabilities of Number of Defaults
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# Parameters
num_trials = 2000
num_obligors = 1000
prob_default = 0.01
correlation = 0.25

# Function to simulate correlated defaults
def simulate_correlated_defaults(num_trials, num_obligors, prob_default, correlation):
    # Generate common factor F
    F = np.random.normal(0, 1, num_trials)
    # Generate idiosyncratic factors epsilon_i
    epsilon = np.random.normal(0, 1, (num_trials, num_obligors))
    # Compute latent variables X_i
    X = correlation * F[:, np.newaxis] + np.sqrt(1 - correlation ** 2) * epsilon
    # Determine defaults
    threshold = norm.ppf(prob_default)
    defaults = (X < threshold).sum(axis=1)
    return defaults

np.random.seed(42)

# Simulate correlated defaults
correlated_defaults = simulate_correlated_defaults(num_trials, num_obligors, prob_default, correlation)

# Compute probabilities for independent defaults
max_defaults = 30  # Maximum number of defaults to consider
independent_probs = [binom.pmf(k, num_obligors, prob_default) for k in range(max_defaults + 1)]

# Compute probabilities for correlated defaults (empirical)
# Compute the empirical probabilities by counting the frequency of each number of defaults across all trials
# np.mean(correlated_defaults == 3): P(X = 3) \approx \frac{1}{2000} \sum_{i=1}^{2000} \mathbf{1}(X_i = 3)
correlated_probs = np.zeros(max_defaults + 1)
for k in range(max_defaults + 1):
    correlated_probs[k] = np.mean(correlated_defaults == k)


# Plot the probabilities
plt.figure(figsize=(12, 6))
plt.plot(range(max_defaults + 1), independent_probs, label='Independent Defaults', color=default_color, marker='o')
plt.plot(range(max_defaults + 1), correlated_probs, label='Correlated Defaults (0.25)', color=non_default_color, marker='x')
plt.title('Probability of Number of Defaults')
plt.xlabel('Number of Defaults')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
