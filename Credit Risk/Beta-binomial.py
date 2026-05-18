# ---------------------------------------------------------------
# Script Name: Default Number Distribution under Dependency Structures
# Author: Hongyi Shen
#
# Description:
# This script compares the distribution of the number of defaults
# in a homogeneous credit portfolio under two dependency assumptions.
#
# First, defaults are modeled as independent Bernoulli random variables
# with the same unconditional default probability.
#
# Second, defaults are modeled using a beta-Bernoulli mixture model,
# where a common random default probability is drawn from a beta
# distribution. Conditional on this probability, defaults are independent;
# unconditionally, they are dependent.
#
# The script produces simulated histograms, theoretical probability
# mass functions, and summary statistics for both cases.
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, betabinom


# --------------------------------------------------
# Parameters
# --------------------------------------------------
np.random.seed(42)

N = 2000          # number of simulations
m = 1000          # number of obligors
p = 0.01          # unconditional default probability
rho_y = 0.005     # default correlation in the dependent case

# Colors
color_d = np.array([210, 150, 150]) / 255   # dependent color
color_i = "silver"                          # independent color

# --------------------------------------------------
# Beta-Bernoulli parameter calibration
# --------------------------------------------------
# We assume:
#
#   P ~ Beta(alpha, beta)
#   Y_i | P ~ Bernoulli(P)
#
# where P is the common random default probability.
#
# The goal is to choose alpha and beta such that:
#
#   1. The unconditional default probability is p:
#
#          E[Y_i] = E[E[Y_i | P]]
#                 = E[P]
#                 = p
#
#      For a beta distribution:
#
#          E[P] = alpha / (alpha + beta)
#
#      Hence:
#
#          alpha / (alpha + beta) = p
#
#
#   2. The unconditional default correlation is rho_y:
#
#          Corr(Y_i, Y_j) = rho_y,   for i != j
#
#      In the beta-Bernoulli mixture model, dependence comes from
#      the common random default probability P.
#
#      Conditional on P, defaults are independent. Therefore,
#
#          Cov(Y_i, Y_j)
#          = Cov(E[Y_i | P], E[Y_j | P])
#          = Var(P)
#
#      Also:
#
#          Var(Y_i) = p(1 - p)
#
#      Hence:
#
#          Corr(Y_i, Y_j)
#          = Var(P) / [p(1 - p)]
#
#      For P ~ Beta(alpha, beta), let
#
#          s = alpha + beta
#
#      Then:
#
#          Var(P)
#          = alpha beta / [(alpha + beta)^2 (alpha + beta + 1)]
#          = p(1 - p) / (s + 1)
#
#      Therefore:
#
#          Corr(Y_i, Y_j)
#          = [p(1 - p) / (s + 1)] / [p(1 - p)]
#          = 1 / (s + 1)
#
#      Setting this equal to rho_y gives:
#
#          rho_y = 1 / (s + 1)
#
#      Solving for s:
#
#          s + 1 = 1 / rho_y
#          s = 1 / rho_y - 1
#
#      Since alpha / (alpha + beta) = p and s = alpha + beta:
#
#          alpha = p * s
#          beta  = (1 - p) * s

s = 1 / rho_y - 1

alpha = p * s
beta = (1 - p) * s

print("Dependency structure: Beta-Bernoulli mixture")
print(f"Unconditional default probability p = {p:.2%}")
print(f"Default correlation rho_y = {rho_y:.2%}")
print(f"alpha = {alpha:.4f}")
print(f"beta  = {beta:.4f}")

# --------------------------------------------------
# Case 1: Independent defaults
# --------------------------------------------------
# Y_i ~ Bernoulli(p)
# D = sum_i Y_i ~ Binomial(m, p)

defaults_ind = np.random.binomial(n=m, p=p, size=N)

# --------------------------------------------------
# Case 2: Dependent defaults
# --------------------------------------------------
# Step 1: simulate the systematic random default probability
#
# p_random has shape (N,).
#
# Each entry p_random[j] is one realization of the portfolio-level
# default probability in simulation trial j.
#
# Important:
# In a given simulation trial j, the same p_random[j] is used for
# all m obligors. This is what creates dependence across obligors.
p_random = np.random.beta(alpha, beta, size=N)

# Step 2: conditional on P, simulate portfolio defaults
defaults_dep = np.random.binomial(n=m, p=p_random)

# --------------------------------------------------
# Plot 1: Simulated histogram
# --------------------------------------------------
max_k_plot = 150
bins = np.arange(0, max_k_plot + 2) - 0.5

plt.figure(figsize=(10, 6))

plt.hist(
    defaults_dep,
    bins=bins,
    color=color_d,
    density=True,
    alpha=0.9,
    label="Dependent defaults: Beta-Bernoulli structure"
)

plt.hist(
    defaults_ind,
    bins=bins,
    color=color_i,
    density=True,
    alpha=0.5,
    label="Independent defaults"
)

plt.xlabel("Number of Defaults")
plt.ylabel("Density")
plt.title("Simulated Default Number Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, max_k_plot)
plt.show()

# --------------------------------------------------
# Plot 2: PMF under each dependency structure
# --------------------------------------------------
k = np.arange(0, m + 1)

# Independent case:
# D ~ Binomial(m, p)
pmf_ind = binom.pmf(k, n=m, p=p)

# Dependent beta-Bernoulli case:
# D ~ Beta-Binomial(m, alpha, beta)
pmf_dep = betabinom.pmf(k, n=m, a=alpha, b=beta)

plt.figure(figsize=(10, 6))

plt.plot(
    k[:max_k_plot + 1],
    pmf_dep[:max_k_plot + 1],
    color=color_d,
    linewidth=2,
    label="Dependent structure: Beta-Binomial PMF"
)

plt.plot(
    k[:max_k_plot + 1],
    pmf_ind[:max_k_plot + 1],
    color=color_i,
    linestyle="--",
    linewidth=2,
    label="Independent structure: Binomial PMF"
)

plt.xlabel("Number of Defaults")
plt.ylabel("Probability")
plt.title("PMF of Default Numbers under Different Dependency Structures")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 60)
plt.show()

# --------------------------------------------------
# Summary statistics
# --------------------------------------------------
mean_ind = m * p
var_ind = m * p * (1 - p)

mean_dep = m * p
var_dep = m * p * (1 - p) * (1 + (m - 1) * rho_y)

print("\nIndependent default structure")
print(f"Simulated mean: {np.mean(defaults_ind):.2f}")
print(f"Simulated standard deviation: {np.std(defaults_ind):.2f}")
print(f"Theoretical mean: {mean_ind:.2f}")
print(f"Theoretical standard deviation: {np.sqrt(var_ind):.2f}")

print("\nDependent default structure: Beta-Bernoulli mixture")
print(f"Simulated mean: {np.mean(defaults_dep):.2f}")
print(f"Simulated standard deviation: {np.std(defaults_dep):.2f}")
print(f"Theoretical mean: {mean_dep:.2f}")
print(f"Theoretical standard deviation: {np.sqrt(var_dep):.2f}")