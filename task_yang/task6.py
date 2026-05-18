"""
Simulate the default behavior of a homogeneous portfolio of
1000 obligors, each with an unconditional default probability of
1%. We consider dependent defaults with default correlation
0.5%. Dependence is introduced through a common random
default probability:
P ∼ Beta(𝛼, 𝛽),
Yi|P ∼ Bernoulli(P).
  (Given P, if Yi=1 then default, else no default)
Simulate this model 2000 times and visualize the distribution
of the number of defaults using a histogram.
Parameters:
N = 2000, np.random.seed(42).
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_sims = 2000
m = 1000

pi = 0.01
# unconditional default prob
rho_y = 0.005
# default correlation rho_y = 1/(𝛼+𝛽+1)
# kappa = 𝛼+𝛽 --- rho_y = 1/(kappa+1) --- kappa = 1/rho_y -1

# Bate distribution
kappa = 1/ rho_y -1
alpha = pi * kappa
beta = (1-pi) * kappa

print(f"kappa = {kappa:.2f}")
print(f"alpha = {alpha:.4f}")
print(f"beta = {beta:.4f}")


# Simulate common random default prob P
P = np.random.beta(
    a=alpha,
    b=beta,
    size=n_sims
)


# Conditional on P, simulate defaults Yi|P ∼ Bernoulli(P).
P_matrix = P.reshape(-1, 1)
# -1: count the number of lines automatically
# 1: reshape the P from one line to one column so that each obligor has one P

defaults = np.random.binomial(
    n=1,
    # n=1 because each company has only one default trial in each simulation:
    # it either defaults (1) or does not default (0).
    p=P_matrix,
    size=(n_sims, m)
)

default_counts = defaults.sum(axis=1)
print("Average number of defaults:")
print(default_counts.mean())

print("Standard deviation of defaults:")
print(default_counts.std())

print("95th percentile:")
print(np.percentile(default_counts, 95))

plt.figure(figsize=(10, 5))

plt.hist(
    default_counts,
    bins=40,
    edgecolor="black",
    label="Simulated defaults"
)

mean_defaults = default_counts.mean()
plt.axvline(
    mean_defaults,
    linestyle="--",
    linewidth=2,
    label=f"Mean defaults = {mean_defaults:.2f}"
)

plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.title("Beta-Bernoulli Default Simulation")
plt.legend()
plt.grid(True)

plt.show()

