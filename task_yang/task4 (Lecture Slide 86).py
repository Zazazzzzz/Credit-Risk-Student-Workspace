"""
Simulating the default behavior of a portfolio of 1000 obligors,
each with a defult prob of 1%.
Two scenarios are considered.
- indeoendent defaults
- correlated defaults,
  where defaults are influenced by a systematic factor with rho = 0.25

Simulate both scenarios for 2000 times and visualize the result through histograms.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

N = 2000 # simulations
p = 0.01
m = 1000 # number of obligors
rho = 0.25

d = norm.ppf(p)
# find the quantile of p
# if Xi < di, then the company defaults

# 1. Simulate independent default indicators for 1000 companies for 2000 times
# Shape: N=2000 rows, m=1000columns
independent_defaults = np.random.binomial(
    n=1, # Yi follows Bernoulli distribution
    p=p,
    size=(N, m)
)

independent_default_counts = independent_defaults.sum(axis=1)
# count the default number across all companies for each simulation


# 2. Simulate the companies with correlated defaults rho = 0.25
F = np.random.normal(0,1, size=(N,1))
# generate one systemic factor for all companies for each simulation

epsilon = np.random.normal(0, 1, size=(N, m))
# shock for each firm in each simulation

X = rho * F + np.sqrt(1 - rho**2) * epsilon
# latent variable factor models (Slide 91)

correlated_defaults = (X <= d).astype(int)
correlated_default_counts = correlated_defaults.sum(axis=1)

print("Independent defaults:")
print(f"Mean number of defaults: {independent_default_counts.mean():.2f}")
print(f"Standard deviation: {independent_default_counts.std():.2f}")
print(f"95th percentile: {np.percentile(independent_default_counts, 95):.2f}")

print("\nCorrelated defaults:")
print(f"Mean number of defaults: {correlated_default_counts.mean():.2f}")
print(f"Standard deviation: {correlated_default_counts.std():.2f}")
print(f"95th percentile: {np.percentile(correlated_default_counts, 95):.2f}")

plt.figure(figsize=(10, 5))

plt.hist(independent_default_counts, bins=30, alpha=0.6, label="Independent defaults")
plt.hist(correlated_default_counts, bins=30, alpha=0.6, label="Correlated defaults")

plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.title("Default Count Distribution: Independent vs Correlated Defaults")
plt.legend()
plt.grid(True)
plt.show()
