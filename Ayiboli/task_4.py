import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
np.random.seed(42)

N = 2000  # number of simulations
m = 1000  # number of obligors
p = 0.01  # default probability
rho = 0.25  # asset correlation

# =========================
# 1. Independent defaults
# =========================
# Each obligor defaults independently with probability p
independent_defaults = np.random.binomial(n=m, p=p, size=N)

# =========================
# 2. Correlated defaults
# =========================
# One-factor model:
# X_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
# Default occurs if X_i < default threshold

threshold = norm.ppf(p)

correlated_defaults = []

for _ in range(N):
    Z = np.random.normal()  # systemic factor
    eps = np.random.normal(size=m)  # idiosyncratic factors

    X = np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps

    defaults = np.sum(X < threshold)
    correlated_defaults.append(defaults)

correlated_defaults = np.array(correlated_defaults)

# =========================
# 3. Plot histograms
# =========================
plt.figure(figsize=(10, 5))
plt.hist(independent_defaults, bins=30, alpha=0.7, edgecolor="black")
plt.title("Independent Defaults")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(correlated_defaults, bins=30, alpha=0.7, edgecolor="black")
plt.title("Correlated Defaults, rho = 0.25")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.show()

# =========================
# 4. Summary statistics
# =========================
print("Independent Defaults")
print("Mean:", np.mean(independent_defaults))
print("Std:", np.std(independent_defaults))
print("99% quantile:", np.quantile(independent_defaults, 0.99))

print("\nCorrelated Defaults")
print("Mean:", np.mean(correlated_defaults))
print("Std:", np.std(correlated_defaults))
print("99% quantile:", np.quantile(correlated_defaults, 0.99))