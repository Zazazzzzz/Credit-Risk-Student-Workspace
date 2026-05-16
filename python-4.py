import matplotlib.pyplot as plt
import numpy as np
# Setup parameters
np.random.seed(42)
N = 2000  # simulations
m = 1000  # obligors
p = 0.01  # default rate
rho = 0.25  # correlation
threshold = -2.326  # cutoff for 1%
# Scenario 1: Independent
ind_defaults = np.random.binomial(n=m, p=p, size=N)
# Scenario 2: Correlated
Z = np.random.normal(size=(N, 1))
epsilon = np.random.normal(size=(N, m))
assets = np.sqrt(rho) * Z + np.sqrt(1 - rho) * epsilon
corr_defaults = np.sum(assets < threshold, axis=1)
custom_bins = np.arange(0, 32) - 0.5
# Plotting to match the image style exactly
plt.figure(figsize=(12, 5))
# Left plot: Independent
plt.subplot(1, 2, 1)
plt.hist(
    ind_defaults,
    bins=custom_bins,
    density=True,
    color="#d38d8d",
    edgecolor="gray",
    alpha=0.85,
    label="Independent",
)
plt.title("Independent", fontsize=10)
plt.xlabel("Number of Defaults", fontsize=9)
plt.ylabel("Density", fontsize=9)
plt.xlim(0, 30)
plt.ylim(0, 0.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=8)
# Right plot:
plt.subplot(1, 2, 2)
plt.hist(
    corr_defaults,
    bins=custom_bins,
    density=True,
    color="#8c8c8c",
    edgecolor="gray",
    alpha=0.85,
    label="Asset Correlation(0.25)",
)
plt.title("Asset Correlation(0.25)", fontsize=10)
plt.xlabel("Number of Defaults", fontsize=9)
plt.ylabel("Density", fontsize=9)
plt.xlim(0, 30)
plt.ylim(0, 0.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()