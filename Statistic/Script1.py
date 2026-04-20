import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --- simulate Bernoulli data ---
np.random.seed(42)
Y = np.random.binomial(n=1, p=0.7, size=1000)

# --- 1. Count (Histogram) ---
plt.figure()
plt.hist(Y, bins=[-0.5, 0.5, 1.5])
plt.title("Count (Histogram)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# --- 2 Density ---
plt.figure()
plt.hist(Y, bins=[-0.5, 0.5, 1.5], density=True)
plt.title("Density")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# --- 3. Cumulative Density (CDF) ---
values = np.sort(Y)
cdf = np.arange(1, len(values)+1) / len(values)

plt.figure()
plt.step(values, cdf)
plt.title("Cumulative Density (CDF)")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
plt.show()