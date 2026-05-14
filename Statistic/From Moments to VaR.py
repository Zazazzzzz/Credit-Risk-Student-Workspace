# ---------------------------------------------------------------
# Script Name: From Moments to VaR
# Author: Hongyi Shen
#
# Description:
# This script demonstrates how we move from basic statistical
# summaries (mean, variance, correlation) to full distributional
# analysis (PDF, CDF), and finally to risk measures such as VaR.
#
# It provides both numerical output and visualizations to show
# how these concepts are connected in risk modeling.
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# --------------------------------------------------
# Step 1: Generate data (two correlated variables)
# --------------------------------------------------
n = 1000
mu = [0, 0]
cov = [[1, 0.6],
       [0.6, 1]]  # covariance matrix (controls correlation)

data = np.random.multivariate_normal(mu, cov, size=n)
X = data[:, 0]
Y = data[:, 1]

# --------------------------------------------------
# Step 2: Mean, Variance, Correlation
# --------------------------------------------------
mean_X = np.mean(X)
var_X = np.var(X)
corr_XY = np.corrcoef(X, Y)[0, 1]

print("Mean (X):", round(mean_X, 3))
print("Variance (X):", round(var_X, 3))
print("Correlation (X,Y):", round(corr_XY, 3))

# plot
n = 1000

# --------------------------------------------------
# Case 1: Correlation = 0 (independent)
# --------------------------------------------------
mean = [0, 0]
cov_uncorr = [[1, 0],
              [0, 1]]

data_uncorr = np.random.multivariate_normal(mean, cov_uncorr, size=n)
x_uncorr = data_uncorr[:, 0]
y_uncorr = data_uncorr[:, 1]

# --------------------------------------------------
# Case 2: Correlation = 0.5
# --------------------------------------------------
rho = 0.5
cov_corr = [[1, rho],
            [rho, 1]]

data_corr = np.random.multivariate_normal(mean, cov_corr, size=n)
x_corr = data_corr[:, 0]
y_corr = data_corr[:, 1]

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Uncorrelated
axes[0].scatter(x_uncorr, y_uncorr, alpha=0.5)
axes[0].set_title("Correlation = 0")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid(alpha=0.3)

# Correlated
axes[1].scatter(x_corr, y_corr, alpha=0.5)
axes[1].set_title("Correlation = 0.5")
axes[1].set_xlabel("X")
axes[1].grid(alpha=0.3)

plt.suptitle("Mean = 0, Variance = 1: Effect of Correlation")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------
# Step 3: PDF (theoretical normal)
# --------------------------------------------------
x_grid = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x_grid, loc=0, scale=1)

# --------------------------------------------------
# Step 4: CDF
# --------------------------------------------------
cdf = norm.cdf(x_grid, loc=0, scale=1)

# --------------------------------------------------
# Step 5: Quantile / VaR
# --------------------------------------------------
alpha = 0.05
var_alpha = norm.ppf(alpha)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# --- Histogram (data) ---
axes[0].hist(X, bins=30, density=True, alpha=0.6, label="Simulated data")
axes[0].plot(x_grid, pdf, label="Normal PDF")
axes[0].set_title("Data vs PDF")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# --- CDF ---
axes[1].plot(x_grid, cdf)
axes[1].axhline(alpha, linestyle=":", label=f"{alpha}")
axes[1].axvline(var_alpha, linestyle="--", label=f"VaR ≈ {var_alpha:.2f}")
axes[1].set_title("CDF")
axes[1].set_xlabel("x")
axes[1].set_ylabel("F(x)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

# --- PDF + VaR ---
axes[2].plot(x_grid, pdf)
axes[2].fill_between(
    x_grid[x_grid <= var_alpha],
    pdf[x_grid <= var_alpha],
    alpha=0.3
)
axes[2].axvline(var_alpha, linestyle="--", label=f"VaR ≈ {var_alpha:.2f}")
axes[2].set_title("PDF and VaR")
axes[2].set_xlabel("x")
axes[2].set_ylabel("Density")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle("From Moments to Distribution to Risk Measure")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()