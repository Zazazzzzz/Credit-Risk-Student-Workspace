import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------
# Step 1: Set the distribution of returns
# --------------------------------------------------
# mu: Expected return, representing the center of the distribution.
# sigma: Volatility, representing the risk or uncertainty.
mu = 0.5
sigma = 2.0

# --------------------------------------------------
# Step 2: Choose a threshold on the original return scale
# --------------------------------------------------
# d: The critical threshold. In credit risk, this can represent a "Default Point".
d = -1.0

# --------------------------------------------------
# Step 3: Standardize the threshold
# --------------------------------------------------
# Standardization formula: Z = (X - mu) / sigma
# Subtracting mu shifts the center to 0; dividing by sigma scales the units.
z_d = (d - mu) / sigma

# --------------------------------------------------
# Step 4: Build the original density (X)
# --------------------------------------------------
# Create a grid of x-values around the mean
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf_x = norm.pdf(x, loc=mu, scale=sigma)

# --------------------------------------------------
# Step 5: Build the standardized density (Z)
# --------------------------------------------------
# Create a grid for the standard normal variable Z ~ N(0, 1)
z = np.linspace(-4, 4, 1000)
pdf_z = norm.pdf(z, loc=0, scale=1)

# --------------------------------------------------
# Step 6: Compute probabilities (Left-tail)
# --------------------------------------------------
# Calculate the probability of returns being below the threshold (P(X <= d))
p_left_original = norm.cdf(d, loc=mu, scale=sigma)
p_left_standardized = norm.cdf(z_d, loc=0, scale=1)

# --------------------------------------------------
# Step 7 & 8: Visualization
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Original Distribution X ~ N(mu, sigma^2)
ax1.plot(x, pdf_x, label=f'Original: N({mu}, {sigma}²)', color='blue', lw=2)
ax1.axvline(d, color='red', linestyle='--', label=f'Threshold d = {d}')
ax1.axvline(mu, color='gray', linestyle=':', label=f'Mean mu = {mu}')
# Shade the risky region (Left tail)
x_fill = np.linspace(mu - 4*sigma, d, 100)
ax1.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='red', alpha=0.3, label='Risk Area')
ax1.set_title("Original Distribution of Returns (X)")
ax1.set_xlabel("Return Value")
ax1.set_ylabel("Probability Density")
ax1.legend()

# Plot 2: Standardized Distribution Z ~ N(0, 1)
ax2.plot(z, pdf_z, label='Standardized: N(0, 1)', color='green', lw=2)
ax2.axvline(z_d, color='red', linestyle='--', label=f'Std. Threshold z_d = {z_d:.2f}')
ax2.axvline(0, color='gray', linestyle=':', label='Mean = 0')
# Shade the risky region (Left tail)
z_fill = np.linspace(-4, z_d, 100)
ax2.fill_between(z_fill, norm.pdf(z_fill, 0, 1), color='red', alpha=0.3, label='Risk Area (Z)')
ax2.set_title("Standardized Distribution (Z)")
ax2.set_xlabel("Standard Deviations from Mean")
ax2.set_ylabel("Probability Density")
ax2.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 9: Interpret the result
# --------------------------------------------------
print(f"--- Results Analysis ---")
print(f"Original Threshold (d):           {d}")
print(f"Standardized Threshold (z_d):     {z_d:.4f}")
print(f"P(X <= d) Original Prob:          {p_left_original:.4%}")
print(f"P(Z <= z_d) Standardized Prob:    {p_left_standardized:.4%}")
print(f"\nConclusion: Both probabilities are identical!")
print(f"Standardization does not change the risk level; it only changes the scale of measurement.")