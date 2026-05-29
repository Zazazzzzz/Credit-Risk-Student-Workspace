# ---------------------------------------------------------------
# Script Name: Merton's Model
# Author: Hongyi Shen
# Description: Simulation of Merton's Model
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
V0 = 100  # Initial value
mu_V = 0.05  # Drift (expected return)
sigma_V = 0.2  # Volatility
T = 1.0  # Time to maturity
B = 90  # Default threshold
N = 1000  # Number of steps in the path
dt = T / N  # Time step

# For reproducibility
np.random.seed(42)

# Generate a Brownian motion
W = np.random.normal(0, np.sqrt(dt), N).cumsum()  # Standard Brownian motion
# Mean 0 (no drift for standard Brownian motion)
# Standard deviation np.sqrt(dt)
# N: N independent random samples from a normal distribution
# cumsum(), this computes the cumulative sum of the delta W_i, if i=k, then W_k = delta W_1 + ... + delta W_k

# Simulate the path of V_T
time_grid = np.linspace(0, T, N) # np.linspace is a NumPy function that creates an array of evenly spaced numbers.
# range() produces integer values.
# np.linspace() generates floating-point numbers, which are often needed for simulations involving time or continuous processes.
VT = V0 * np.exp((mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W)

# Plot the path
plt.figure(figsize=(10, 6))
plt.plot(time_grid, VT, label="Path of $V_T$")
plt.axhline(y=B, color='r', linestyle='--', label=f"Default Threshold B={B}")
plt.title("Simulated Path of $V_T$")
plt.xlabel("Time")
plt.ylabel("$V_T$")
plt.legend()
plt.grid()
plt.show()





