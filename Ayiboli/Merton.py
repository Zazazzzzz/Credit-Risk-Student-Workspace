#Simulation of Merton========================================

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
W = np.random.normal(0, np.sqrt(dt), N).cumsum()

# Simulate the path of V_T
time_grid = np.linspace(0, T, N)
VT = V0 * np.exp((mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W)

# Check default
if VT[-1] < B:
    print("Default")
else:
    print("No default")

print("Final asset value:", VT[-1])

# Plot
plt.plot(time_grid, VT, label="Firm value")
plt.axhline(B, color="red", linestyle="--", label="Default threshold")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Merton Model Simulation")
plt.legend()
plt.show()
