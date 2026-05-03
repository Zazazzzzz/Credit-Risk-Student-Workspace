import numpy as np
import matplotlib.pyplot as plt

V_0 = 100
mu_V = 0.05
sigma_V = 0.2
T = 1.0
B = 90
N = 1000
dt = T / N
np.random.seed(42)

W_increments = np.random.normal(0, np.sqrt(dt), N)
W = np.insert(np.cumsum(W_increments), 0, 0)

time_grid = np.linspace(0, T, N+1)

VT = V_0 * np.exp((mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W)
default = VT[-1] < B
print("Default:", default)

plt.plot(time_grid, VT)
plt.axhline(y=B, color='r', linestyle='--', label='Threshold B')
plt.legend()
plt.title("Merton Model Simulation")
plt.xlabel("Time")
plt.ylabel("Asset Value")

plt.show()












