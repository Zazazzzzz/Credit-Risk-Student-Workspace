import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V0 = 100
mu_V = 0.05
sigma_V = 0.2
T = 1.0
B = 90
N = 1000
dt = T / N

np.random.seed(42)

W = np.random.normal(loc=0, scale=np.sqrt(dt), size= N).cumsum()
W = np.insert(W, 0, 0)

time_grid = np.linspace(0, T, N+1)
VT = V0 * np.exp((mu_V - 0.5 * sigma_V**2 ) * time_grid + sigma_V * W)

plt.xlabel("Time")
plt.ylabel("$V_T$")
plt.plot(time_grid,VT,label="Path of $V_T$")
plt.title("Simulated Path of $V_T$")
plt.axhline(y=90, color="Red", linestyle="--",label="Default Threshold B=90")
plt.legend()
plt.legend()
plt.legend()
plt.savefig("mertontask1.png", dpi=300, bbox_inches="tight")
plt.show()









