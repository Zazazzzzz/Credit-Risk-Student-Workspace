# ---------------------------------------------------------------
# Script Name: Task 1 - Merton Model
# Author: Shazia Ishaq
# Description: Simulate default path under Merton's model
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
V0 = 100
mu_V = 0.05
sigma_V = 0.2
T = 1.0
B = 90
N = 1000
dt = T / N

# for reproducibility
np.random.seed(42)

# ── Step 1: Simulate Brownian Motion ──────────────────────────
# W_t is a standard Brownian motion (Wiener process)
# Each increment dW ~ N(0, sqrt(dt))
# cumsum() builds the cumulative path W_t from increments
W = np.random.normal(0, np.sqrt(dt), N).cumsum()

# ── Step 2: Simulate Asset Value Path ─────────────────────────
# Using the exact GBM solution (Ito's lemma):
# V_t = V_0 * exp((mu_V - 0.5*sigma_V^2)*t + sigma_V*W_t)
# The term -0.5*sigma_V^2 is the Ito correction for log-normality
time_grid = np.linspace(0, T, N)
VT = V0 * np.exp((mu_V - 0.5 * sigma_V ** 2) * time_grid + sigma_V * W)

# ── Step 3: Assess Default ────────────────────────────────────
# A firm defaults if its asset value ever drops below debt B
default_occurred = any(VT <= B)
min_value = VT.min()
final_value = VT[-1]


# ── Step 4: Interpret Results ─────────────────────────────────
if default_occurred:
    print("The firm crossed the default threshold during the year.")
    print("In practice this triggers credit events and restructuring.")
else:
    print("The firm survived the full year without default.")
    print(f"Safety margin above default threshold: {min_value - B:.4f}")

# ── Step 5: Plot Asset Value Path ─────────────────────────────
plt.figure(figsize=(10, 6))

plt.plot(time_grid, VT, color='steelblue',
         linewidth=1.5, label="Asset Value Path $V_t$")

plt.axhline(y=B, color='red', linestyle='--',
            linewidth=1.5, label=f"Default Threshold B = {B}")

plt.axhline(y=V0, color='gray', linestyle=':',
            linewidth=1, label=f"Initial Value V0 = {V0}")

# Shade the danger zone below threshold
plt.fill_between(time_grid, 0, B,
                 alpha=0.1, color='red', label='Default Zone')

plt.title("Merton Structural Model: Simulated Asset Value Path\n"
          f"GBM with drift={mu_V}, volatility={sigma_V}, "
          f"horizon={T} year", fontsize=12)
plt.xlabel("Time (Years)")
plt.ylabel("Asset Value ($)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# I will ask The prof Shen, about this zone for better understanding