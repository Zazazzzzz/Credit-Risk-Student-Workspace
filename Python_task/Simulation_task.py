# ---------------------------------------------------------------
# Script Name: Portfolio Loss Simulation
# Author: Shazia
# Description: Session 2 - Portfolio Simulation
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ── Setup ──────────────────────────────────────────────────────
n_firms = 10
default_probs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
loss_per_default = 50000
n_simulations = 1000

# ── Step 1: Simulate default indicators ────────────────────────
# For each simulation, for each firm, does it default? (1=yes, 0=no)
defaults = np.random.binomial(n=1, p=default_probs, size=(n_simulations, n_firms))

# ── Step 2: Compute losses for each firm ───────────────────────
losses = defaults * loss_per_default

# ── Step 3: Aggregate portfolio losses ─────────────────────────
total_losses = losses.sum(axis=1)

# ── Step 4: Analyse the loss distribution ──────────────────────
print(f"Average loss per simulation:   €{total_losses.mean():,.0f}")
print(f"Maximum loss in 1000 runs:     €{total_losses.max():,.0f}")
print(f"Minimum loss in 1000 runs:     €{total_losses.min():,.0f}")
print(f"Loss exceeded in worst 5%:     €{np.percentile(total_losses, 95):,.0f}")

# ── Step 5: Plot the loss distribution ─────────────────────────
plt.figure(figsize=(12, 6))
plt.hist(total_losses, bins=30, color='steelblue', edgecolor='black')
plt.axvline(total_losses.mean(), color='green', linestyle='--', label=f'Mean: €{total_losses.mean():,.0f}')
plt.axvline(np.percentile(total_losses, 95), color='red', linestyle='--', label=f'95% VaR: €{np.percentile(total_losses, 95):,.0f}')
plt.title('Portfolio Loss Distribution (1000 Simulations)')
plt.xlabel('Total Portfolio Loss (€)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()