import random
import matplotlib.pyplot as plt

simulations = 2000
total_losses = []

for i in range(simulations):
    loss = 0

    for j in range(50):
        if random.random() < 0.005:
            loss = loss + (1 / 100) * 0.4

    for j in range(50):
        if random.random() < 0.002:
            loss = loss + (1 / 100) * 0.3

    total_losses.append(loss)

el1 = 50 * 0.005 * 0.4 * (1 / 100)
el2 = 50 * 0.002 * 0.3 * (1 / 100)
theoretical_el = el1 + el2
print("Theoretical Expected Loss:", theoretical_el)

simulated_el = sum(total_losses) / len(total_losses)
print("Simulated Average Loss:", simulated_el)

plt.figure(figsize=(10, 6))

plt.hist(
    total_losses,
    bins=40,
    density=True,
    color="#d9a5a5",
    edgecolor="#bfa2a2",
    label="Histogram",
)

import numpy as np
from scipy.stats import gaussian_kde

x = np.linspace(min(total_losses) - 0.002, max(total_losses) + 0.002, 500)
density_function = gaussian_kde(total_losses)
density_function.set_bandwidth(bw_method=0.1)
y = density_function(x)

plt.plot(x, y, color="gray", linewidth=2, label="KDE Curve")

plt.title("Density Plot of Total Loss Across 2000 Simulations")
plt.xlabel("Total Loss")
plt.ylabel("Density (Probability)")
plt.grid(axis="y", linestyle="--")
plt.legend()
plt.show()