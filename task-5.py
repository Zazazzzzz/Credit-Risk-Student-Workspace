import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

# Dummy simulation data
np.random.seed(42)
simulated_defaults = np.random.negative_binomial(1, 0.1, 10000)

#  Compute theoretical probabilities for independent defaults
n, p = 1000, 0.01
k_values = np.arange(0, 50)
binomial_probs = binom.pmf(k_values, n, p)

# Compute empirical probabilities for correlated defaults from simulation
total_simulations = len(simulated_defaults)
correlated_probs = [
    np.sum(simulated_defaults == k) / total_simulations for k in k_values
]

# 4. Plot the results using a line-and-dot plot (marker='o')
plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    binomial_probs,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    label="Independent (Binomial)",
    color="blue",
)
plt.plot(
    k_values,
    correlated_probs,
    marker="o",
    linestyle="-",
    linewidth=1.5,
    label="Correlated (Simulation)",
    color="orange",
)

#  Chart styling
plt.title("Comparison of Default Number Distributions")
plt.xlabel("Number of Defaults (k)")
plt.ylabel("Probability")
plt.xlim(-1, 40)
plt.legend()
plt.grid(axis="both", linestyle="--", alpha=0.5)

plt.show()