import numpy as np
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)
n = 1000
mu = [0, 0]

# Task 1: Generate Data
# Case A: ρ = 0 (Independent)
cov_a = [[1, 0], [0, 1]]
data_a = np.random.multivariate_normal(mu, cov_a, n)

# Case B: ρ = 0.5 (Correlated)
cov_b = [[1, 0.5], [0.5, 1]]
data_b = np.random.multivariate_normal(mu, cov_b, n)

# Task 2: Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

def style_plot(ax, data, title):
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    ax.set_title(title)
    ax.grid(True, linestyle='--')
    ax.set_aspect('equal')

style_plot(axes[0], data_a, "Correlation = 0")
style_plot(axes[1], data_b, "Correlation = 0.5")

plt.tight_layout()
plt.show()