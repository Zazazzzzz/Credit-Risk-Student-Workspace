import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


N = 2000
m = 1000
p = 0.01
rho = 0.25
np.random.seed(42)

def simulate_correlated_defaults(N, m, p, rho):
    F = np.random.normal(0,1,N)
    epsilon = np.random.normal(0,1,(N,m))
    X = rho * F[:,np.newaxis] + np.sqrt(1 - rho ** 2) * epsilon
    d = norm.ppf(p)
    defaults = (X < d).sum(axis=1)
    return defaults

# independent defaults
independent_defaults = simulate_correlated_defaults(N, m, p, 0)

# correlated defaults
correlated_defaults = simulate_correlated_defaults(N, m, p, rho)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,6))
ax1.hist(independent_defaults, bins= range(0,31), density=True, label="Independent")
ax1.set_title("Independent")
ax1.set_xlabel("Number of defaults")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True, axis="y", linestyle="--", linewidth=0.5)


ax2.hist(correlated_defaults, bins= range(0,31), density=True, label="Correlated(0.25)")
ax2.set_title("Correlated(0.25)")
ax2.set_xlabel("Number of defaults")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("task4.png", dpi=300, bbox_inches="tight")
plt.show()




