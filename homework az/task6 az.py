import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
m = 1000
n_sims = 2000
mu = 0.01
rho = 0.005
strength = (1 / rho) - 1
alpha = mu * strength
beta = (1 - mu) * strength

defaults_count = []

for _ in range(n_sims):
    p_sim = np.random.beta(alpha, beta)
    y = np.random.binomial(n=1, p=p_sim, size=m)
    defaults_count.append(y.sum())

plt.figure(figsize=(10, 6))
plt.hist(defaults_count, bins=range(0, max(defaults_count) + 2),
         density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Beta-Bernoulli Default Distribution (m=1000, rho={rho * 100}%)')
plt.xlabel('Number of Defaults')
plt.ylabel('Frequency')
plt.show()