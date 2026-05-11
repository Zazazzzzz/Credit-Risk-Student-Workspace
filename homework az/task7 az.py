import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
n_sims = 2000
n1 = 50
p1 = 0.005
lgd1 = 0.4

n2 = 50
p2 = 0.002
lgd2 = 0.3

weight = 1 / 100

el1 = n1 * p1 * lgd1 * weight
el2 = n2 * p2 * lgd2 * weight
total = el1 + el2
print(f"Expected Loss: {total:.4f} (即 {total*100:.2f}%)")

defaults_g1 = np.random.binomial(n=n1, p=p1, size=n_sims)
defaults_g2 = np.random.binomial(n=n2, p=p2, size=n_sims)


portfolio_losses = (defaults_g1 * lgd1 + defaults_g2 * lgd2) * weight

plt.figure(figsize=(10, 6))
plt.hist(portfolio_losses, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(total, color='red', linestyle='dashed', linewidth=2, label=f'Expected Loss: {total:.4f}')

plt.title('Portfolio Loss Distribution (Independent Defaults)')
plt.xlabel('Portfolio Loss (Fraction of Total Portfolio)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
