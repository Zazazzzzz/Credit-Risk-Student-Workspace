import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# settings
mu, sigma = 0.5, 2.0
d = -1.0
# standardization: z = (x - mu) / sigma
z_d = (d - mu) / sigma
# prob of falling below d
prob = norm.cdf(d, mu, sigma)
# plotting
plt.figure(figsize=(10, 4))
# plot 1: original
plt.subplot(1, 2, 1)
x_axis = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.fill_between(x_axis, norm.pdf(x_axis, mu, sigma), where=(x_axis <= d), color='red', alpha=0.3)
plt.axvline(d, color='red', ls='--')
plt.title('Original X')
# plot 2: standard normal
plt.subplot(1, 2, 2)
z_axis = np.linspace(-4, 4, 500)
plt.plot(z_axis, norm.pdf(z_axis, 0, 1), color='orange')
plt.fill_between(z_axis, norm.pdf(z_axis, 0, 1), where=(z_axis <= z_d), color='red', alpha=0.3)
plt.axvline(z_d, color='red', ls='--')
plt.title(f'Standard Z (z={z_d:.2f})')
plt.tight_layout()
plt.show()
print(f"Target d: {d}")
print(f"Z-score: {z_d:.4f}")
print(f"P(X < d): {prob:.4f}")
