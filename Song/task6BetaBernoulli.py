import numpy as np
import matplotlib.pyplot as plt


m = 1000
N = 2000
p = 0.01
rho = 0.005
np.random.seed(42)

alpha = p * (1 / rho - 1)
beta = (1 - p) * (1 / rho - 1)

P = np.random.beta(alpha, beta, N)
defaults = np.random.binomial(m, P)

plt.hist(defaults,
         bins=30,
         density=True,
         alpha=0.6,
         edgecolor='black',
         label='Histogram')

counts = np.bincount(defaults)
pmf = counts / N
x = np.arange(len(pmf))

plt.plot(x, pmf, 'o-', label='PMF = counts / N')

plt.title("Histogram and PMF of Defaults")
plt.xlabel("Number of defaults")
plt.ylabel("Density / PMF")
plt.legend()
plt.show()
plt.savefig("Histogram and PMF of Defaults.png", dpi=300, bbox_inches="tight")