import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 2000          #num of sim
n_obligors = 1000
p = 0.01 #each oblig as 1% chance of default
rho = 0.005 #dependence bw obligor defualts

s = 1 / rho - 1  #(alpha+beta)
alpha = p * s
beta = (1 - p) * s

print(f"alpha = {alpha:}, beta = {beta:}")

defaults = []

for _ in range(N): #reapeat for 2000 times the portfolio simulation
    #Common random default probability
    P = np.random.beta(alpha, beta) #same across obligors for this one simulation
    # default for firms
    Y = np.random.binomial(1, P, size=n_obligors)

    #Tot defaults in portfolio
    defaults.append(np.sum(Y))

defaults = np.array(defaults)

values, counts = np.unique(defaults, return_counts=True)

pmf = counts / N

# Histogram
plt.hist(defaults,
         bins=30,
         density=True,
         alpha=0.6,
         edgecolor='black',
         label='Histogram')
# PMF
plt.plot(values,
         pmf,
         'o-',
         linewidth=2,
         label='Estimated PMF')

plt.xlabel("Number of Defaults")
plt.ylabel("Proba")
plt.title("Portfolio Defaults")
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

