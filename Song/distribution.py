import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

alpha = 2
beta = 8

p = np.random.beta(alpha, beta , size = 1000)

plt.hist(p, bins = 50 , density = True , color = 'blue')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.show()

np.random.seed(42)
k = 2
theta = 3
lam = np.random.gamma(shape=k, scale=theta, size = 10000)

plt.hist(lam, bins = 50 , density = True , color = 'blue')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.show()
