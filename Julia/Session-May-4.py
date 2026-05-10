# here rho is asset corr, in zaza notes rho is the corr bw F and X
#Homework do Task 4 and 5 and Task 1 in Credit Risk
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

N = 2000        #simulati
m = 1000        #firms
p = 0.01        #defaultprobability
rho = 0.25      #correlation

# Default threshold
threshold = norm.ppf(p)


independent_defaults = np.random.binomial(1,p,size=(N, m))

#Number of defaults per sim
losses_independent = independent_defaults.sum(axis=1)
-
losses_correlated = np.zeros(N)

for k in range(N):
    F = np.random.normal() #systematic factor
    epsilon = np.random.normal(size=m) #idiosyncratic

    X = np.sqrt(rho)*F + np.sqrt(1-rho)*epsilon    # here rho is asset corr, in zaza notes rho is the corr bw F and X
    defaults = (X < threshold).astype(int)
    losses_correlated[k] = defaults.sum()


plt.figure(figsize=(12,5))

# Independent
plt.subplot(1,2,1)
plt.hist(losses_independent, bins=30)
plt.title("Independent Defaults")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")

# Correlated
plt.subplot(1,2,2)
plt.hist(losses_correlated, bins=30)
plt.title("Correlated Defaults")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# In matrix form
threshold = norm.ppf(p)

ind_defaults = np.random.binomial(
    1,
    p,
    size=(N, m)
)

losses_ind = ind_defaults.sum(axis=1)

F = np.random.normal(size=(N, 1))

epsilon = np.random.normal(size=(N, m))

X = np.sqrt(rho)*F + np.sqrt(1-rho)*epsilon

corr_defaults = (X < threshold).astype(int)

# Portfolio losses
losses_corr = corr_defaults.sum(axis=1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(losses_ind, bins=30)
plt.title("Independent Defaults")

plt.subplot(1,2,2)
plt.hist(losses_corr, bins=30)
plt.title("Correlated Defaults")

plt.show()