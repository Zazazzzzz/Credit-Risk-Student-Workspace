import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_simulations = 2000      # number of simulations
n_obligors = 1000         # number of obligors

pd_unconditional = 0.01   # unconditional default probability = 1%
default_corr = 0.005      # default correlation = 0.5%
# For Beta(alpha, beta):
alpha_plus_beta = 1 / default_corr - 1   # rho_y = 1 / (alpha + beta + 1)     rho = 1 /(s + 1)   s = (1 - rho)/ rho
alpha = pd_unconditional * alpha_plus_beta   #  E[P] = alpha / (alpha + beta) = p     a = p * ( a + b ) = p * s
beta = (1 - pd_unconditional) * alpha_plus_beta   #E[P] = alpha / (alpha + beta) = p

print("alpha + beta = ", alpha_plus_beta)
print("alpha = ", alpha)
print("beta = ", beta)

print("check mean PD =", alpha / (alpha + beta))
print("check default correlation = ", 1/ (alpha + beta + 1))
# Step 2: Simulate random portfolio-level PDs
# Each simulation draws one common portfolio-level PD
random_pd = np.random.beta(alpha, beta, size=n_simulations)
default_counts = np.random.binomial(
    n=n_obligors,
    p=random_pd,
    size=n_simulations
)
print("Mean number of defaults =", np.mean(default_counts))
print("Std of number of defaults =", np.std(default_counts))
print("Min number of defaults =", np.min(default_counts))
print("Max number of defaults =", np.max(default_counts))

plt.figure(figsize=(8, 5))
plt.hist(default_counts, bins=30, density=True, edgecolor="black", alpha=0.7)

plt.xlabel("Number of Defaults")
plt.ylabel("Estimated Probability")
plt.title("Task 6: Beta-Binomial Default Count Distribution")
plt.grid(alpha=0.3)
plt.show()














