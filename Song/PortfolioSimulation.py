from pickle import REDUCE

import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1])
loss_given_default = 50000
n_simulations = 1000
np.random.seed(42)

u = np.random.uniform(0, 1, size=(n_simulations,10))
defaults = u < p
number_of_defaults = defaults.sum(axis=1)

total_losses = number_of_defaults * loss_given_default
theoretical_EL = np.sum(p * loss_given_default)
simulated_average_loss = total_losses.mean()

VaR_95 = np.quantile(total_losses, 0.95)
VaR_99 = np.quantile(total_losses, 0.99)

print("theoretical_EL:", theoretical_EL)
print("simulated_average_loss:", simulated_average_loss)
print("VaR_95%:", VaR_95)
print("VaR_99%:", VaR_99)
print("Maximum Loss:", total_losses.max())


plt.figure(figsize=(8, 5))
plt.hist(total_losses, bins=20, edgecolor="black")
plt.xlabel("Total Loss")
plt.ylabel("Frequency")
plt.title("Simulated Portfolio Loss Distribution")

plt.axvline(theoretical_EL, linestyle="--", color = "Red", label="Theoretical EL")
plt.axvline(VaR_99, linestyle="--", label="VaR 99%")
plt.legend()
plt.show()





