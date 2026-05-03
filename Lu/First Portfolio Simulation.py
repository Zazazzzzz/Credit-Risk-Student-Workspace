import numpy as np
import matplotlib.pyplot as plt

n_firms = 10

PD = np.linspace(0.01, 0.10, n_firms)

EAD = 100000
LGD = 0.5
loss_per_default = EAD * LGD  # 50000

n_sim = 10000

losses = []

for _ in range(n_sim):

    defaults = np.random.binomial(1, PD)

    individual_losses = defaults * loss_per_default

    total_loss = np.sum(individual_losses)

    losses.append(total_loss)

losses = np.array(losses)

EL = np.mean(losses)

VaR_95 = np.percentile(losses, 95)

print("Expected Loss:", EL)
print("VaR 95%:", VaR_95)
plt.hist(losses, bins=30)
plt.title("Portfolio Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.show()