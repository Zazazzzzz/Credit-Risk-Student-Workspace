import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

simulations = 2000
obligors = 1000

alpha = 1.99
beta = 197.01
p_samples = np.random.beta(alpha, beta, size=simulations)
dependent_defaults = np.random.binomial(obligors, p_samples)

# Independent defaults
independent_defaults = []
for i in range(simulations):
    defaults = np.random.binomial(obligors, 0.01)
    independent_defaults.append(defaults)
plt.figure(figsize=(10, 6))

plt.hist(
    dependent_defaults,
    bins=150,
    range=(0, 150),
    density=True,
    alpha=0.6,
    color="skyblue",
    label="Dependent defaults: Beta-Bernoulli structure",
)
plt.hist(
    independent_defaults,
    bins=150,
    range=(0, 150),
    density=True,
    alpha=0.4,
    color="orange",
    label="Independent defaults",
)

plt.title("Simulated Default Number Distribution")
plt.xlabel("Number of Defaults")
plt.ylabel("Density")

plt.xlim(0, 150)
plt.ylim(0, 0.14)

plt.grid(True)
plt.legend()

plt.show()