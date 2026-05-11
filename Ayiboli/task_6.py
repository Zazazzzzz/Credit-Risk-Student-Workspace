import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1000  # number of obligors
N = 2000  # number of simulations
pi = 0.01  # unconditional default probability
rho = 0.005  # default correlation = 0.5%

np.random.seed(42)

# Calculate beta parameters a and b
a_plus_b = 1 / rho - 1
a = pi * a_plus_b
b = (1 - pi) * a_plus_b

print("a =", a)
print("b =", b)

# Simulation
default_counts = []

for _ in range(N):
    # Step 1: draw common random default probability Q
    Q = np.random.beta(a, b)

    # Step 2: simulate number of defaults conditional on Q
    M = np.random.binomial(m, Q)

    default_counts.append(M)

# Convert to numpy array
default_counts = np.array(default_counts)

# Plot histogram
plt.hist(default_counts, bins=30, edgecolor="black")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.title("Beta-Bernoulli Default Simulation")
plt.show()