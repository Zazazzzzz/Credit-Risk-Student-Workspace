# ============================================
# Simulating Weather as a Random Variable
# Course: Introduction to Credit Risk and Applications in Python
# Topic: Random Variables and Simulation
# Date: 2026-04-23
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# Description
# --------------------------------------------
# This script simulates weather outcomes using a discrete random variable.
# We generate outcomes for one region and compare two independent regions.
# The goal is to understand probability, simulation, and visualization.

# --------------------------------------------
# Setup: single region
# --------------------------------------------

weather_states = ["hail", "rain", "sunny"]
probabilities = [0.1, 0.3, 0.6]

# one trial
one_day = np.random.choice(weather_states, p=probabilities)
print("One day outcome:", one_day)

# multiple trials
N = 100
simulated_weather = np.random.choice(weather_states, size=N, p=probabilities)
print("First 10 simulations:", simulated_weather[:10])

# --------------------------------------------
# Encoding for plotting
# --------------------------------------------

hail, rain, sunny = 1, 2, 3
values = [hail, rain, sunny]

N = 10
simulated_weather = np.random.choice(values, size=N, p=probabilities)
x = np.arange(N)

# plotting (state-based)
plt.figure(figsize=(10, 4))
plt.scatter(x, simulated_weather)
plt.yticks([1, 2, 3], ["hail", "rain", "sunny"])
plt.xlabel("Simulation Number")
plt.ylabel("Weather Outcome")
plt.title("Simulated Weather Outcomes (10 Days)")
plt.grid(alpha=0.3)
plt.show()

# --------------------------------------------
# Object-oriented plotting
# --------------------------------------------

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

ax.scatter(x, simulated_weather)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["hail", "rain", "sunny"])
ax.set_xlabel("Simulation Number")
ax.set_ylabel("Weather Outcome")
ax.set_title("Simulated Weather Outcomes (10 Days)")
ax.grid(alpha=0.3)

plt.show()

# --------------------------------------------
# Two regions (independent)
# --------------------------------------------

probabilities1 = [0.1, 0.3, 0.6]   # Region 1
probabilities2 = [0.02, 0.7, 0.28] # Region 2

N = 100
sim_r1 = np.random.choice(values, size=N, p=probabilities1)
sim_r2 = np.random.choice(values, size=N, p=probabilities2)

# combine into DataFrame
sim = np.column_stack([sim_r1, sim_r2])
df = pd.DataFrame(sim, columns=["Region1", "Region2"])

# mapping to labels
mapping = {1: "hail", 2: "rain", 3: "sunny"}
df_weather = df.replace(mapping)

# --------------------------------------------
# Histogram comparison
# --------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

bins = [0.5, 1.5, 2.5, 3.5]

# Region 1
axes[0].hist(df["Region1"], bins=bins)
axes[0].set_title("Region 1")
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(["hail", "rain", "sunny"])
axes[0].set_xlabel("Weather")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", alpha=0.3)

# Region 2
axes[1].hist(df["Region2"], bins=bins)
axes[1].set_title("Region 2")
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["hail", "rain", "sunny"])
axes[1].set_xlabel("Weather")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------
# Takeaways (printed)
# --------------------------------------------

print("\nTakeaways:")
print("- Region 2 has much lower hail risk (2% vs 10%).")
print("- Region 1 has more sunny days; Region 2 has more rain.")
print("- Simulated frequencies approximate true probabilities.")
print("- Independence assumption simplifies modeling but is restrictive.")