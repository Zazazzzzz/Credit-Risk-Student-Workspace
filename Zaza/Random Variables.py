import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# the weather tomorrow
# 30 percent rain, 60 percent sunny, 10 percent hail

# one trial
weather_states = ["hail", "rain", "sunny"] # possible weather outcomes
probabilities = [0.1, 0.3, 0.6] # corresponding probabilities
one_day = np.random.choice(weather_states, p=probabilities)
print(one_day)

# N trial
N = 100
simulated_weather = np.random.choice(weather_states, size=N, p=probabilities)
print(simulated_weather[:10])

# encoding
hail = 1
rain = 2
sunny = 3
values = [hail, rain, sunny]
probabilities = [0.1, 0.3, 0.6]

N = 10
simulated_weather = np.random.choice(values, size=N, p=probabilities)
x = np.arange(N)  # 0, 1, ..., 99
print(simulated_weather)

plt.figure(figsize=(10, 4))
plt.scatter(x, simulated_weather)
plt.yticks([1, 2, 3], ["hail", "rain", "sunny"])
plt.xlabel("Simulation Number")
plt.ylabel("Weather Outcome")
plt.title("Simulated Weather Outcomes (10 Days)")
plt.grid(alpha=0.3)
plt.show()

# Create figure → make it active
# Add scatter → to active figure
# Add labels → to active figure
# Show → display active figure


# a strict object-oriented way
# create figure and axes
plot = plt.figure(figsize=(10, 4))
ax = plot.add_subplot(111)   # 1 row, 1 column, position 1

# plot using axes
ax.scatter(x, simulated_weather)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["hail", "rain", "sunny"])
ax.set_xlabel("Simulation Number")
ax.set_ylabel("Weather Outcome")
ax.set_title("Simulated Weather Outcomes (10 Days)")
ax.grid(alpha=0.3)
plot.show()

# Figure  → the whole canvas
# Axes    → the actual plot inside the canvas


# Two regions
hail = 1
rain = 2
sunny = 3
values = [hail, rain, sunny]
probabilities1 = [0.1, 0.3, 0.6]   # Region 1
probabilities2 = [0.02, 0.7, 0.28] # Region 2

N = 100
sim_r1 = np.random.choice(values, size=N, p=probabilities1)
sim_r2 = np.random.choice(values, size=N, p=probabilities2)

sim = np.vstack([sim_r1, sim_r2]) # vertically
sim = np.column_stack([sim_r1, sim_r2]) # column stack

df = pd.DataFrame(sim, columns=["Region1", "Region2"])

mapping = {1: "hail", 2: "rain", 3: "sunny"}
df_weather = df.replace(mapping)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True) # use the same y-axis range and scale

# common bins for discrete values 1,2,3
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

# ## Setup
#
# We model weather in two regions as **independent random variables** with three outcomes (hail, rain, sunny) and fixed probabilities:
#
# * Region 1: (0.1, 0.3, 0.6)
# * Region 2: (0.02, 0.7, 0.28)
#
# We simulate many days to approximate these distributions and compare outcomes.
#
# ---
#
# ## Takeaways
#
# * **Risk (hail):** Region 2 is much safer (2% vs 10%).
# * **Trade-off:** Region 1 offers more sunny days, Region 2 more rain.
# * **Statistical insight:** Simulation reproduces the underlying probabilities.
# * **Modeling insight:** Independence and fixed probabilities make the model simple and tractable, but also restrictive.


