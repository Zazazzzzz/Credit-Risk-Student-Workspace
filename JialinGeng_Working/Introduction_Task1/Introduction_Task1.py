# The answers of the Task1 in the Introduction folder.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# possible weather outcomes, type(values)
hail, rain, sunny = 1, 2, 3
values = [hail, rain, sunny]
weather_outcomes = ["hail", "rain", "sunny"]

# Step 1. Simulate weather outcomes
N = 100
np.random.seed(42)

#Region 1
probabilities1 = [0.1, 0.3, 0.6] # corresponding probabilities for region_1
region1_sim = np.random.choice(values, size=N, p=probabilities1)

#Region 2
probabilities2 = [0.02, 0.7, 0.28] # corresponding probabilities for region_2
region2_sim = np.random.choice(values, size=N, p=probabilities2)

# Step 2. Store the two simulation results together in one object.
sim_data = (region1_sim,region2_sim)

# Questions & Answers:
# What does one simulated value represent?
# The outcomes for a single day in two regions.
# What does (N=100) represent in this context?
# The number of simulated days.
# Why is this a simulation rather than real data?
# Because we generate outcomes artificially using a random process and given probabilities, not by observing real weather.

# Organize the simulated data
# Combine the two simulated series into a table with two columns:


df = pd.DataFrame({
    "Region1": region1_sim,
    "Region2": region2_sim
})

print(df.head())

# What does one row represent?
# One simulated day’s weather in both regions.
# What does one column represent?
# The time series of weather outcomes for one specific region.

# Step 3. Visualize the outcomes as a scatter plot

# Map numeric values to labels
weather_labels = {1: "hail", 2: "rain", 3: "sunny"}

plt.figure(figsize=(10, 5))
plt.scatter(range(N), region1_sim, label="Region 1", alpha=0.6)
plt.scatter(range(N), region2_sim, label="Region 2", alpha=0.6)

plt.yticks(list(weather_labels.keys()), list(weather_labels.values()))
plt.xlabel("Simulation day")
plt.ylabel("Weather outcome")
plt.title("Weather outcomes over 100 days")
plt.legend()
plt.show()

# Questions:
# Which outcome appears most often in Region 1?
# Sunny
# Which outcome appears most often in Region 2?
# Rain
# Does the picture roughly match the given probabilities?
# Yes

# Step 4. Draw histograms for both regions

fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Use one subplot for each region

axes[0].hist(region1_sim, bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, align='mid')
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(["hail", "rain", "sunny"])
axes[0].set_title("Region 1")
axes[0].set_ylim(0, 80)  # the same y-axis scale for both plots

axes[1].hist(region2_sim, bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, align='mid')
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["hail", "rain", "sunny"])
axes[1].set_title("Region 2")
axes[1].set_ylim(0, 80)  # the same y-axis scale for both plots

plt.show()

# Questions:
#
# Which region has the higher hail risk?
# Region1
# Which region has the higher rain probability?
# Region2
# Which region seems better to you and why?
# I prefer the region2, because I dislike the extreme weather(hail) and rain is also not bad for me.

# Step 5. Count what happened out of 100 simulations

counts1 = {v: int(np.sum(region1_sim == v)) for v in values}
counts2 = {v: int(np.sum(region2_sim == v)) for v in values}

# Replace numeric keys with weather labels
weather_labels = {1: "hail", 2: "rain", 3: "sunny"}

counts1_labeled = {weather_labels[k]: v for k, v in counts1.items()}
counts2_labeled = {weather_labels[k]: v for k, v in counts2.items()}

print("Region 1 counts:", counts1_labeled)
print("Region 2 counts:", counts2_labeled)

# Questions:
#
# Out of 100 simulated days, how many hail days occurred in Region 1?
# 13
# Out of 100 simulated days, how many hail days occurred in Region 2?
# 2
# Are the observed counts exactly equal to the theoretical probabilities times 100?
# No, because of the random sampling variation (simulation noise). The larger the sample size, the more accurate it become.
# Why or why not?
# Because random draws fluctuate around the expected values; exact equality only in expectation, not in a single finite sample. This is just a fixed sample.（np.random.seed(42)）

# Step 6. Compute relative frequencies

freq1 = {k: v / N for k, v in counts1.items()}
freq2 = {k: v / N for k, v in counts2.items()}

# Replace numeric keys with weather labels
weather_labels = {1: "hail", 2: "rain", 3: "sunny"}

freq1_labeled = {weather_labels[k]: v for k, v in freq1.items()}
freq2_labeled = {weather_labels[k]: v for k, v in freq2.items()}

print("Region 1 empirical frequencies:", freq1_labeled)
print("Region 2 empirical frequencies:", freq2_labeled)

print("Region 1 theoretical:", dict(zip(weather_outcomes, probabilities1)))
print("Region 2 theoretical:", dict(zip(weather_outcomes, probabilities2)))

# Questions:
# Compare the empirical frequencies with the theoretical probabilities
# Are they close?
# close for N = 100， but not exact.
# What would happen if (N) were increased to 1000 or 10000?
# Empirical frequencies would get closer to theoretical probabilities (law of large numbers).


# Step 7. Interpret the results from a risk perspective
# Take **hail** as the bad outcome.
# Region 1: hail probability 10% (empirical maybe ~10%)
# Region 2: hail probability 2% (empirical maybe ~1–2%)

# Questions:
# Which region is riskier with respect to hail?
# Region 1.
# If you are risk-averse, which region would you choose?
# Region 2, because it has lower hail risk.


# Step 8. Reflect on the assumptions

# This model is simple because of strong assumptions.

# Questions:

# What does independence mean here?
# Weather on one day does not affect the next day, and Region 1’s weather does not affect Region 2’s.
# What does it mean that the probabilities are fixed?
# The chance of hail/rain/sunny is constant over all 100 days (no seasons or trends).
# Why is this model easy to simulate?
# No complex dependencies, just independent random draws from a known distribution.
# Why may it still be unrealistic in real life?
# Weather is often correlated across days (persistence), regions (spatial correlation), and probabilities change seasonally.
