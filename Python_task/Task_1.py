import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Define weather states
# -------------------------
hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

# Probabilities
probabilities1 = [0.1, 0.3, 0.6]
probabilities2 = [0.02, 0.7, 0.28]

# Number of simulated days
N = 100

# -------------------------
# Step 1: Simulate outcomes
# -------------------------
region1 = np.random.choice(values, size=N, p=probabilities1)
region2 = np.random.choice(values, size=N, p=probabilities2)

# -------------------------
# Step 2: Put into table
# -------------------------
weather = pd.DataFrame({
    "Region1": region1,
    "Region2": region2
})

print("\nFirst 10 simulated days:")
print(weather.head(10))


# -------------------------
# Step 3: Scatter plot
# -------------------------
plt.figure()
plt.scatter(range(N), region1, label="Region 1")
plt.scatter(range(N), region2, label="Region 2")

plt.yticks(
    [1,2,3],
    ["hail","rain","sunny"]
)

plt.xlabel("Simulation Day")
plt.ylabel("Weather Outcome")
plt.title("Weather Simulation")
plt.legend()
plt.show()


# -------------------------
# Step 4: Histograms
# -------------------------
fig, ax = plt.subplots(1,2)

ax[0].hist(region1, bins=[0.5,1.5,2.5,3.5])
ax[0].set_title("Region 1")

ax[1].hist(region2, bins=[0.5,1.5,2.5,3.5])
ax[1].set_title("Region 2")

plt.show()


# -------------------------
# Step 5: Counts
# -------------------------
print("\nCounts Region 1")
print(pd.Series(region1).value_counts().sort_index())

print("\nCounts Region 2")
print(pd.Series(region2).value_counts().sort_index())


# -------------------------
# Step 6: Relative frequencies
# -------------------------
print("\nEmpirical Frequencies Region 1")
print(pd.Series(region1).value_counts(normalize=True).sort_index())

print("\nEmpirical Frequencies Region 2")
print(pd.Series(region2).value_counts(normalize=True).sort_index())

# -------------------------
# Step 7 Risk Interpretation
# -------------------------

#print("\nRISK INTERPRETATION")
#print("Region 1 has higher hail risk (10%) than Region 2 (2%).")
#print("A risk-averse decision maker would prefer Region 2.")
#print("Region 2 has more rain, but much lower severe hail risk.")