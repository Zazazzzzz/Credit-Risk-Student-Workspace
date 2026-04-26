### Task: Simulating Weather Outcomes in Two Regions
### Setting
# We model tomorrow’s weather in two regions. The possible outcomes are:
#* hail = 1
#* rain = 2
#* sunny = 3

#task today============================================================================
#1
import numpy as np
hail = 1
rain = 2
sunny = 3
values = [hail, rain, sunny]
#2
probabilities1 = [0.1, 0.3, 0.6]
probabilities2 = [0.02, 0.7, 0.28]

#3: simulate 100 days
N = 100

region1 = np.random.choice(values, size=N, p=probabilities1)
region2 = np.random.choice(values, size=N, p=probabilities2)

print("Region 1:", region1)
print("Region 2:", region2)
#step2 merge both=====================================================================

import pandas as pd

weather_data = pd.DataFrame({
    "Region1": region1,
    "Region2": region2
})

print("\nWeather data:")
print(weather_data)

print("\nStep 1 answers:")
print("One simulated value represents the weather outcome for one region on one simulated day.")
print("N = 100 means that we simulate 100 possible days.")
print("This is a simulation because the data is generated from assumed probabilities, not observed from real weather.")

print("\nStep 2 answers:")
print("One row represents one simulated day.")
print("One column represents the simulated weather outcomes for one region.")

#step3 scatter plot===================================================================

import matplotlib.pyplot as plt

x = range(1, N + 1)

plt.scatter(x, region1, label="Region 1")
plt.scatter(x, region2, label="Region 2")
plt.yticks([1, 2, 3], ["hail", "rain", "sunny"])
plt.xlabel("Simulation number")
plt.ylabel("Weather outcome")
plt.title("Weather simulation")
plt.legend()
plt.show()

#step4 histograms=====================================================================

plt.subplot(1, 2, 1)
plt.hist(region1, bins=[0.5, 1.5, 2.5, 3.5])
plt.xticks([1, 2, 3], ["hail", "rain", "sunny"])
plt.title("Region 1")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(region2, bins=[0.5, 1.5, 2.5, 3.5])
plt.xticks([1, 2, 3], ["hail", "rain", "sunny"])
plt.title("Region 2")

plt.show()

#step5 count outcomes=================================================================

counts_region1 = weather_data["Region1"].value_counts().sort_index()
counts_region2 = weather_data["Region2"].value_counts().sort_index()

counts = pd.DataFrame({
    "Region1": counts_region1,
    "Region2": counts_region2
})

counts = counts.fillna(0)
counts = counts.astype(int)

counts.index = ["hail", "rain", "sunny"]

print("\nCounts out of 100 simulations:")
print(counts)

#step6 relative frequencies===========================================================

relative_frequencies = counts / N

print("\nRelative frequencies:")
print(relative_frequencies)

theoretical_probabilities = pd.DataFrame({
    "Region1": probabilities1,
    "Region2": probabilities2
}, index=["hail", "rain", "sunny"])

print("\nTheoretical probabilities:")
print(theoretical_probabilities)

#step7 and step8 interpretation=======================================================

print("\nStep 3 answers:")
print("The most common outcome in Region 1 is usually sunny.")
print("The most common outcome in Region 2 is usually rain.")
print("The picture should roughly match the given probabilities, but not exactly because the sample is random.")

print("\nStep 4 answers:")
print("Region 1 has the higher hail risk because its hail probability is 10%, compared with 2% in Region 2.")
print("Region 2 has the higher rain probability.")
print("Region 1 may look better if we prefer sunny weather, but Region 2 is safer if we mainly want to avoid hail.")

print("\nStep 5 answers:")
print("Hail days in Region 1:", counts.loc["hail", "Region1"])
print("Hail days in Region 2:", counts.loc["hail", "Region2"])
print("The observed counts are not exactly equal to the theoretical probabilities times 100 because of randomness.")

print("\nStep 6 answers:")
print("The empirical frequencies should be close to the theoretical probabilities.")
print("If N increases to 1000 or 10000, the empirical frequencies should become closer to the theoretical probabilities.")

print("\nStep 7 answers:")
print("Region 1 is riskier with respect to hail.")
print("A risk-averse person would choose Region 2 because it has lower hail probability.")

print("\nStep 8 answers:")
print("Independence means that the weather outcome on one simulated day does not affect another simulated day.")
print("Fixed probabilities mean that the probabilities do not change over time.")
print("The model is easy to simulate because each day can be generated using the same probability distribution.")
print("It may be unrealistic because real weather can depend on seasons, previous days, geography, and changing conditions.")
