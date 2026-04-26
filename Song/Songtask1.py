import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

probabilities1 = [0.1, 0.3, 0.6]   # Region 1
probabilities2 = [0.02, 0.7, 0.28] # Region 2

# Step 1. Simulate weather outcomes

n = 100

sim_1 = np.random.choice(values, size=n, p=probabilities1)
sim_2 = np.random.choice(values, size=n, p=probabilities2)

"""

1. One simulated value represents the weather outcome for one day in one region. The simulation uses numerical coding, where 1 corresponds to “hail”, 2 corresponds to “rain”, and 3 corresponds to “sunny”.

2. N = 100 means we simulate 100 days.

3.This is a simulation because the data are randomly generated based on the given probability distribution.

"""

# Step 2. Organize the simulated data

df = pd.DataFrame({"Region1": sim_1,"Region2": sim_2})

mapping = {1: "hail", 2: "rain", 3: "sunny"}
df_weather = df.replace(mapping)

print(df_weather.head(10))

"""

1. One row represents one simulated day, showing the weather outcomes for both regions on that day.

2. Each column represents a region. “Region1” shows the simulated weather outcomes for Region 1, and “Region2” shows the simulated weather outcomes for Region 2.

"""

# Step 3. Visualize the outcomes as a scatter plot

plt.figure()

x = np.arange(1, n + 1)

plt.scatter(x, sim_1, label="Region1")
plt.scatter(x, sim_2, label="Region2")

plt.yticks([1, 2, 3], ["hail", "rain", "sunny"])

plt.xlabel("Simulation number")
plt.ylabel("Weather outcome")

plt.title("Simulated Weather Outcomes")

plt.legend(loc="best")
plt.show()

"""

1. Sunny.

2. Rain.

3. Yes. Region 1 has a higher probability of sunny weather, and region 2 has a higher probability of rain.

"""

# Step 4. Draw histograms for both regions

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

axes[0].hist(sim_1, bins=[0.5, 1.5, 2.5, 3.5])
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(["hail", "rain", "sunny"])
axes[0].set_title("Region 1")
axes[0].set_ylabel("Frequency")

axes[1].hist(sim_2, bins=[0.5, 1.5, 2.5, 3.5])
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["hail", "rain", "sunny"])
axes[1].set_title("Region 2")

plt.tight_layout()
plt.show()

"""

1. Region 1.

2. Region 2.

3. Region 1 seems better because it has the highest frequency of sunny outcomes. Additionally, it has a lower probability of rain compared to Region 2. Although hail is an extreme weather condition and its probability is higher in Region 1 than in Region 2, I can accept this risk.

"""

# Step 5. Count what happened out of 100 simulations

hail_r1 = (sim_1 == 1).sum()
rain_r1 = (sim_1 == 2).sum()
sunny_r1 = (sim_1 == 3).sum()

hail_r2 = (sim_2 == 1).sum()
rain_r2 = (sim_2 == 2).sum()
sunny_r2 = (sim_2 == 3).sum()

print("Region 1:", hail_r1, rain_r1, sunny_r1)
print("Region 2:", hail_r2, rain_r2, sunny_r2)

"""
Region 1: 14 24 62
Region 2: 2 77 21

1. 14 days.

2. 2 days.

3 & 4.The observed counts are not exactly equal to the theoretical probabilities times 100. This is because the data are randomly generated, and due to randomness and finite sample size, the results can vary.

"""

# Step 6. Compute relative frequencies

freq_r1 = [ hail_r1 / n, rain_r1 / n, sunny_r1 / n ]

freq_r2 = [ hail_r2 / n, rain_r2 / n, sunny_r2 / n ]

print("Region 1 frequencies:", freq_r1)
print("Region 2 frequencies:", freq_r2)

""""

Region 1 frequencies: 0.14 0.34 0.52
Region 1 frequencies: 0.01 0.66 0.33

1. The empirical frequencies are close to the theoretical probabilities. For Region 1, (0.14, 0.34, 0.52) are close to (0.1, 0.3, 0.6). For Region 2, (0.01, 0.66, 0.33) are close to (0.02, 0.7, 0.28).

2. Yes, they are reasonably close, but not exactly equal due to randomness.

3.If N were increased, the empirical frequencies would become closer to the theoretical probabilities. As the number of trials increases, the actual results get closer to the theoretical probabilities.

"""

# Step 7. Interpret the results from a risk perspective

""""

1. Region 1 is riskier with respect to hail because it has a higher probability of hail than Region 2.

2. If I am risk-averse, I would choose Region 2 because it has a lower risk of hail, which is considered the bad outcome.

"""

# Step 8. Reflect on the assumptions

""""

1. Independence means that the weather outcome of one simulated day does not depend on the weather outcomes of previous days.

2. Fixed probabilities mean that the probabilities of hail, rain, and sunny weather stay the same for all simulated days.

3. This model is easy to simulate because each day is generated randomly using the same fixed probability distribution.

4. The model may be unrealistic in real life because weather is usually not fully independent. For example, weather conditions can depend on the previous day, 1. Independence means that the weather outcome of one simulated day does not depend
on the weather outcomes of previous days.

"""