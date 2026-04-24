from encodings.punycode import decode_generalized_number

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setting
hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

prob1 = [0.1, 0.3, 0.6]
prob2 = [0.02, 0.7, 0.28]

# Step1. stimulate weather outcomes

# 1&2. stimulate weather outcomes
region1 = np.random.choice(values, size=100, p=prob1)
region2 = np.random.choice(values, size=100, p=prob2)

# 3. store the outcomes
results = np.column_stack((region1, region2))

print(results[:10])

'''Questions:
* What does one simulated value represent?
  It represents the weather of the specific day, if the value is 1, then the weather is hail.

* What does (N=100) represent in this context?
  It means that the weather conditions of 100 days are simulated.


* Why is this a simulation rather than real data?
  Because simulations are more ideal for tests. 
  Also, we can compare the simulation with the status in the real world to find patterns.
'''


# Step2. organize the stimulated data
df = pd.DataFrame({"region1": region1, "region2": region2})
# check for the first few rows
# print(df.head())

'''
Questions:

* What does one row represent?
  It represents the simulated weather condition for region 1 and region 2 on the same day.
  
* What does one column represent?
  It represents all the represented weather for a specific region.
'''


# Step3. visualize the outcomes as a scatter plot
plt.figure()

# put the 100 simulation numbers on the x-axis
x = np.arange(1,101)

# put the 3 weather outcomes on the y-axis and label the y-axis using hail, rain, sunny instead of 1,2,3
plt.yticks([1,2,3], ["hail", "rain", "sunny"])

# scatter the simulation outcome
plt.scatter(x, region1, label="region1")
plt.scatter(x, region2, label="region2")


plt.xlabel("Simulation number")
plt.ylabel("Weather outcome")
plt.title("Weather simulation outcome")
plt.legend() # To demonstrate which scatter represents which region
plt.show()

"""
Questions:

* Which outcome appears most often in Region 1?
  Sunny.
  
* Which outcome appears most often in Region 2?
  Rain.
  
* Does the picture roughly match the given probabilities?
  Yes.
"""


# Step4. draw histograms for both regions
fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)
# axes: return to axes[0] and axes[1], the two figures of region1 and region2
# sharey=True: use the same y-axis scale for both plots

bins = [0.5, 1.5, 2.5, 3.5]
# set 1, 2 and 3 in the middle

# region1
axes[0].hist(df["region1"], bins=bins, edgecolor='black')
axes[0].set_title("Region 1 Distribution")
axes[0].set_xticks([1, 2, 3],["hail", "rain", "sunny"])
axes[0].set_xlabel("Weather Outcome")
axes[0].set_ylabel("Frequency")

# region2
axes[1].hist(df["region2"], bins=bins, edgecolor='black')
axes[1].set_title("Region 2 Distribution")
axes[1].set_xticks([1, 2, 3],["hail", "rain", "sunny"])
axes[1].set_xlabel("Weather Outcome")
axes[1].set_ylabel("Frequency")

plt.show()

"""
Questions:

* Which region has the higher hail risk?
  Region 1.

* Which region has the higher rain probability?
  Region 2.

* Which region seems better to you and why?
  Region 1 seems better to me because it has the higher probability of sunny days and I love sunny days.
  
"""


# Step5. Count what happened out of 100 simulations
region1_counts = df["region1"].value_counts().sort_index()
region2_counts = df["region2"].value_counts().sort_index()

print("Region1:\n", region1_counts)
print("Region2:\n", region2_counts)

"""
Questions:

* Out of 100 simulated days, how many hail days occurred in Region 1?
  8.
  
* Out of 100 simulated days, how many hail days occurred in Region 2?
  4.

* Are the observed counts exactly equal to the theoretical probabilities times 100?
  No.
  
* Why or why not?
  Because the theoretical probability is the probability when the simulation is repeated for infinite times.

"""


# Step6. Compute relative frequencies
region1_freq = df["region1"].value_counts(normalize=True).sort_index()
region2_freq = df["region2"].value_counts(normalize=True).sort_index()
# normalize=True helps to make value counts into frequency