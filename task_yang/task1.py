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

