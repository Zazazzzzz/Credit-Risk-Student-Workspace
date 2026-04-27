#basic statistics
#Task number 1
import numpy as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

probabilities1 = [0.1, 0.3, 0.6]
probabilities2 = [0.02, 0.7, 0.28]

# Task
N=100

pos_1=np.random.choice(values, N,p=probabilities1)
pos_2=np.random.choice(values, N,p=probabilities2)

df = pd.DataFrame({
    "Region_1": pos_1,
    "Region_2": pos_2
})
print(df[:10])

"""Questions to answer: 
* What does one simulated value represent?
* What does (N=100) represent in this context?
* Why is this a simulation rather than real data? """

""" 
One value represents one draw from the assumed probability distribution above. They are a random variable. 

The 100 are the number of simulated outcomes. 

As said before, these are random pulls from a distribution, they are not real life observations of weather realization.
These outcomes are theoretical (solely based on the prob. distrib.). 

"""

"""
Questions:

* What does one row represent?
* What does one column represent?"""

"""
The rows are the observations/draw number
Column 1 is the Region 1, Col2 is Region 2
"""


x = range(len(df))
plt.figure()

plt.scatter(x, df["Region_1"], label="Region 1")
plt.scatter(x, df["Region_2"], label="Region 2")

# Replace numeric y-axis with labels
plt.yticks([1, 2, 3], ["hail", "rain", "sunny"])

plt.xlabel("Simulation")
plt.ylabel("Outcome")
plt.title("Weather Simulation")
plt.legend()
plt.show()


"""* Which outcome appears most often in Region 1?

Region 1 is mostly sunny. 

* Which outcome appears most often in Region 2?

In Region 2 it appears to rain a lot. 

* Does the picture roughly match the given probabilities?

Yes this matches our highest probabilities for each regions, 0.6 and 0.7 
respectivaly

"""

#Counting:
hail_days_r1= (df["Region_1"]==1).sum()
hail_days_r2= (df["Region_2"]==1).sum()

print(f"Hail days in Region 1: {hail_days_r1}, Hail days in Region 2: {hail_days_r2}")



""" 
* Out of 100 simulated days, how many hail days occurred in Region 1?
* Out of 100 simulated days, how many hail days occurred in Region 2?
* Are the observed counts exactly equal to the theoretical probabilities times 100?

No it does not match the probabilities exactly. 

* Why or why not?
This is because these are 100 random draws from the given distribution,
the draws will scatter around the probabilities.

"""

# * Compare the empirical frequencies with the theoretical probabilities
empirical_freq_hail_reg_1= hail_days_r1/N
empirical_freq_hail_reg_2= hail_days_r2/N

empirical_freq_rain_reg1= ((df["Region_1"]==2).sum())/N
empirical_freq_rain_reg2= ((df["Region_2"]==2).sum())/N

empirical_freq_sunny_reg1= ((df["Region_1"]==3).sum())/N
empirical_freq_sunny_reg2= ((df["Region_2"]==3).sum())/N

print(f"Empirical_freq_Region_1:hail {empirical_freq_hail_reg_1},rain {empirical_freq_rain_reg1}, sunny {empirical_freq_sunny_reg1}")

print(f"Empirical_freq_Region_2:hail {empirical_freq_hail_reg_2},rain {empirical_freq_rain_reg2}, sunny {empirical_freq_sunny_reg2}")


#* Are they close?
# More or less. If N was increased they would get even closer to the given
#probabilities (by the Law of large numbers).

"""Take **hail** as the bad outcome.

Questions:

* Which region is riskier with respect to hail?

One would rather live in Region 2. 

* If you are risk-averse, which region would you choose? 
Again one would try to minimize the downside risk by living in Region 2 """

"""## Step 8. Reflect on the assumptions

This model is simple because of strong assumptions.

Questions:

* What does independence mean here?
The outcome does not depend on the previous day for example. It's probability "stands alone".

* What does it mean that the probabilities are fixed?
Every simulation has the same probability distribution. 

* Why is this model easy to simulate?
We have a known distribution and use a simple random draw and we assume independence. 

* Why may it still be unrealistic in real life?
Many more factors influence weather outcomes and they may be linked. 
"""
