import numpy as np

#setting
hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

prob1 = [0.1, 0.3, 0.6]
prob2 = [0.02, 0.7, 0.28]

# step1: stimulate weather outcomes
region1 = np.random.choice(values, size=100, p=prob1)
region2 = np.random.choice(values, size=100, p=prob2)
# store the outcomes
results = np.column_stack((region1, region2))

print(results[:10])

#Questions:
#* What does one simulated value represent?
#  It represents the weather of the specific day, if the value is 1,
#      then the weather is hail.

#* What does (N=100) represent in this context?
#  It means that the weather conditions of 100 days are stimulated.

#* Why is this a simulation rather than real data?
#  Because stimulations are more ideal for tests.

