import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

probabilities1 = [0.1, 0.3, 0.6]   # Region 1
probabilities2 = [0.02, 0.7, 0.28] # Region 2

n = 100

sim_1 = np.random.choice(values, size=n, p=probabilities1)
sim_2 = np.random.choice(values, size=n, p=probabilities2)

df = pd.DataFrame({
    "Region1": sim_1,
    "Region2": sim_2
})

mapping = {1: "hail", 2: "rain", 3: "sunny"}
df_weather = df.replace(mapping)

print(df_weather.head(10))