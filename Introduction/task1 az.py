import numpy as np
import random
import numpy as np

N = 100
weather_options = [1, 2, 3]

probs_1 = [0.10, 0.30, 0.60]
region_1_sim = np.random.choice(weather_options, size=N, p=probs_1)

probs_2 = [0.02, 0.70, 0.28]
region_2_sim = np.random.choice(weather_options, size=N, p=probs_2)

simulation_results = {
    "Region_1": region_1_sim,
    "Region_2": region_2_sim
}

print("Region 1 前10天:", simulation_results["Region_1"][:10])
print("Region 2 前10天:", simulation_results["Region_2"][:10])