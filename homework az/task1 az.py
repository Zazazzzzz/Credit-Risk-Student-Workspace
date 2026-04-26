
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 100
weather_options = [1, 2, 3]  # 1:hail, 2:rain, 3:sunny
probs1 = [0.1, 0.3, 0.6]
probs2 = [0.02, 0.7, 0.28]

region1_sim = np.random.choice(weather_options, size=N, p=probs1)
region2_sim = np.random.choice(weather_options, size=N, p=probs2)

# Step 2
df = pd.DataFrame({
    'Region1': region1_sim,
    'Region2': region2_sim
})

#Step 3
plt.figure(figsize=(10, 5))

plt.scatter(range(N), df['Region1'], label='Region 1', alpha=0.5)
plt.scatter(range(N), df['Region2'], label='Region 2', alpha=0.5)

plt.yticks([1, 2, 3], ['hail', 'rain', 'sunny']) # 将数字替换为文字标签
plt.xlabel('Simulation Day')
plt.ylabel('Weather Outcome')
plt.title('Scatter Plot of Weather Outcomes')
plt.legend()
plt.show()

#Step 4:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Region 1
ax1.hist(df['Region1'], bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, color='skyblue')
ax1.set_title('Region 1 Distribution')
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['hail', 'rain', 'sunny'])

# Region 2
ax2.hist(df['Region2'], bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, color='orange')
ax2.set_title('Region 2 Distribution')
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['hail', 'rain', 'sunny'])

plt.show()

#step 1 Q1 simulated value represent weather of region 1 or region2
#Q2 N=100 represent we simulated data 100 times just like simple size
#Q3 real data is fact. simulation is creating random data by using distribution in computer

#step2 Q1 weather in same day but different region
#Q2 weather in same region but different day

#step3 Q1 sunny
#Q2 Rain
#Q3 yes

#step 4 Q1 Region1
#Q2 region2
#Q3  I love good weather region1 p(sunny) is higher than region2 so region 1 is better
