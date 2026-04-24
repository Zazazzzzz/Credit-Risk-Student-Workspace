import numpy as np
hail = 1
rain = 2
sunny = 3
values = [hail, rain, sunny]
probabilities1 = [0.1, 0.3, 0.6]
probabilities2 = [0.02, 0.7, 0.28]
n = 100
res1 = np.random.choice(values, size=n, p=probabilities1)
res2 = np.random.choice(values, size=n, p=probabilities2)
sim_results = [res1, res2]
import pandas as pd
df = pd.DataFrame({
    'Region1': res1,
    'Region2': res2
})
print(df.head(50))
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(range(n), df['Region1'], label='Region 1', alpha=0.6, marker='o')
plt.scatter(range(n), df['Region2'], label='Region 2', alpha=0.6, marker='x')
plt.yticks([1, 2, 3], ['hail', 'rain', 'sunny'])
plt.xlabel('Simulation Day (N)')
plt.ylabel('Weather Outcome')
plt.title('Step 3: Scatter Plot of Weather Outcomes')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax1.hist(df['Region1'], bins=[0.5, 1.5, 2.5, 3.5])
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['hail', 'rain', 'sunny'])
ax1.set_ylabel('Frequency (Number of Days)')
ax2.hist(df['Region2'], bins=[0.5, 1.5, 2.5, 3.5])
ax2.set_title('Region 2: Weather Distribution')
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['hail', 'rain', 'sunny'])
plt.tight_layout()
plt.show()
counts1 = df['Region1'].value_counts().sort_index()
counts2 = df['Region2'].value_counts().sort_index()
freq1 = counts1 / n
freq2 = counts2 / n
print(f"Region 1 - Observed Probabilities:\n{freq1}")