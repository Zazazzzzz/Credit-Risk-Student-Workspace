import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #这个是导入normal distribution

np.random.seed(42)

N = 2000     # number of simulations
m = 1000     # number of obligors（1000个债务人）
p = 0.01     # default probability
rho = 0.25
d = norm.ppf(p)  #计算 threshold (d=Φ−1(p)) normal distribution的反函数
print("Default probability p:", p)
print("Default threshold d:", d)

#scenario 1 : independent default

independent_defaults = np.random.binomial(      #Binomial(n=1,p)
    n=1,
    p=p,
    size=(N, m)                      #N=2000，m=1000,生成2000*1000的矩阵
)
independent_counts = independent_defaults.sum(axis=1)   #sum(axis=1)的意思是对每一行求和
#scenario 2 : Correlated defaults with one systematic factor
F = np.random.normal(
    loc=0,
    scale=1,
    size=(N, 1)
)

epsilon = np.random.normal(
    loc=0,
    scale=1,
    size=(N, m)
)

X = rho * F + np.sqrt(1 - rho ** 2) * epsilon
correlated_defaults = (X <= d).astype(int)#判断是否违约
correlated_counts = correlated_defaults.sum(axis=1)

print("\nIndependent defaults:")
print("Mean number of defaults:", independent_counts.mean())
print("Standard deviation:", independent_counts.std())

print("\nCorrelated defaults:")
print("Mean number of defaults:", correlated_counts.mean())
print("Standard deviation:", correlated_counts.std())
# Plot histograms
max_count = max(independent_counts.max(), correlated_counts.max())

bins = np.arange(-0.5, max_count + 1.5, 1)

plt.figure(figsize=(10, 5))

plt.hist(
    independent_counts,
    bins=bins,
    alpha=0.6,
    label="Independent defaults"
)

plt.hist(
    correlated_counts,
    bins=bins,
    alpha=0.6,
    label="Correlated defaults"
)

plt.xlabel("Number of defaults")
plt.ylabel("Frequency")
plt.title("Default Distribution: Independent vs Correlated Defaults")
plt.legend()
plt.show()



