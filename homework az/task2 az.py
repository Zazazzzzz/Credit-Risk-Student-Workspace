import matplotlib.pyplot as plt
import numpy as np
n = 1000
mean = [0 , 0]
cov_a = [[1 , 0],
         [0 , 1]]
cov_b = [[1 , 0.5],
         [0.5 , 1]]
data_a = np.random.multivariate_normal(mean=mean, cov=cov_a, size=n)
data_b = np.random.multivariate_normal(mean=mean, cov=cov_b, size=n)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5), sharex=True, sharey=True)

ax1.scatter(data_a[:, 0], data_a[:, 1], alpha=0.5, color='blue', s=10)
ax1.set_title('A correlation = 0')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True)

ax2.scatter(data_b[:, 0], data_b[:, 1], alpha=0.5, color='red', s=10)
ax2.set_title('B correlation = 0.5')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True)

plt.show()

#Q1: it looks like a ball; NO, X and Y independent
#Q2: it looks like compressed; as X increased Y increased
#Q3:It is the linear association strength between the variables that changes.
#Correlation changes the joint behavior of the variables.