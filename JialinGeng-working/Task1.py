import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 为了结果可重复
np.random.seed(42)

# 编码
hail = 1
rain = 2
sunny = 3

values = [hail, rain, sunny]

# 概率
probabilities1 = [0.1, 0.3, 0.6]
probabilities2 = [0.02, 0.7, 0.28]

# 模拟天数
N = 100

region1 = np.random.choice(values, size=N, p=probabilities1)
region2 = np.random.choice(values, size=N, p=probabilities2)

# 放在一个对象中
simulations = [region1, region2]



