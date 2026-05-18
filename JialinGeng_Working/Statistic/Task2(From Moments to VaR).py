# ---------------------------------------------------------------
# Script Name: From Moments to VaR
# Author: Hongyi Shen
#
# Description:
# This script demonstrates how we move from basic statistical
# summaries (mean, variance, correlation) to full distributional
# analysis (PDF, CDF), and finally to risk measures such as VaR.
#
# It provides both numerical output and visualizations to show
# how these concepts are connected in risk modeling.
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   # 从 scipy.stats 模块中导入 norm（正态分布对象）

np.random.seed(42)

# --------------------------------------------------
# Step 1: Generate data (two correlated variables)
# --------------------------------------------------
n = 1000
mu = [0, 0]
cov = [[1, 0.6],
       [0.6, 1]]  # covariance matrix (controls correlation)

data = np.random.multivariate_normal(mu, cov, size=n)
X = data[:, 0]
Y = data[:, 1]

# --------------------------------------------------
# Step 2: Mean, Variance, Correlation
# --------------------------------------------------
mean_X = np.mean(X)
var_X = np.var(X)
corr_XY = np.corrcoef(X, Y)[0, 1]

print("Mean (X):", round(mean_X, 3))   # round(..., 3) 保留 3 位小数
print("Variance (X):", round(var_X, 3))
print("Correlation (X,Y):", round(corr_XY, 3))

# plot
n = 1000

# --------------------------------------------------
# Case 1: Correlation = 0 (independent)
# --------------------------------------------------
mean = [0, 0]
cov_uncorr = [[1, 0],
              [0, 1]]

data_uncorr = np.random.multivariate_normal(mean, cov_uncorr, size=n)
x_uncorr = data_uncorr[:, 0]
y_uncorr = data_uncorr[:, 1]

# --------------------------------------------------
# Case 2: Correlation = 0.5
# --------------------------------------------------
rho = 0.5
cov_corr = [[1, rho],
            [rho, 1]]

data_corr = np.random.multivariate_normal(mean, cov_corr, size=n)
x_corr = data_corr[:, 0]
y_corr = data_corr[:, 1]

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)   # plt.subplots(1, 2) 表示创建 1 行 2 列的子图布局。figsize=(12, 6) 设置画布长宽。sharex=True, sharey=True 表示两个子图共享相同的 X 轴和 Y 轴刻度，方便直观对比。返回值 fig 是画布，axes 是包含两个子图对象的数组。

# Uncorrelated
axes[0].scatter(x_uncorr, y_uncorr, alpha=0.5)
axes[0].set_title("Correlation = 0")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid(alpha=0.3)

# Correlated
axes[1].scatter(x_corr, y_corr, alpha=0.5)
axes[1].set_title("Correlation = 0.5")
axes[1].set_xlabel("X")
axes[1].grid(alpha=0.3)

plt.suptitle("Mean = 0, Variance = 1: Effect of Correlation")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --------------------------------------------------
# Step 3: PDF (theoretical normal)
# --------------------------------------------------
x_grid = np.linspace(-4, 4, 1000)   # 在 -4 到 4 之间等间距地生成 1000 个点，作为 X 轴的网格线，用来绘制平滑的曲线。
pdf = norm.pdf(x_grid, loc=0, scale=1)

# --------------------------------------------------
# Step 4: CDF
# --------------------------------------------------
cdf = norm.cdf(x_grid, loc=0, scale=1)

# --------------------------------------------------
# Step 5: Quantile / VaR
# --------------------------------------------------
alpha = 0.05
var_alpha = norm.ppf(alpha)  # norm.ppf(alpha) 是 CDF 的逆函数（分位数函数/百分位点函数）。

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# --- Histogram (data) ---
axes[0].hist(X, bins=30, density=True, alpha=0.6, label="Simulated data")
axes[0].plot(x_grid, pdf, label="Normal PDF")
axes[0].set_title("Data vs PDF")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# --- CDF ---
axes[1].plot(x_grid, cdf)
axes[1].axhline(alpha, linestyle=":", label=f"{alpha}")   # axhline 绘制一条水平虚线（点状 :），高度为 alpha（0.05）。axhline：代表 Axes Horizontal Line（水平线）。alpha（即 0.05）：指定这条线画在纵坐标 y=0.05 的位置。linestyle=":"：设置线型为点线（Dotted line）。
axes[1].axvline(var_alpha, linestyle="--", label=f"VaR ≈ {var_alpha:.2f}")   # axvline 绘制一条垂直虚线（虚线 --），位置在临界点 var_alpha 处，用于标识 VaR 的位置。标签中动态格式化保留两位小数。axvline：代表 Axes Vertical Line（垂直线）。var_alpha（前面算出来的 −1.64）：指定这条线画在横坐标 x=−1.64 的位置。linestyle="--"：设置线型为破折号虚线（Dashed line）。f"VaR ≈ {var_alpha:.2f}"：利用 Python 的 f-string 语法，把计算结果保留两位小数填进标签里，显示为 "VaR ≈ -1.64"。
axes[1].set_title("CDF")
axes[1].set_xlabel("x")
axes[1].set_ylabel("F(x)")
axes[1].legend()   # 作用是：在第二个子图（中间那个 CDF 图）的角落里，弹出一个“图例方框”（Legend）。当你调用 axes[1].legend() 时，它就会在中间这个子图里巡视一圈，把所有带 label 的线条全部抓过来，然后打包画出一个小方框：
axes[1].grid(axis="y", alpha=0.3)   # 作用是：在中间这个 CDF 子图的背景上，画上一层淡淡的“水平参考网格线”。axis="y" 的意思是：“只绘制穿过 Y 轴刻度点的水平线。” （横向的线，垂直方向的竖线则不画）。

# --- PDF + VaR ---
axes[2].plot(x_grid, pdf)
axes[2].fill_between(
    x_grid[x_grid <= var_alpha],
    pdf[x_grid <= var_alpha],
    alpha=0.3
)                       # axes.fill_between(x, y1, y2=0, where=None, color=None, alpha=1.0) 为了让你彻底记住它，我们可以把它想象成一个“油漆工刷墙”的过程，每个参数都对应着油漆工的一个动作：x (刷子的横向扫过范围)：决定了从左到右，在哪个横坐标区间内刷油漆。y1 (油漆的顶部边界)：第一条曲线。默认情况下，油漆的上限高度被这条线死死压住。y2 (油漆的底部边界，默认是 0)：第二条线。如果不写，默认就是地面（X 轴）。油漆工会把 y1 到 y2 夹在中间的区域填满。where (施工允许条件)：一个布尔条件（开关）。只有当条件为 True 的地方，油漆工才会动刷子，否则就留白（比如你脚本里的 x_grid <= var_alpha）。color & alpha (油漆桶的颜色和稀释度)：调配油漆。color 决定什么颜色，alpha 决定加多少水（透明度），防止把背景的网格线盖死。
axes[2].axvline(var_alpha, linestyle="--", label=f"VaR ≈ {var_alpha:.2f}")
axes[2].set_title("PDF and VaR")
axes[2].set_xlabel("x")
axes[2].set_ylabel("Density")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle("From Moments to Distribution to Risk Measure")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()