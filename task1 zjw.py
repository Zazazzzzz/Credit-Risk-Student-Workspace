import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------
# Step 1: Set the distribution of returns
# --------------------------------------------------
# mu: 预期收益 (Expected Return)，代表分布的中心
# sigma: 波动率 (Volatility)，代表风险或不确定性
mu = 0.5
sigma = 2.0

# --------------------------------------------------
# Step 2: Choose a threshold on the original return scale
# --------------------------------------------------
# d: 临界阈值。在信用风险中，这可能代表违约点 (Default Point)
d = -1.0

# --------------------------------------------------
# Step 3: Standardize the threshold
# --------------------------------------------------
# 标准化公式: Z = (X - mu) / sigma
# 减去 mu 是为了平移中心到 0，除以 sigma 是为了缩放单位距离
z_d = (d - mu) / sigma

# --------------------------------------------------
# Step 4: Build the original density (X)
# --------------------------------------------------
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf_x = norm.pdf(x, loc=mu, scale=sigma)

# --------------------------------------------------
# Step 5: Build the standardized density (Z)
# --------------------------------------------------
z = np.linspace(-4, 4, 1000)
pdf_z = norm.pdf(z, loc=0, scale=1)

# --------------------------------------------------
# Step 6: Compute probabilities
# --------------------------------------------------
# 计算左侧尾部概率（违约概率/风险值）
p_left_original = norm.cdf(d, loc=mu, scale=sigma)
p_left_standardized = norm.cdf(z_d, loc=0, scale=1)

# --------------------------------------------------
# Step 7 & 8: Visualization
# --------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: 原始分布 X ~ N(0.5, 2^2)
ax1.plot(x, pdf_x, label=f'Original: N({mu}, {sigma}²)', color='blue', lw=2)
ax1.axvline(d, color='red', linestyle='--', label=f'Threshold d = {d}')
ax1.axvline(mu, color='gray', linestyle=':', label=f'Mean mu = {mu}')
# 填充左侧风险区域
x_fill = np.linspace(mu - 4*sigma, d, 100)
ax1.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='red', alpha=0.3, label='Risk Area')
ax1.set_title("Original Distribution of Returns (X)")
ax1.set_xlabel("Return Value")
ax1.set_ylabel("Probability Density")
ax1.legend()

# Plot 2: 标准化分布 Z ~ N(0, 1)
ax2.plot(z, pdf_z, label='Standardized: N(0, 1)', color='green', lw=2)
ax2.axvline(z_d, color='red', linestyle='--', label=f'Standardized z_d = {z_d:.2f}')
ax2.axvline(0, color='gray', linestyle=':', label='Mean = 0')
# 填充左侧风险区域
z_fill = np.linspace(-4, z_d, 100)
ax2.fill_between(z_fill, norm.pdf(z_fill, 0, 1), color='red', alpha=0.3, label='Risk Area (Z)')
ax2.set_title("Standardized Distribution (Z)")
ax2.set_xlabel("Standard Deviations from Mean")
ax2.set_ylabel("Probability Density")
ax2.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Step 9: Interpret the result
# --------------------------------------------------
print(f"--- 结果分析 ---")
print(f"原始阈值 (d):             {d}")
print(f"标准化阈值 (z_d):         {z_d:.4f}")
print(f"P(X <= d) 原始概率:       {p_left_original:.4%}")
print(f"P(Z <= z_d) 标准化概率:   {p_left_standardized:.4%}")
print(f"\n结论: 两个概率完全相等！")
print(f"标准化并没有改变风险的大小，只是改变了测量风险的刻度。")