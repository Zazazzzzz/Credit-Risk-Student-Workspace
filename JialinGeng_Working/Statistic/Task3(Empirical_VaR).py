# ---------------------------------------------------------------
# Script Name: Analysis of Daily Returns and 95% Value at Risk (VaR)
# Author: Hongyi Shen
#
# Description:
# - Data cleaning and preprocessing
# - Return calculation
# - Distribution visualization
# - Risk measurement via VaR (95%)
# --------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# Load raw data
# --------------------------------------------
file_path = "./Task3(EFM Asia daily).xlsx"
raw = pd.read_excel(file_path, engine="openpyxl", header=None)

# --------------------------------------------
# Extract relevant data
# --------------------------------------------
data = raw.iloc[7:, :2].copy()   # raw.iloc[7:, :2]：跳过前 7 行（可能是一些说明文字），只提取第 8 行往后的前 2 列（日期和价格）。
data.columns = ["Date", "Price"]

# Convert types
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

# Clean data
data = data.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
# 这行代码采用了链式调用（Method Chaining），一口气连续做了三件事：1：.dropna(subset=["Date", "Price"])：清除垃圾：扔掉 Date 列或者 Price 列中包含空值（即刚才转换出来的 NaT 和 NaN）的整行数据。确保留下的每一行都是既有日期、又有价格的有效数据。
# 2: .sort_values("Date")：按时间排序：根据 Date 列，把数据按照时间从早到晚（升序）重新排列。
# 3: .reset_index(drop=True)：重置行索引：因为经历了跳过前 7 行、剔除空值、重新排序，此时表格左侧的行索引（0, 1, 2...）早就乱套了（可能第一行变成了 58，第二行变成了 12）。这个操作把旧的、乱掉的索引扔掉（drop=True），重新从 0, 1, 2, 3... 开始挨个按顺序打上崭新的行标签，让表格看起来整整齐齐。
# DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

# --------------------------------------------
# Compute returns
# --------------------------------------------
data["Return"] = data["Price"].pct_change()   # .pct_change()：这是 Pandas 的内置函数，全称是 Percentage Change（百分比变化）。它会计算当前行与前一行之间的变动比例。
returns_df = data.dropna(subset=["Return"]).copy()

# --------------------------------------------
# Compute statistics
# --------------------------------------------
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)   # .quantile(0.05)：这是计算分位数（Quantile）的函数。0.05 代表 5% 分位数。
# The 5% quantile of returns:
# This is the value such that 5% of returns are BELOW it.
# Since returns can be negative, this typically captures the left tail (losses).

VaR_95 = -quantile_5
# Value at Risk (VaR) at 95%
# Important: VaR is defined in terms of LOSS, not return.
# Returns:
#   positive → gain
#   negative → loss
# VaR convention:
#   reported as a positive number representing a potential LOSS
# The 5% quantile is usually negative (e.g., -2.5%),
# meaning: in the worst 5% of cases, returns are worse than -2.5%.
# To express this as a loss (positive number), we take the negative.
# "With 95% confidence, the loss will not exceed X% over the given time horizon."

# --------------------------------------------
# Plot
# --------------------------------------------
plt.figure(figsize=(9, 5))   # 作用：初始化一张图表（画布）。参数：figsize=(9, 5) 表示画布的尺寸，宽 9 英寸，高 5 英寸。
plt.hist(returns_df["Return"], bins=40)

plt.axvline(mean_return, linestyle="--", linewidth=2,
            label=f"Mean = {mean_return:.2%}")

plt.axvline(quantile_5, linestyle="--", linewidth=2,
            label=f"5% Quantile = {quantile_5:.2%}")

plt.axvline(-VaR_95, linestyle=":", linewidth=2,
            label=f"VaR 95% = {VaR_95:.2%}")

plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.title("EFM Asia Daily Return Distribution and 95% VaR")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------------------------
# Print results
# --------------------------------------------
print("Mean daily return:", mean_return)
print("5% return quantile:", quantile_5)
print("95% VaR:", VaR_95)
