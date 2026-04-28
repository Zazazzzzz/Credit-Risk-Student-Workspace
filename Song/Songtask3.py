import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Prepare Data

df = pd.read_excel("EFM Asia daily.xlsx", header=6)
print(df.head(20))
print(df.columns)

df.columns = ["Date", "Price"]
df["Date"] = pd.to_datetime(df["Date"] , errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna()
df = df.sort_values("Date")
print(df.head(20))

df["Return"] = df["Price"].pct_change()
returns_df = df.dropna()
print(returns_df.head())

# Task 2: Plot the Return Distribution
plt.figure(figsize=(8, 5))
plt.hist(returns_df["Return"], bins=40)

mean_returns = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
plt.axvline(mean_returns, linestyle="--", color="red", label="Mean Return")
plt.axvline(quantile_5, linestyle="--", color="blue", label="5% Quantile")

plt.xlabel("Return")
plt.ylabel("Frequency")
plt.title("Return Distribution")
plt.legend()
plt.show()

# Task 3: Compute Risk Measures

VaR_95 = -quantile_5
print("Mean return:", mean_returns)
print("5% quantile:", quantile_5)
print("95% VaR:", VaR_95)

"""

Mean return: 0.00022780826250694812
5% quantile: -0.024808728524132517
95% VaR: 0.024808728524132517

"""

# Task 4: Interpretation Questions

"""

1. The return distribution is not symmetric and is slightly left-skewed.

Most returns are close to zero, but the left tail is longer, indicating larger negative returns.
There are also some extreme values, suggesting fat tails.

2. The mean return is close to zero, indicating that on average the daily return is small. 
However, the 95% VaR is around -2%, which means that in the worst 5% of cases, the loss can be relatively large.

This shows that while average outcomes are mild, extreme negative outcomes can be significant. Therefore, the mean return understates the true risk in the data.

3. A 95% VaR of X% means that, with 95% confidence, the daily loss will not exceed X%.

4. Extreme losses are rare, but they do occur. The left tail shows that large negative returns happen occasionally.

"""





