import pandas as pd
import matplotlib.pyplot as plt
import os


# Task 3: Return Distribution and Value at Risk (VaR)

# step 1: load data ================================================================

folder = os.path.dirname(__file__)
file_path = os.path.join(folder, "../Statistic/EFM Asia daily.xlsx")

# The useful table starts after the first 6 rows in the Excel file.
data = pd.read_excel(file_path, skiprows=6, engine="openpyxl")

print("Original data:")
print(data.head())

# Rename the two columns to make them easier to use.
data.columns = ["Date", "Price"]

# Convert date and price columns.
# Some rows at the bottom are text, so errors="coerce" changes them to NaN.
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")     #笔记：errors="coerce"代表如果有错误变成NaN
data["Price"] = data["Price"].astype(str)
data["Price"] = data["Price"].str.replace(",", "")#zhege这个地方不去逗号会提示错误！！
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

# Keep only rows with a real date and price.
data = data.dropna()

# Sort by date.
data = data.sort_values("Date") #这里删掉了NaN这种类似行

print("\nClean data:")
print(data.head())


# step 2: calculate daily returns ==================================================

data["Return"] = data["Price"] / data["Price"].shift(1) - 1 #shift 这里代表向下移动一格(上一期）就是变化之后第一行会除nan这里得注意清洗

# Store the result in a DataFrame called returns_df.
returns_df = data[["Date", "Return"]]
returns_df = returns_df.dropna()

print("\nDaily returns:")
print(returns_df.head())


# step 3: compute risk measures ====================================================
#————这部分后续再复习看一下————
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
VaR_95 = -quantile_5

print("\nRisk measures:")
print("Mean return:", mean_return)
print("5% quantile:", quantile_5)
print("95% VaR:", VaR_95)

print("\nRisk measures in percent:")
print("Mean return: {:.2%}".format(mean_return))
print("5% quantile: {:.2%}".format(quantile_5))
print("95% VaR: {:.2%}".format(VaR_95))


# step 4: plot return distribution ================================================

plt.figure(figsize=(10, 5))

plt.hist(returns_df["Return"], bins=40)

plt.axvline(mean_return, color="tab:blue", linestyle="--",
            label="Mean = {:.2%}".format(mean_return))
plt.axvline(quantile_5, color="tab:blue", linestyle="--",
            label="5% Quantile = {:.2%}".format(quantile_5))
plt.axvline(VaR_95, color="tab:blue", linestyle=":",
            label="VaR 95% = {:.2%}".format(VaR_95))

plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.title("EFM Asia Daily Return Distribution and 95% VaR")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()


# step 5: interpretation questions =================================================

print("\nInterpretation:")
print("1. The distribution is not perfectly symmetric. There are some extreme values on both sides.")
print("2. The mean return is very small, but the VaR is much larger. This means average return is calm,")
print("   but bad days can still create much bigger losses.")
print("3. A 95% VaR of {:.2%} means that on the worst 5% of days, the loss can be more than about {:.2%}.".format(VaR_95, VaR_95))
print("4. Extreme losses are not frequent. They are rare, but they are important for risk management.")
