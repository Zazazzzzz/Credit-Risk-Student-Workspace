import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma


shape = 2.0
scale = 3.0
size = 1000

gamma_data = np.random.gamma(shape=shape, scale=scale, size=size)

plt.hist(gamma_data, bins=30, density=True)
plt.title("Random Gamma Distr")
plt.xlabel("Value")
plt.ylabel("Density")

x = np.linspace(min(gamma_data), max(gamma_data))
y = gamma.pdf(x, a=shape, scale=scale)

plt.plot(x, y, linewidth=2, label="PDF")

plt.title("Gamma Distr with PDF")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()


#Task 3
file_p = r"C:\Users\jurh0\PycharmProjects\CreditRisk\Credit-Risk-Student-Workspace\Statistic\EFM Asia daily.xlsx"
df = pd.read_excel(file_p)

print(df.columns)
print(df.head(n=15))

df= pd.read_excel(file_p,skiprows=6)

print(df.head(n=15))
df.columns=["Date","Price"]

df["Date"] = pd.to_datetime(df["Date"],errors='coerce')
df=df.dropna(subset=["Date"])

df = df.sort_values(by="Date")

df["Price"]=pd.to_numeric(df["Price"],errors='coerce')
df=df.dropna(subset=["Price"])

df['Return'] = df["Price"].pct_change()

returns_df = df[["Date", 'Return']].dropna().reset_index(drop=True)

print(returns_df.head(n=15))

# Task 2: Plot the Return Distribution
returns=returns_df["Return"]

plt.figure()
plt.hist(returns,bins=40)

mean_return=returns.mean()
var_95=returns.quantile(0.05)

plt.axvline(mean_return,linestyle="--", label=f"Mean:{mean_return:.2f}",color="r")
plt.axvline(var_95,linestyle="--",label=f"5% Quantile: {var_95:.2f}",color="g")

plt.xlabel("return")
plt.ylabel("Freq.")
plt.title("Histogram of Returns")

plt.legend()
plt.show()

#Compute Risk measures
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
VaR_95 = -quantile_5

print(f"mean:{round(mean_return,4)}, quantile:{round(quantile_5,4)}, VaR:{round(VaR_95,4)}")

#VaR is conventionally reported as a positive number
#representing the size of a loss, not the return itself

#4 Interpretation questions


# 1. Distribution shape
#    - Is the return distribution symmetric?
#not perfectly, slightly left skewed (longer tail on the left)
#    - Do you observe fat tails or extreme values?
# we have fat tails, still relevant bulk of observation at +/- 6%

# 2. Mean vs risk
#    - Compare the mean return to the VaR.
#mean : 0 VaR: 2%
#    - What does this tell you about average vs extreme outcomes?
#On average returns tend to be zero, but on bad days one can lose 2 %

# Mean does not represent the downside risk, the risk lies in the tails

# 3. VaR interpretation
#    - Explain in words:
#      “What does a 95% VaR of X% mean?”
# With a confidence of 95%, the loss will not exceed 2%

# 4. Tail behavior
#    - Look at the left tail (loss side)
#    - Are extreme losses frequent or rare?
#Extreme losses (up to 6% are rare but do occur)

#Even though the average return is zero, negative skeweness and fat tails,
#we have a meaningful downside risk, with sometimes extremes
#supassing the 95% VaR threshold..

