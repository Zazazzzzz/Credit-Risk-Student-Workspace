import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Load and Prepare Data

# load dataset
df = pd.read_excel("EFM Asia daily.xlsx", header=6)
df = df.rename(columns={
    "EFM ASIA Standard (Large+Mid Cap)": "Price"
})
# Identify the price column.
print(df.columns)
print(df.head())


# Convert the date column to datetime.
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Date", "Price"])

# Sort the data by date.
df = df.sort_values("Date")
print(df.head())

# - Compute daily returns:
#     Return_t = (Price_t / Price_{t-1}) - 1
df["returns_df"] = df["Price"] / df["Price"].shift(1) -1 # also can be written as df["returns_df"] = df["Price"].pct_change()
# .shift(1): use the price of the previous day

returns_df = df.dropna(subset=["returns_df"])
returns_df = returns_df[["Date", "returns_df"]]
# save the calculated returns into a DataFrame
print(returns_df.head())


# Task 2: Plot the Return Distribution
returns = returns_df["returns_df"]
mean_return = float(returns.mean())
quantile_5 = float(returns.quantile(0.05))

plt.figure(figsize=(10, 6))
plt.hist(returns_df["returns_df"], bins=50, edgecolor="black", alpha=0.5)

plt.axvline(
    mean_return,
    linestyle="--",
    label=f"Mean = {mean_return:.4f}"
)

plt.axvline(quantile_5, linestyle="--", linewidth=2,
            label=f"5% Quantile (VaR) = {quantile_5:.4f}")

plt.xlabel("Return")
plt.ylabel("Frequency")
plt.title("Distribution of Daily Returns")

plt.legend()
plt.show()


# Task 3: Compute Risk Measures
returns_df = returns_df.rename(columns={"returns_df": "Return"})

mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
VaR_95 = -quantile_5  #the number should be positive

print(f"Mean Return: {mean_return:.4f}")
print(f"5% Quantile: {quantile_5:.4f}")
print(f"95% VaR: {VaR_95:.4f}")

'''
Mean Return: 0.0002
The mean return of 0.0002 indicates that the average daily return is approximately 0.02%, 
which is close to zero and typical for financial return series.

5% Quantile: -0.0248
The 5% quantile of -0.0248 means that in the worst 5% of cases, 
the daily return is at or below -2.48%.

95% VaR: 0.0248
The 95% Value at Risk (VaR) of 0.0248 implies that, with 95% confidence, 
the maximum expected daily loss will not exceed 2.48%.
'''

# Task 4: Interpretation Questions

# Answer the following:

'''
1. Distribution shape
   - Is the return distribution symmetric?
     The return distribution is not perfectly symmetric.
   
   - Do you observe fat tails or extreme values?
     It is typically slightly skewed and shows fat tails, meaning there are more extreme values than would be expected under a normal distribution.
     In particular, the left tail (loss side) often appears heavier, indicating the presence of extreme negative returns.
   

2. Mean vs risk
   - Compare the mean return to the VaR.
     The mean return (≈ 0.02%) is very small compared to the VaR (≈ 2.48%).
     
   - What does this tell you about average vs extreme outcomes?
      This shows that while the average outcome is close to zero, the potential losses in extreme scenarios are much larger.

3. VaR interpretation
   - Explain in words:
     “What does a 95% VaR of X% mean?”
     
     With 95% confidence, the daily loss will not exceed 2.48%.
     There is a 5% probability that the loss will be greater than 2.48%.

4. Tail behavior
   - Look at the left tail (loss side).
     Extreme losses (in the left tail) are relatively rare, but they do occur.
     
   - Are extreme losses frequent or rare?
     The presence of fat tails suggests that these extreme losses happen more frequently than predicted by a normal distribution.
'''
