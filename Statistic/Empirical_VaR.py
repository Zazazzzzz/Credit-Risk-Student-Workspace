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
file_path = "Statistic\EFM Asia daily.xlsx"
raw = pd.read_excel(file_path, engine="openpyxl", header=None)

# --------------------------------------------
# Extract relevant data
# --------------------------------------------
data = raw.iloc[7:, :2].copy()
data.columns = ["Date", "Price"]

# Convert types
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

# Clean data
data = data.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

# --------------------------------------------
# Compute returns
# --------------------------------------------
data["Return"] = data["Price"].pct_change()
returns_df = data.dropna(subset=["Return"]).copy()

# --------------------------------------------
# Compute statistics
# --------------------------------------------
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
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
plt.figure(figsize=(9, 5))
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
