import pandas as pd
import matplotlib.pyplot as plt

# --- Task 1: Data Preparation ---
# Load data and convert dates
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Return'] = df['Price'].pct_change()
returns_df = df.dropna()

# --- Task 2: Plot Distribution ---
# Calculate statistics
mean_val = returns_df["Return"].mean()
q5_val = returns_df["Return"].quantile(0.05)
# Plot histogram with vertical lines
returns_df["Return"].hist(bins=40, edgecolor='white')
plt.axvline(mean_val, color='red', label='Mean')
plt.axvline(q5_val, color='orange', label='5% Quantile')
plt.legend()
plt.show()

# --- Task 3: Compute Risk Measures ---
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
VaR_95 = -quantile_5
print(f"Mean: {mean_return:.4%}")
print(f"VaR (95%): {VaR_95:.4%}")
