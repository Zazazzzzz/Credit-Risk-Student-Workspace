import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist

#Load the two regional equity index datasets from Excel files.   Step 1
asia_file = "EM Asia monthly.xls"
EM_file = "EM Europe and Middle East monthly.xls"
#clean the data   Step  2
asia_data = pd.read_excel(
    asia_file,
    sheet_name="History Index",
    skiprows=6
)

EM_data = pd.read_excel(
    EM_file,
    sheet_name="History Index",
    skiprows=6
)

print(asia_data.head())
print(EM_data.head())
asia_data = asia_data.iloc[:, :2].copy()
EM_data = EM_data.iloc[:, :2].copy()
#rename
asia_data.columns = ["Date", "Asia_Price"]
EM_data.columns = ["Date", "EM_Price"]
#wash the data
asia_data["Date"] = pd.to_datetime(asia_data["Date"], errors="coerce")
EM_data["Date"] = pd.to_datetime(EM_data["Date"], errors="coerce")

asia_data["Asia_Price"] = pd.to_numeric(asia_data["Asia_Price"], errors="coerce")
EM_data["EM_Price"] = pd.to_numeric(EM_data["EM_Price"], errors="coerce")

asia_data = asia_data.dropna()
EM_data = EM_data.dropna()

asia_data = asia_data.sort_values("Date")
EM_data = EM_data.sort_values("Date")
#  compute monthly log returns   (Step 2)
asia_data["r_asia"] = np.log(
    asia_data["Asia_Price"] / asia_data["Asia_Price"].shift(1)
)

EM_data["r_EM"] = np.log(
    EM_data["EM_Price"] / EM_data["EM_Price"].shift(1)
)
# Remove the first row because there is no previous price to calculate the return
asia_returns = asia_data.dropna(subset=["r_asia"])
EM_returns = EM_data.dropna(subset=["r_EM"])

print(asia_returns.head())
print(EM_returns.head())
#STEP 3
returns = pd.merge(
    asia_returns[["Date", "r_asia"]],
    EM_returns[["Date", "r_EM"]],
    on="Date",
    how="inner"
)

print(returns.head())
print("Number of common observations:", len(returns))

#Step 4   Estimate the empirical correlation between the two regional market return series.
market_corr = returns["r_asia"].corr(returns["r_EM"])

print("Empirical correlation between Asia and Europe/ME returns:")
print(market_corr)
#     Step 5: Simulate two correlated standard normal regional market factors
num_simulations = 10000

mean = [0, 0]   # Mean vector of the two standard normal factors

# Correlation matrix of the two regional market factors
corr_matrix = [
    [1, market_corr],
    [market_corr, 1]
]#because F_asia 和自己完全相关，所以是 1

rng = np.random.default_rng(42)  # Set random seed for reproducible results

# Simulate correlated regional market factors
factors = rng.multivariate_normal(
    mean=mean,
    cov=corr_matrix,
    size=num_simulations
)

F_asia = factors[:, 0]
F_EM = factors[:, 1]   #把第一列取出来作为亚洲因子，把第二列取出来作为欧洲/中东因子

print(F_asia[:5])
print(F_EM[:5])
#  step  6
num_companies = 10
num_asia = 5
num_EM = 5

rho = 0.5

# Simulate company-specific shocks
epsilon = rng.standard_normal(size=(num_simulations, num_companies))

# Create an empty matrix to store creditworthiness variables
X = np.zeros((num_simulations, num_companies))

# Companies 1-5 belong to Asia
X[:, 0:5] = (
    rho * F_asia[:, None]
    + np.sqrt(1 - rho ** 2) * epsilon[:, 0:5]
)

# Companies 6-10 belong to Europe / Middle East
X[:, 5:10] = (
    rho * F_EM[:, None]
    + np.sqrt(1 - rho ** 2) * epsilon[:, 5:10]
)

print(X[:5])

default_probability = 0.05

default_threshold = NormalDist().inv_cdf(default_probability)

print("Default threshold:")
print(default_threshold)
#  Step 9

default_indicator = X <= default_threshold

number_of_defaults = default_indicator.sum(axis=1)

print("Average number of defaults:")
print(number_of_defaults.mean())

print("Default count frequency:")
for k in range(num_companies + 1):
    print(f"{k} defaults: {(number_of_defaults == k).sum()}")

# Step 10

plt.figure(figsize=(8, 5))

plt.hist(
    number_of_defaults,
    bins=np.arange(-0.5, num_companies + 1.5, 1),
    edgecolor="black"
)

plt.xticks(range(num_companies + 1))
plt.xlabel("Number of defaults")
plt.ylabel("Frequency")
plt.title("Simulated Number of Defaults in the Portfolio")

plt.show()











