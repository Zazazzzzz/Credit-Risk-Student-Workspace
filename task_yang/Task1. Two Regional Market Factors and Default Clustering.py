"""
The goal is to simulate the number of defaults in this portfolio
using a Gaussian threshold model with two correlated regional
systematic factors.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

#   Step 1:
#   Load the two regional equity index datasets from Excel files.
df_asia = pd.read_excel("EM Asia monthly.xls", header = 6)
df_asia = df_asia.rename (columns={
    "EM ASIA Standard (Large+Mid Cap)":"Price"
})
df_asia["Date"] = pd.to_datetime(df_asia["Date"], errors="coerce")
df_asia["Price"] = pd.to_numeric(df_asia["Price"], errors="coerce")
df_asia = df_asia.dropna(subset=["Date", "Price"])
df_asia = df_asia.sort_values("Date")
print(df_asia.head())

df_europe = pd.read_excel("EM Europe and Middle East monthly.xls", header = 6)
df_europe = df_europe.rename(columns={
    "EM EUROPE & MIDDLE EAST Standard (Large+Mid Cap)":"Price"
})
df_europe["Date"] = pd.to_datetime(df_europe["Date"], errors="coerce")
df_europe["Price"] = pd.to_numeric(df_europe["Price"], errors="coerce")
df_europe = df_europe.dropna(subset=["Date", "Price"])
df_europe = df_europe.sort_values("Date")
print(df_europe.head())

#   Step 2:
#   Clean the price data and compute monthly log returns for both
#   index series.
df_asia["Asia_Return"] = df_asia["Price"].pct_change()
df_europe["Europe_Return"] = df_europe["Price"].pct_change()

df_asia_returns = df_asia.dropna(subset=["Asia_Return"])
asia_returns = df_asia_returns[["Date", "Asia_Return"]]
print(asia_returns.head())
df_europe_returns = df_europe.dropna(subset=["Europe_Return"])
europe_returns = df_europe_returns[["Date", "Europe_Return"]]
print(europe_returns.head())

#   Step 3:
#   Merge the two return series by date so that only common monthly
#   observations are used.
df_returns = pd.merge(
    asia_returns,
    europe_returns,
    on="Date",
    how="inner"
)
print(df_returns.head())
print(df_returns.tail())
print(df_returns.shape)

# Step 4:
# Estimate the empirical correlation between the two
# regional market return series
empirical_corr = df_returns["Asia_Return"].corr(df_returns["Europe_Return"])

print("Empirical correlation between Asia and Europe & Middle East returns:")
print(empirical_corr)

# Step 5:
#   Simulate two standard normal regional market factors:
#
#       F_asia      ~ N(0, 1)
#       F_europe_me ~ N(0, 1)
#
#   The two factors should be simulated with correlation equal to the
#   empirical correlation estimated from the historical market data.

n_sims = 10000

# Correlation matrix
corr_matrix = np.array([
    [1.0, empirical_corr],
    [empirical_corr, 1.0]
])

# Mean vector
mean_vector = np.array([0.0, 0.0])

# Simulate correlated standard normal factors for asia and europe in 2000 simulations
factors = np.random.multivariate_normal(
    mean=mean_vector,
    cov=corr_matrix,
    size=n_sims
)

df_factors = pd.DataFrame(
    factors,
    columns=["F_asia", "F_europe"]
)

simulated_corr = df_factors["F_asia"].corr(df_factors["F_europe"])

print("Simulated correlation:")
print(simulated_corr)


#   Step 6:
#   Assign each company to its own regional factor:
#
#       Companies 1-5:   Asia factor:
#           column*1, line*10000 (because they are in the same region)
#       Companies 6-10:  Europe/Middle East factor: column*1, line*10000

F_asia = df_factors["F_asia"].values.reshape(-1,1)
F_asia = np.repeat(F_asia, 5, axis=1)

F_europe = df_factors["F_europe"].values.reshape(-1, 1)
F_europe = np.repeat(F_europe, 5, axis=1)
# .values: converts the selected pandas Series into a NumPy array.
# .reshape(-1, 1) reshapes the array into a column vector.
#    -1 means Python automatically determines the number of rows.
#    1 means the reshaped array should have one column.
F_region = np.concatenate([
    F_asia, F_europe
],
    axis=1
)

# print(F_region.shape) - (10000,10)


#   Step 7:
#   For each company, simulate the creditworthiness variable:
#       Number of companies:        10
#       Asia companies:             5
#       Europe/ME companies:        5
#       Default probability:        5%
#       Systematic sensitivity rho: 0.5
#       Number of simulations:      10,000

n_companies = 10
n_asia = 5
n_europe = 5
p = 0.05
rho = 0.5


# Company-specific shocks
epsilon = np.random.normal(
    loc=0,
    scale=1,
    size=(n_sims, n_companies)
)

X = rho * F_region + np.sqrt(1 - rho**2) * epsilon
# print(X.shape) - (10000,10)


#   Step 8:
#   Convert the default probability into a default threshold:
#
#       default threshold = Phi^{-1}(p)
#
#   A company defaults if:
#
#       X_i <= Phi^{-1}(p)
default_threshold = norm.ppf(p)
print("Default threshold:")
print(default_threshold)

default_indicators = (X <= default_threshold).astype(int)
# print(default_indicators.shape) - (10000,10)


#   Step 9:
#   Count the number of defaulted companies in each simulation.
number_of_defaults = default_indicators.sum(axis=1)
print("First 10 simulated numbers of defaults:")
print(number_of_defaults[:10])


#   Step 10:
#   Plot a histogram of the simulated number of defaults.
plt.figure(figsize=(10, 5))
plt.hist(number_of_defaults, bins=np.arange(0, n_companies + 2) - 0.5, edgecolor="black")

plt.xlabel("Number of defaulted companies")
plt.ylabel("Frequency")
plt.title("Histogram of Simulated Number of Defaults")
plt.xticks(range(0, n_companies + 1))
plt.grid(True)

plt.show()


