# ============================================================
# Task: Two Industry Factors and Default Clustering
# ============================================================
#
# This task illustrates how dependence between financial market
# variables can be connected to credit portfolio risk.
#
# We use two monthly regional equity index series:
#
#   1. Emerging Markets Asia
#   2. Emerging Markets Europe and Middle East
#
# The two market return series are used to estimate their empirical
# correlation. In the credit risk simulation, we consider a portfolio
# of 10 companies:
#
#   - 5 companies belong to the Asia market
#   - 5 companies belong to the Europe/Middle East market
#
# The two regional market factors are simulated as correlated standard
# normal variables. Their correlation is estimated from the historical
# market return data.
#
# Each company is exposed only to the market factor of its own region.
#
# The creditworthiness variable of company i is:
#
#   X_i = rho * F_industry(i) + sqrt(1 - rho^2) * epsilon_i
#
# where:
#
#   X_i             = creditworthiness variable of company i
#   F_industry(i)  = systematic factor of company i's own region
#   epsilon_i      = company-specific shock
#   rho            = sensitivity to systematic risk
#
# A company defaults if:
#
#   X_i <= Phi^{-1}(p)
#
# where p is the default probability.
#
# Assumptions:
#
#   Number of companies:        10
#   Asia companies:             5
#   Europe/ME companies:        5
#   Default probability:        5%
#   Systematic sensitivity rho: 0.5
#   Number of simulations:      10,000
#
# The final output is a histogram showing the simulated distribution
# of the number of defaulted companies.
# ============================================================


# ============================================================
# 1. Import packages
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# 2. Load regional equity index data
# ============================================================

current_dir = os.getcwd()
direct_credit = os.path.join(current_dir, "Credit Risk")

index_as = pd.read_excel(
    os.path.join(direct_credit, "EM Asia monthly.xls"),
    header=6
)

index_me = pd.read_excel(
    os.path.join(direct_credit, "EM Europe and Middle East monthly.xls"),
    header=6
)


# ============================================================
# 3. Clean price data and compute monthly log returns
# ============================================================

def prepare_price_df(df, price_col):
    df = df.copy()
    df = df.dropna()

    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")

    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    df = df.sort_values("Date").reset_index(drop=True)
    df["log_return"] = np.log(df[price_col]).diff()

    return df.dropna(subset=["log_return"])


df_index_as = prepare_price_df(index_as, index_as.columns[1])
df_index_me = prepare_price_df(index_me, index_me.columns[1])


# ============================================================
# 4. Match the two return series by date
# ============================================================

returns = pd.merge(
    df_index_as[["Date", "log_return"]],
    df_index_me[["Date", "log_return"]],
    on="Date",
    how="inner",
    suffixes=("_asia", "_me")
)

returns = returns.dropna().reset_index(drop=True)

print("First observations of the matched return data:")
print(returns.head())


# ============================================================
# 5. Estimate correlation between the two regional markets
# ============================================================
# This correlation is used to simulate two correlated regional
# systematic factors.
#
# Important:
# This is not the default correlation. It is the correlation between
# the two market factors.

corr_matrix = returns[["log_return_asia", "log_return_me"]].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

market_corr = corr_matrix.loc["log_return_asia", "log_return_me"]

print(
    f"\nEstimated correlation between Emerging Markets Asia and "
    f"Emerging Markets Europe/Middle East: {market_corr:.4f}"
)


# Scatter plot of the two historical return series
plt.figure(figsize=(7, 5))
plt.scatter(
    returns["log_return_asia"],
    returns["log_return_me"],
    alpha=0.6
)
plt.xlabel("Emerging Markets Asia Monthly Log Return")
plt.ylabel("Emerging Markets Europe/Middle East Monthly Log Return")
plt.title("Historical Dependence Between Two Regional Market Returns")
plt.grid(True)
plt.show()


# ============================================================
# 6. Set credit portfolio assumptions
# ============================================================

np.random.seed(42)

N_sim = 10000      # number of simulations
m = 10             # total number of companies
p = 0.05           # default probability of each company
rho = 0.5          # sensitivity to systematic risk

n_asia = 5         # number of Asia companies
n_me = 5           # number of Europe/Middle East companies

# Industry label for each company
company_region = np.array(
    ["Asia"] * n_asia +
    ["Europe_ME"] * n_me
)


# ============================================================
# 7. Simulate two correlated regional market factors
# ============================================================
# We simulate:
#
#   F_asia      ~ N(0, 1)
#   F_europe_me ~ N(0, 1)
#
# with correlation equal to the empirical market correlation.
#
# Therefore, each factor is standard normal marginally, but the two
# factors are not independent.

market_corr_matrix = np.array([
    [1.0, market_corr],
    [market_corr, 1.0]
])

market_factors = np.random.multivariate_normal(
    mean=[0, 0],
    cov=market_corr_matrix,
    size=N_sim
)

F_asia_sim = market_factors[:, 0]
F_me_sim = market_factors[:, 1]

print("\nCorrelation of simulated market factors:")
print(np.corrcoef(F_asia_sim, F_me_sim))


# Optional: plot simulated market factors
plt.figure(figsize=(7, 5))
plt.scatter(F_asia_sim, F_me_sim, alpha=0.2)
plt.xlabel("Simulated Asia Market Factor")
plt.ylabel("Simulated Europe/Middle East Market Factor")
plt.title("Simulated Correlated Regional Market Factors")
plt.grid(True)
plt.show()


# ============================================================
# 8. Assign each company to its own regional factor
# ============================================================
# The first 5 companies are affected by the Asia factor.
# The last 5 companies are affected by the Europe/Middle East factor.
#
# Each company receives only one systematic factor:
#
#   Asia company:       F_i = F_asia
#   Europe/ME company:  F_i = F_europe_me
#
# However, defaults across the two groups can still be dependent
# because F_asia and F_europe_me are correlated.

F_company = np.where(
    company_region.reshape(1, -1) == "Asia",
    F_asia_sim.reshape(-1, 1),
    F_me_sim.reshape(-1, 1)
)


# ============================================================
# 9. Simulate defaults using a Gaussian threshold model
# ============================================================
# The creditworthiness variable of company i is:
#
#   X_i = rho * F_i + sqrt(1 - rho^2) * epsilon_i
#
# where F_i is the regional systematic factor assigned to company i.
#
# A company defaults if:
#
#   X_i <= Phi^{-1}(p)

epsilon = np.random.normal(0, 1, size=(N_sim, m))

default_threshold = norm.ppf(p)

X_sim = (
    rho * F_company
    + np.sqrt(1 - rho ** 2) * epsilon
)

defaults = X_sim <= default_threshold

default_numbers = defaults.sum(axis=1)


# ============================================================
# 10. Plot the simulated distribution of default numbers
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(
    default_numbers,
    bins=np.arange(-0.5, m + 1.5, 1),
    density=True,
    edgecolor="black"
)
plt.xlabel("Number of Defaulted Companies")
plt.ylabel("Estimated Probability")
plt.title("Default Clustering with Two Correlated Regional Factors")
plt.xticks(range(m + 1))
plt.grid(axis="y")
plt.show()


# ============================================================
# 11. Summarize simulation results
# ============================================================

print("\nSummary statistics of simulated default numbers:")
print(pd.Series(default_numbers).describe())

print("\nEstimated probability of each default count:")
print(
    pd.Series(default_numbers)
    .value_counts(normalize=True)
    .sort_index()
)


# ============================================================
# 12. Optional: compare defaults by region
# ============================================================
# This helps students see that defaults can cluster within each region,
# while still being connected across regions through the correlation
# between the two regional factors.

defaults_asia = defaults[:, :n_asia].sum(axis=1)
defaults_me = defaults[:, n_asia:].sum(axis=1)

regional_default_summary = pd.DataFrame({
    "Asia defaults": defaults_asia,
    "Europe/ME defaults": defaults_me,
    "Total defaults": default_numbers
})

print("\nRegional default summary:")
print(regional_default_summary.describe())

print("\nCorrelation between Asia default counts and Europe/ME default counts:")
print(regional_default_summary[["Asia defaults", "Europe/ME defaults"]].corr())


plt.figure(figsize=(7, 5))
plt.scatter(defaults_asia, defaults_me, alpha=0.2)
plt.xlabel("Number of Asia Defaults")
plt.ylabel("Number of Europe/Middle East Defaults")
plt.title("Default Counts by Region")
plt.xticks(range(n_asia + 1))
plt.yticks(range(n_me + 1))
plt.grid(True)
plt.show()


# ============================================================
# 13. Interpretation guide
# ============================================================
# The historical market data are used only to estimate the correlation
# between the two regional market factors.
#
# In the simulation, each regional factor is standard normal:
#
#   F_asia      ~ N(0, 1)
#   F_europe_me ~ N(0, 1)
#
# The two factors are correlated, so bad market conditions in one
# region tend to be associated with bad conditions in the other region.
#
# The first 5 companies are exposed to the Asia factor.
# The last 5 companies are exposed to the Europe/Middle East factor.
#
# Firms in the same region are dependent because they share the same
# regional factor. Firms in different regions are also dependent because
# the two regional factors are correlated.
#
# This creates default clustering across the full portfolio.
# ============================================================