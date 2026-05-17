# ---------------------------------------------------------------
# Script Name: Two Regional Market Factors and Default Clustering
# Author: Shazia Ishaq
# Description: This script uses REAL market data from two regions
#              to estimate how correlated they are, then simulates
#              defaults in a portfolio of 10 companies using a
#              two-factor Gaussian threshold model.
#
# WHAT IS NEW COMPARED TO PORTFOLIO_THRESHOLD.PY?
# ------------------------------------------------
# In Portfolio_Threshold.py we used ONE common factor for all
# companies in the portfolio.
#
# Here we use TWO REGIONAL FACTORS:
#   Factor 1: Emerging Markets Asia (affects 5 Asia companies)
#   Factor 2: Emerging Markets Europe/Middle East (affects 5 ME companies)
#
# The two factors are CORRELATED with each other
# because global markets tend to move together.
# We estimate this correlation from REAL historical data!
#
# This is much more realistic than assuming one global factor.
#
# REAL WORLD CONNECTION:
# At RWE, credit risk analysts use regional factors to model
# how energy companies in different regions might default together
# during global market stress events.
#
# THE MODEL:
# For each company i:
#   X_i = rho * F_region(i) + sqrt(1-rho^2) * epsilon_i
#
# Where:
#   F_region(i) = regional factor for company i (Asia or Europe/ME)
#   epsilon_i   = company-specific shock
#   rho         = sensitivity to regional systematic risk
#   X_i         = creditworthiness (default if X_i <= Phi^-1(p))
#
# ASSUMPTIONS:
#   Total companies:    10
#   Asia companies:     5
#   Europe/ME companies:5
#   Default probability: 5%
#   Systematic sensitivity rho: 0.5
#   Simulations: 10,000
# ---------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pip
from scipy.stats import norm
pip.main(['install', 'xlrd'])

# ---------------------------------------------------------------
# STEP 1: LOAD THE REAL MARKET DATA
# ---------------------------------------------------------------
# We load two Excel files containing monthly equity index data:
# - EM Asia: tracks stock prices of Asian emerging market companies
# - EM Europe/ME: tracks stock prices in Europe and Middle East
#
# These indices represent the economic conditions in each region.
# When index falls -> economy struggling -> more defaults likely!
#
# header=6 means the actual data starts at row 7 of the Excel file
# (rows 1-6 contain metadata/headers that we skip)

# ---------------------------------------------------------------
# STEP 1: LOAD THE REAL MARKET DATA
# ---------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

# Your Excel files are now inside Shazia/data/
data_dir = os.path.join(current_dir, "data")

asia_file = os.path.join(data_dir, "EM Asia monthly.xls")

me_file = os.path.join(
    data_dir,
    "EM Europe and Middle East monthly.xls"
)

# Debug: print paths to confirm correctness
print("Asia file path:", asia_file)
print("Europe/ME file path:", me_file)

# Load Excel files
index_as = pd.read_excel(asia_file, header=6)
index_me = pd.read_excel(me_file, header=6)

print("Asia data loaded successfully!")
print(f"Shape: {index_as.shape}")
print("First few rows:")
print(index_as.head(3))

print("\nEurope/ME data loaded successfully!")
print(f"Shape: {index_me.shape}")
print("First few rows:")
print(index_me.head(3))

# ---------------------------------------------------------------
# STEP 2: CLEAN DATA AND CALCULATE MONTHLY LOG RETURNS
# ---------------------------------------------------------------
# The raw data from Excel is often messy:
# - Numbers stored as text with commas (e.g., "1,234.56")
# - Missing values (NaN) that need to be removed
# - Dates in different formats
#
# We create a function to clean BOTH datasets the same way
# This avoids writing the same code twice (good programming!)
#
# WHY LOG RETURNS?
# Log return = log(Price_t / Price_{t-1})
# = log(Price_t) - log(Price_{t-1})
#
# Log returns are preferred in finance because:
# - They are additive over time (simple returns are not)
# - They are approximately symmetric (better statistical properties)
# - They are used in the GBM formula (connects to Merton model!)

def prepare_price_df(df, price_col):
    # Make a copy so we do not modify the original data
    df = df.copy()

    # Remove rows with any missing values
    df = df.dropna()

    # Remove extra spaces from column names
    # Excel sometimes adds spaces that cause errors
    df.columns = df.columns.str.strip()

    # Convert Date column to proper datetime format
    # format="mixed" handles different date formats automatically
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")

    # Clean the price column:
    # 1. Convert to string first
    # 2. Remove commas (e.g., "1,234" -> "1234")
    # 3. Convert to float number
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Sort by date from oldest to newest
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate monthly log return
    # np.log(df[price_col]) = log of price series
    # .diff() = difference between consecutive values
    # = log(P_t) - log(P_{t-1}) = log(P_t/P_{t-1})
    df["log_return"] = np.log(df[price_col]).diff()

    # Remove first row (no return for first observation)
    return df.dropna(subset=["log_return"])


# Apply cleaning function to both datasets
# index_as.columns[1] gets the name of the second column (price column)
df_index_as = prepare_price_df(index_as, index_as.columns[1])
df_index_me = prepare_price_df(index_me, index_me.columns[1])

print(f"\nAsia data after cleaning: {df_index_as.shape[0]} monthly observations")
print(f"Date range: {df_index_as['Date'].min()} to {df_index_as['Date'].max()}")
print(f"\nEurope/ME data after cleaning: {df_index_me.shape[0]} monthly observations")
print(f"Date range: {df_index_me['Date'].min()} to {df_index_me['Date'].max()}")

# ---------------------------------------------------------------
# STEP 3: MERGE THE TWO RETURN SERIES BY DATE
# ---------------------------------------------------------------
# We need data for the SAME months in both regions
# Some months might be in one dataset but not the other
# pd.merge with how='inner' keeps only COMMON dates
# This ensures we compare apples to apples!

returns = pd.merge(
    df_index_as[["Date", "log_return"]],   # Asia returns
    df_index_me[["Date", "log_return"]],   # Europe/ME returns
    on="Date",                              # match by date
    how="inner",                           # keep only common dates
    suffixes=("_asia", "_me")              # add suffix to distinguish columns
)

returns = returns.dropna().reset_index(drop=True)

print("\nMatched return data (first 5 rows):")
print(returns.head())
print(f"\nTotal matched months: {len(returns)}")

# ---------------------------------------------------------------
# STEP 4: ESTIMATE CORRELATION BETWEEN REGIONS
# ---------------------------------------------------------------
# We calculate the empirical correlation between Asia and Europe/ME
# This measures: when Asia markets fall, do Europe/ME markets also fall?
#
# Correlation ranges from -1 to +1:
# +1 = always move in same direction (perfectly correlated)
#  0 = completely independent (no relationship)
# -1 = always move in opposite directions
#
# This correlation is KEY for our credit risk model!
# Higher correlation = more default clustering = more systemic risk!

corr_matrix = returns[["log_return_asia", "log_return_me"]].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

market_corr = corr_matrix.loc["log_return_asia", "log_return_me"]

print(f"\nEstimated market correlation between")
print(f"EM Asia and EM Europe/Middle East: {market_corr:.4f}")

if market_corr > 0.7:
    print("STRONG positive correlation - regions move very closely together")
elif market_corr > 0.4:
    print("MODERATE positive correlation - some co-movement between regions")
elif market_corr > 0:
    print("WEAK positive correlation - slight tendency to move together")
else:
    print("NEGATIVE correlation - regions tend to move in opposite directions")

# Scatter plot of the two return series
# Shows the relationship between Asia and Europe/ME monthly returns
plt.figure(figsize=(8, 6))
plt.scatter(
    returns["log_return_asia"],
    returns["log_return_me"],
    alpha=0.6,
    color='steelblue',
    edgecolor='black',
    linewidth=0.5
)
plt.xlabel("EM Asia Monthly Log Return", fontsize=11)
plt.ylabel("EM Europe/Middle East Monthly Log Return", fontsize=11)
plt.title(f"Historical Market Returns: Asia vs Europe/ME\n"
          f"Empirical Correlation = {market_corr:.4f}",
          fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# STEP 5 & 6: SET PORTFOLIO ASSUMPTIONS
# ---------------------------------------------------------------
# Now we define our credit portfolio:
# 10 companies total, split equally between Asia and Europe/ME
# Each company has 5% default probability
# Companies are connected to their regional factor with rho=0.5

np.random.seed(42)

N_sim = 10000       # number of simulations (10,000 for accurate estimates)
m = 10              # total companies in portfolio
p = 0.05            # default probability per company (5%)
rho = 0.5           # systematic sensitivity
                    # 50% of risk comes from regional factor
                    # 50% is company-specific

n_asia = 5          # 5 companies in Asia region
n_me = 5            # 5 companies in Europe/Middle East region

# Label each company with its region
company_region = np.array(
    ["Asia"] * n_asia +
    ["Europe_ME"] * n_me
)
# = ["Asia", "Asia", "Asia", "Asia", "Asia",
#    "Europe_ME", "Europe_ME", "Europe_ME", "Europe_ME", "Europe_ME"]

print("\n" + "=" * 55)
print("   CREDIT PORTFOLIO SETUP")
print("=" * 55)
print(f"Total companies:         {m}")
print(f"Asia companies:          {n_asia}")
print(f"Europe/ME companies:     {n_me}")
print(f"Default probability:     {p*100:.0f}%")
print(f"Systematic sensitivity:  {rho}")
print(f"Number of simulations:   {N_sim:,}")
print(f"Empirical correlation:   {market_corr:.4f}")

# ---------------------------------------------------------------
# STEP 7: SIMULATE TWO CORRELATED REGIONAL FACTORS
# ---------------------------------------------------------------
# We simulate two market factors using multivariate normal:
#   F_asia ~ N(0,1)
#   F_europe_me ~ N(0,1)
# But they are CORRELATED with correlation = market_corr
#
# np.random.multivariate_normal generates correlated normals
# We need to specify the covariance matrix:
# [[1, corr],    (variance of Asia=1, covariance=corr)
#  [corr, 1]]   (covariance=corr, variance of ME=1)

market_corr_matrix = np.array([
    [1.0, market_corr],
    [market_corr, 1.0]
])

market_factors = np.random.multivariate_normal(
    mean=[0, 0],               # both factors have mean zero
    cov=market_corr_matrix,    # with real empirical correlation
    size=N_sim                 # generate 10,000 scenarios
)
# Result shape: (10000, 2)
# Column 0: 10,000 Asia factor realizations
# Column 1: 10,000 Europe/ME factor realizations

F_asia_sim = market_factors[:, 0]      # Asia factor for all simulations
F_me_sim = market_factors[:, 1]        # Europe/ME factor for all simulations

print(f"\nSimulated factor correlation check:")
actual_corr = np.corrcoef(F_asia_sim, F_me_sim)[0, 1]
print(f"Target correlation:  {market_corr:.4f}")
print(f"Achieved correlation:{actual_corr:.4f}")
print(f"(Should be very close!)")

# Optional scatter plot of simulated factors
plt.figure(figsize=(7, 5))
plt.scatter(F_asia_sim, F_me_sim, alpha=0.15, color='steelblue')
plt.xlabel("Simulated Asia Market Factor")
plt.ylabel("Simulated Europe/Middle East Market Factor")
plt.title(f"Simulated Correlated Regional Market Factors\n"
          f"Correlation = {actual_corr:.4f}",
          fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# STEP 8: ASSIGN REGIONAL FACTOR TO EACH COMPANY
# ---------------------------------------------------------------
# Companies 1-5 (Asia) are exposed to F_asia
# Companies 6-10 (Europe/ME) are exposed to F_me
#
# np.where works like IF-ELSE:
# IF company_region == "Asia" -> use F_asia_sim
# ELSE -> use F_me_sim
#
# We need to reshape arrays for broadcasting:
# company_region.reshape(1,-1) shape: (1, 10)
# F_asia_sim.reshape(-1,1)     shape: (10000, 1)
# F_me_sim.reshape(-1,1)       shape: (10000, 1)
# Result F_company shape:      (10000, 10)

F_company = np.where(
    company_region.reshape(1, -1) == "Asia",
    F_asia_sim.reshape(-1, 1),
    F_me_sim.reshape(-1, 1)
)
# Now F_company[i, j] = regional factor for simulation i, company j

print(f"\nFactor assignment check:")
print(f"F_company shape: {F_company.shape}")
print(f"First simulation - Asia companies use Asia factor,")
print(f"Europe/ME companies use Europe/ME factor.")

# ---------------------------------------------------------------
# STEP 9: SIMULATE DEFAULTS
# ---------------------------------------------------------------
# For each company i in each simulation:
# X_i = rho * F_region(i) + sqrt(1-rho^2) * epsilon_i
#
# Then company i defaults if X_i <= Phi^(-1)(p)
# norm.ppf(p) = norm.ppf(0.05) = -1.645
# This is the 5th percentile of the standard normal
# Exactly 5% of standard normal values fall below -1.645

epsilon = np.random.normal(0, 1, size=(N_sim, m))
# shape (10000, 10) - one shock per company per simulation

default_threshold = norm.ppf(p)
print(f"\nDefault threshold Phi^(-1)({p}) = {default_threshold:.4f}")
print(f"Company defaults if creditworthiness X falls below {default_threshold:.4f}")

X_sim = (rho * F_company + np.sqrt(1 - rho**2) * epsilon)
# shape (10000, 10)
# Each entry is the creditworthiness of one company in one simulation

defaults = X_sim <= default_threshold
# shape (10000, 10) - True if company defaults, False otherwise

default_numbers = defaults.sum(axis=1)
# shape (10000,) - total defaults per simulation (0 to 10)

# ---------------------------------------------------------------
# STEP 10: PLOT DEFAULT DISTRIBUTION
# ---------------------------------------------------------------
# Histogram shows: how often do 0, 1, 2, ..., 10 companies default?
# The x-axis can only go from 0 to 10 (only 10 companies!)

plt.figure(figsize=(10, 6))
plt.hist(
    default_numbers,
    bins=np.arange(-0.5, m + 1.5, 1),  # bins centered on 0,1,2,...,10
    density=True,
    edgecolor='black',
    color='thistle',
    alpha=0.85
)
plt.xlabel("Number of Defaulted Companies", fontsize=12)
plt.ylabel("Estimated Probability", fontsize=12)
plt.title(f"Default Clustering with Two Correlated Regional Factors\n"
          f"10 companies, PD=5%, rho=0.5, "
          f"Market correlation={market_corr:.4f}",
          fontsize=12, fontweight='bold')
plt.xticks(range(m + 1))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# STEP 11: SUMMARY STATISTICS
# ---------------------------------------------------------------

print("\n" + "=" * 55)
print("   SIMULATION RESULTS SUMMARY")
print("=" * 55)
print(pd.Series(default_numbers).describe().round(4))

print(f"\nExpected number of defaults: {m * p:.1f}")
print(f"Simulated mean defaults:     {default_numbers.mean():.4f}")

print("\nProbability of each default count:")
prob_table = (pd.Series(default_numbers)
              .value_counts(normalize=True)
              .sort_index())
for k, prob in prob_table.items():
    bar = "#" * int(prob * 200)
    print(f"  {int(k):2d} defaults: {prob:.4f} {bar}")

# ---------------------------------------------------------------
# STEP 12: COMPARE DEFAULTS BY REGION
# ---------------------------------------------------------------
# Do Asia and Europe/ME companies default at the same time?
# High correlation between regional defaults = systemic risk!

defaults_asia = defaults[:, :n_asia].sum(axis=1)
# Count defaults among Asia companies only (columns 0-4)

defaults_me = defaults[:, n_asia:].sum(axis=1)
# Count defaults among Europe/ME companies only (columns 5-9)

regional_summary = pd.DataFrame({
    "Asia defaults": defaults_asia,
    "Europe/ME defaults": defaults_me,
    "Total defaults": default_numbers
})

print("\n" + "=" * 55)
print("   REGIONAL DEFAULT ANALYSIS")
print("=" * 55)
print(regional_summary.describe().round(4))

regional_corr = regional_summary[
    ["Asia defaults", "Europe/ME defaults"]
].corr()
print(f"\nCorrelation between Asia and Europe/ME defaults:")
print(regional_corr.round(4))

print(f"\nInterpretation:")
corr_val = regional_corr.iloc[0, 1]
print(f"Default correlation between regions = {corr_val:.4f}")
print(f"Even though companies in different regions use different")
print(f"factors, they still co-default because the two regional")
print(f"factors themselves are correlated (rho={market_corr:.4f}).")
print(f"This shows how global market conditions spread credit risk")
print(f"across different geographic regions simultaneously.")

# Scatter plot of regional defaults
plt.figure(figsize=(7, 5))
plt.scatter(defaults_asia, defaults_me,
            alpha=0.2, color='steelblue')
plt.xlabel("Number of Asia Defaults")
plt.ylabel("Number of Europe/Middle East Defaults")
plt.title(f"Default Counts by Region\n"
          f"Cross-regional default correlation = {corr_val:.4f}",
          fontsize=11)
plt.xticks(range(n_asia + 1))
plt.yticks(range(n_me + 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\n" + "=" * 55)
print("   KEY TAKEAWAYS")
print("=" * 55)
print(f"1. Real market correlation between regions: {market_corr:.4f}")
print(f"2. This correlation was used to simulate correlated factors")
print(f"3. Companies in same region share the same factor")
print(f"4. Cross-regional defaults still occur because factors")
print(f"   are correlated with each other")
print(f"5. This creates systemic risk across the full portfolio")
print(f"6. Expected defaults = {m*p:.1f}, but extreme scenarios")
print(f"   can have {default_numbers.max()} defaults at once!")