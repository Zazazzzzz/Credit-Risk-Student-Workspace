import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# step 1

asia = pd.read_excel("EM Asia monthly.xls",header=6)
europe = pd.read_excel("EM Europe and Middle East monthly.xls",header=6)
print(asia.head())
print(europe.head())


# step 2

asia.columns = ["Date","Asia_Price"]
europe.columns = ["Date","Europe_Price"]

asia["Date"] = pd.to_datetime(asia["Date"] , errors="coerce")
europe["Date"] = pd.to_datetime(europe["Date"] , errors="coerce")
asia["Asia_Price"]=pd.to_numeric(asia["Asia_Price"], errors="coerce")
europe["Europe_Price"]=pd.to_numeric(europe["Europe_Price"], errors="coerce")
asia=asia.dropna ()
europe=europe.dropna ()

asia["Asia_Return"]=np.log(asia["Asia_Price"]/asia["Asia_Price"].shift(1))
europe["Europe_Return"]=np.log(europe["Europe_Price"]/europe["Europe_Price"].shift(1))
asia=asia.dropna()
europe=europe.dropna()

print(asia.head())
print(europe.head())


# step 3

Returns = pd.merge(asia[["Date","Asia_Return"]], europe[["Date","Europe_Return"]], on=["Date"], how="inner")
print(Returns.head())


# step 4

corr_matrix = Returns[["Asia_Return","Europe_Return"]].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

market_corr = corr_matrix.loc["Asia_Return","Europe_Return"]
print("Empirical correlation",market_corr)

# step 5

market_corr_matrix = np.array([[1.0, market_corr], [market_corr, 1.0]])
market_factors = np.random.multivariate_normal(mean=[0,0], cov=market_corr_matrix, size=10000)

F_asia= market_factors[:,0]
F_europe= market_factors[:,1]

print("\nCorrelation of simulated market factors:")
print(np.corrcoef(F_asia, F_europe)[0, 1])


# step 6

F_region_assignment = {
    "companies_1_to_5": F_asia,
    "companies_6_to_10": F_europe
}


# step 7

rho = 0.5
p_default = 0.05
N = 10000
n_companies = 10

epsilon = np.random.normal(0, 1, size=(N, n_companies))

X_asia = ( rho * F_region_assignment["companies_1_to_5"][:, np.newaxis] + np.sqrt(1 - rho ** 2) * epsilon[:, 0:5])
X_europe = (rho * F_region_assignment["companies_6_to_10"][:, np.newaxis]+ np.sqrt(1 - rho ** 2) * epsilon[:, 5:10])
X = np.column_stack([X_asia, X_europe])

print("\nShape of X:")
print(X.shape)


# step 8

default_threshold = norm.ppf(p_default)

print("\nDefault threshold:")
print(default_threshold)


# step 9

defaults = X <= default_threshold
default_numbers = defaults.sum(axis=1)

print("\nNumber of defaulted companies in each simulation:")
print(default_numbers)
print(default_numbers.shape)


# step 10

plt.hist(
    default_numbers,
    bins=range(0, 12),
    density=True,
    label="Two regional factors"
)

plt.title("Two Regional Market Factors")
plt.xlabel("Number of Defaults")
plt.ylabel("Density")
plt.legend()
plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()







