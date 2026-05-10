#Task 1 Two regional markets factors #gausian threshold model
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

base_path = Path(__file__).resolve().parent.parent

asian_file = base_path / "Credit Risk" / "EM Asia monthly.xls"
eu_file = base_path / "Credit Risk" / "EM Europe and Middle East monthly.xls"

Asian_data = pd.read_excel(asian_file, engine="xlrd")
EU_ME_data = pd.read_excel(eu_file, engine="xlrd")

print(Asian_data.head(15))
print(EU_ME_data.head(15))

Asian_data= Asian_data.iloc[7:,]
EU_ME_data= EU_ME_data.iloc[7:,]

Asian_data.columns=["Date","Price"]
EU_ME_data.columns=["Date","Price"]

Asian_data["Price"] = pd.to_numeric(Asian_data["Price"], errors="coerce")
EU_ME_data["Price"] = pd.to_numeric(EU_ME_data["Price"], errors="coerce")

Asian_data=Asian_data.dropna()
EU_ME_data=EU_ME_data.dropna()

Asian_data["Date"]=pd.to_datetime(Asian_data["Date"])
EU_ME_data["Date"]=pd.to_datetime(EU_ME_data["Date"])

# I forgot about index setting
Asian_data.set_index("Date",inplace=True)
EU_ME_data.set_index("Date",inplace=True)

print(Asian_data.head(15))
print(EU_ME_data.head(15))

#computing the log returns
Asian_data["log_return"]=np.log(1+Asian_data["Price"].pct_change(1))
EU_ME_data["log_return"]=np.log(1+EU_ME_data["Price"].pct_change(1))

Asian_data=Asian_data.dropna()
EU_ME_data=EU_ME_data.dropna()

print(Asian_data.head(10))
print(EU_ME_data.head(10))

returns=pd.concat([Asian_data["log_return"],EU_ME_data["log_return"]],axis=1,join="inner") #inner only keeps common dates

returns.columns=["Asia_Return","EU_ME_Return"]
print(returns.head(10))

corr_Asia_EU_ME= returns.corr()
print(corr_Asia_EU_ME)

#carefull check if empty rows!!
print(returns.isna().sum()) #all good!

np.random.seed(42)

rho_emp = returns["Asia_Return"].corr(returns["EU_ME_Return"])

n_sims = 10000

z1 = np.random.normal(size=n_sims)
z2 = np.random.normal(size=n_sims) #until here it s pure randomness but
#we want to add the common rho


F_asia = z1    #the correlated factors
F_eu = rho_emp * z1 + np.sqrt(1 - rho_emp**2) * z2 #first part is the common shock
#last ensures randomness with var =1

n_companies = 10
region = np.array(["Asia"] * 5 + ["EU"] * 5)

rho = 0.5
p_default = 0.05
threshold = norm.ppf(p_default)

X = np.zeros((n_sims, n_companies))

for i in range(n_companies):
    if i < 5:
        F = F_asia
    else:
        F = F_eu
    epsilon = np.random.normal(size=n_sims)

    X[:, i] = rho * F + np.sqrt(1 - rho ** 2) * epsilon

print("Default threshold:", threshold)
defaults = (X <= threshold).sum(axis=1)


plt.figure()
plt.hist(defaults, bins=range(0, 12), edgecolor="black")
plt.title("Simulated Number of Defaults")
plt.xlabel("Number of Defaults")
plt.ylabel("Frequency")
plt.show()


