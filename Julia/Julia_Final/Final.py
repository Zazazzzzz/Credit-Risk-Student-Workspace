import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import beta
from pathlib import Path
import matplotlib.pyplot as plt

#load data
base_path = Path(__file__).resolve().parent.parent.parent
data_dir = base_path / "Final Assignment" / "Data"

default_history = pd.read_csv(data_dir / "default_history_20y.csv")
entities_info = pd.read_csv(data_dir / "entities_info.csv")
industry_idx_returns = pd.read_csv(data_dir / "industry_index_returns.csv")
portfolio_1 = pd.read_csv(data_dir / "portfolio_1.csv")
portfolio_2 = pd.read_csv(data_dir / "portfolio_2.csv")

#clean default_history
#Standardise entity codes:remove spaces,uppercase
default_history["entity_code"] = default_history["entity_code"].str.strip().str.upper()
#dates
default_history["date"] = pd.to_datetime(default_history["date"], errors="coerce")
#Missing default_event= no default recorded -> fill with 0
default_history["default_event"] = default_history["default_event"].fillna(0).astype(int)
#Standardise source_flag
default_history["source_flag"] = default_history["source_flag"].str.strip().str.lower()
# Remove duplicate (matched by entity and date) rows
default_history = default_history.drop_duplicates(subset=["entity_code", "date"])


#Clean entities
#Standardise entity codes
entities_info["entity_code"] = entities_info["entity_code"].str.strip().str.upper()
#industry names
print("Industries in entities_info (before):", entities_info["industry"].unique())
entities_info["industry"] = entities_info["industry"].str.strip()
entities_info["industry"] = entities_info["industry"].replace({
    "industry a": "Industry A",
    "Ind. A": "Industry A",
    "IndustryA": "Industry A",
    "industry b": "Industry B",
    "Ind. B": "Industry B",
    "IndustryB": "Industry B",
})
print("Industries in entities_info (after):", entities_info["industry"].unique())

#Cut irrelevant_score
entities_info = entities_info.drop(columns=["irrelevant_score"])
#Standardise size_bucket caps: large to Large
entities_info["size_bucket"] = entities_info["size_bucket"].str.strip().str.title()
#fill missing text fields with space
entities_info["analyst_note"] = entities_info["analyst_note"].fillna("").str.strip()
entities_info["region_note"] = entities_info["region_note"].fillna("").str.strip()
#Remove duplicate entity codes, keep first
entities_info = entities_info.drop_duplicates(subset="entity_code")

# clean industry_index_returns
#industry names (same as above)
print("Industries in index_returns (before):", industry_idx_returns["industry"].unique())
industry_idx_returns["industry"] = industry_idx_returns["industry"].str.strip()
industry_idx_returns["industry"] = industry_idx_returns["industry"].replace({
    "industry a": "Industry A",
    "Ind. A": "Industry A",
    "IndustryA": "Industry A",
    "industry b": "Industry B",
    "Ind. B": "Industry B",
    "IndustryB": "Industry B",
})
print("Industries in index_returns (after):", industry_idx_returns["industry"].unique())
#dates
industry_idx_returns["date"] = pd.to_datetime(industry_idx_returns["date"], errors="coerce")
# Drop rows with unparseable dates
industry_idx_returns = industry_idx_returns.dropna(subset=["date"])
# Sort chronologically
industry_idx_returns = industry_idx_returns.sort_values(["industry", "date"]).reset_index(drop=True)
# Remove duplicate (date, industry)
industry_idx_returns = industry_idx_returns.drop_duplicates(subset=["date", "industry"])

#clean portfolios
# Portfolio 1
portfolio_1["entity_code"] = portfolio_1["entity_code"].str.strip().str.upper()
portfolio_1["booking_date"] = pd.to_datetime(portfolio_1["booking_date"], errors="coerce")
portfolio_1["internal_comment"] = portfolio_1["internal_comment"].fillna("").str.strip()
portfolio_1 = portfolio_1.drop_duplicates()

#Portfolio 2
portfolio_2["entity_code"] = portfolio_2["entity_code"].str.strip().str.upper()
#Fix 4 entity codes missing a dash (BSQ4OLC → B-SQ4OLC)
#These are valid entities in entities_info but were recorded without the separator
portfolio_2["entity_code"] = portfolio_2["entity_code"].apply(
    lambda x: x[0] + "-" + x[1:] if len(x) == 7 and "-" not in x else x
)
portfolio_2["booking_date"] = pd.to_datetime(portfolio_2["booking_date"], errors="coerce")
portfolio_2["internal_comment"] = portfolio_2["internal_comment"].fillna("").str.strip()
portfolio_2 = portfolio_2.drop_duplicates()

# MERGE
#build latest default table (one row per entity, most recent record)
latest_default = default_history.sort_values("date", ascending=False)
latest_default = latest_default.drop_duplicates(subset="entity_code", keep="first")
latest_default = latest_default[["entity_code", "date", "default_event", "source_flag"]]
latest_default = latest_default.rename(columns={
    "date": "last_default_date",
    "default_event": "last_default_event",
    "source_flag": "last_default_source"
})

#portfolio 1:merge entity info then default status
portfolio_1 = portfolio_1.merge(entities_info, on="entity_code", how="left")
portfolio_1 = portfolio_1.merge(latest_default, on="entity_code", how="left")
portfolio_1["last_default_event"] = portfolio_1["last_default_event"].fillna(0).astype(int)
portfolio_1 = portfolio_1.sort_values("booking_date").reset_index(drop=True)

#portfolio 2 same
portfolio_2 = portfolio_2.merge(entities_info, on="entity_code", how="left")
portfolio_2 = portfolio_2.merge(latest_default, on="entity_code", how="left")
portfolio_2["last_default_event"] = portfolio_2["last_default_event"].fillna(0).astype(int)
portfolio_2 = portfolio_2.sort_values("booking_date").reset_index(drop=True)

# Check for unmatched entities (will have NaN entity_name after merge)
p1_unmatched = portfolio_1[portfolio_1["entity_name"].isna()]["entity_code"].tolist()
p2_unmatched = portfolio_2[portfolio_2["entity_name"].isna()]["entity_code"].tolist()
if p1_unmatched:
    print("Portfolio 1 - entity codes not found in entities_info:", p1_unmatched)
if p2_unmatched:
    print("Portfolio 2 - entity codes not found in entities_info:", p2_unmatched)


print("Portfolio1, missing EAD:", portfolio_1["EAD"].isna().sum())
print("Portfolio2, missing EAD:", portfolio_2["EAD"].isna().sum())

output_dir = Path(__file__).resolve().parent

# Full merged tables
portfolio_1.to_csv(output_dir / "portfolio_1_final2.csv", index=False)
portfolio_2.to_csv(output_dir / "portfolio_2_final2.csv", index=False)

#drop rows where EAD is missing
portfolio_1_analysis = portfolio_1.dropna(subset=["EAD"])
portfolio_2_analysis = portfolio_2.dropna(subset=["EAD"])

# Portfolio 2 contains 4 entity codes that have
#no matching record in entities_info. Their LGD, PD, industry,... are all missing.
#Since EL = PD x LGD x EAD requires all three inputs, these entities have to be dropped
#in order to correctly proceed with our analysis.
#The full data is preserved in portfolio_2_final2.csv for docu
portfolio_2_analysis = portfolio_2_analysis.dropna(subset=["lgd", "base_pd_hint"])

#Portfolio 1 has 1 entity missing
portfolio_1_analysis = portfolio_1_analysis.dropna(subset=["lgd", "base_pd_hint"])

portfolio_1_analysis.to_csv(output_dir / "portfolio_1_clean2.csv", index=False)
portfolio_2_analysis.to_csv(output_dir / "portfolio_2_clean2.csv", index=False)

print("Port1 rows:", len(portfolio_1_analysis))
print("Port2 rows:", len(portfolio_2_analysis))
print(portfolio_1_analysis.info())
print(portfolio_2_analysis.info())

#Why are there missing rows rows?
#portfolio1
print("Raw rows p1:", len(portfolio_1))
print("After dropna EAD p1:", len(portfolio_1.dropna(subset=["EAD"])))
print("After dropna lgd/pd p1:", len(portfolio_1.dropna(subset=["EAD","lgd","base_pd_hint"])))
#portfolio2
print("Raw rows p2:", len(portfolio_2))
print("After dropna EAD p2:", len(portfolio_2.dropna(subset=["EAD"])))
print("After dropna lgd/pd p2:", len(portfolio_2.dropna(subset=["EAD","lgd","base_pd_hint"])))
#The data cleaning is completed. Port1 has 26 rows (all non-Null), port2 has 26 rows

#Block 1
#portfolio 1 statistical despcription
print(f'Number of obligors in portfolio 1:{len(portfolio_1_analysis)}')
print("Portfolio 1:")
print(portfolio_1_analysis[["base_pd_hint", "lgd", "EAD"]].describe())
print(f'Total Exposure at default: {portfolio_1_analysis["EAD"].sum():,.2f} Dollar')

#portfolio 2 statistical despcription
print(f'Number of obligors in portfolio 2:{len(portfolio_1_analysis)}')
print("Portfolio 2:")
print(portfolio_2_analysis[["base_pd_hint", "lgd", "EAD"]].describe())
print(f'Total Exposure at default: {portfolio_2_analysis["EAD"].sum():,.2f} Dollar')

#Visual descriptive statistic [FIGURE 1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#PD Distribution
ax1.hist(portfolio_1_analysis["base_pd_hint"], bins=10, alpha=0.5, label="Portfolio 1")
ax1.hist(portfolio_2_analysis["base_pd_hint"], bins=10, alpha=0.5, label="Portfolio 2")
ax1.set_xlabel("PD")
ax1.set_ylabel("Count")
ax1.set_title("PD Distribution")
ax1.legend()

#EAD Distribution
ax2.hist(portfolio_1_analysis["EAD"], bins=10, alpha=0.5, label="Portfolio 1")
ax2.hist(portfolio_2_analysis["EAD"], bins=10, alpha=0.5, label="Portfolio 2")
ax2.set_xlabel("EAD")
ax2.set_ylabel("Count")
ax2.set_title("EAD Distribution")
ax2.legend()

plt.tight_layout()
plt.show()

# Exposure concentration (FIGURE 2)
portfolio_1_analysis["EAD_pct"] = portfolio_1_analysis["EAD"] / portfolio_1_analysis["EAD"].sum() * 100
portfolio_2_analysis["EAD_pct"] = portfolio_2_analysis["EAD"] / portfolio_2_analysis["EAD"].sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Portfolio 1
axes[0].bar(portfolio_1_analysis["entity_code"], portfolio_1_analysis["EAD_pct"])
axes[0].set_title("Exposure Concentration Portfolio 1")
axes[0].set_xlabel("Entity")
axes[0].set_ylabel("% of Total EAD")
axes[0].tick_params(axis="x", rotation=90) #To turn the labels vertical

# Portfolio 2
axes[1].bar(portfolio_2_analysis["entity_code"], portfolio_2_analysis["EAD_pct"])
axes[1].set_title("Exposure Concentration Portfolio 2")
axes[1].set_xlabel("Entity")
axes[1].set_ylabel("% of Total EAD")
axes[1].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# Regional composition #not in report
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

portfolio_1_analysis["region"].value_counts().plot(kind="bar", ax=axes[0])
axes[0].set_title("Regional Composition Portf 1")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

portfolio_2_analysis["region"].value_counts().plot(kind="bar", ax=axes[1])
axes[1].set_title("Regional Composition Portf 2")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

#BLOCK II: Gaussian One-Factor-Model
np.random.seed(42)

#To increase efficiency, the columns are transormed into arrays with the help of the value command
pd_1=portfolio_1_analysis["base_pd_hint"].values
lgd_1=portfolio_1_analysis["lgd"].values
ead_1=portfolio_1_analysis["EAD"].values
pd_2=portfolio_2_analysis["base_pd_hint"].values
lgd_2=portfolio_2_analysis["lgd"].values
ead_2=portfolio_2_analysis["EAD"].values


n_simulations=100000
rho=0.2  #correlation of Basel II

#Def. thresholds
threshold_1=norm.ppf(pd_1)
threshold_2=norm.ppf(pd_2)

#Store portfolio losses
losses_1=np.zeros(n_simulations)
losses_2=np.zeros(n_simulations)

for i in range(n_simulations):
    #systematic factor (Stand. normal!)
    F=np.random.normal(0, 1)

    #idiosyncratic factors (one/obligor and also stand. normal)
    eps_1=np.random.normal(0, 1, len(pd_1))
    eps_2=np.random.normal(0, 1, len(pd_2))

    #asset returns, we use the Formula
    A_1=np.sqrt(rho)* F+ np.sqrt(1-rho) *eps_1
    A_2=np.sqrt(rho)*F+np.sqrt(1-rho)*eps_2

    #Default if return is below threshold
    defaults_1=A_1<threshold_1
    defaults_2=A_2<threshold_2

    #Portfolio loss=sum of EAD*LGD for defaulted obligors
    losses_1[i]=np.sum(ead_1[defaults_1]*lgd_1[defaults_1])
    losses_2[i]=np.sum(ead_2[defaults_2]*lgd_2[defaults_2])


#RISK MEASURES
EL_1 = np.mean(losses_1)
VaR99_1=np.percentile(losses_1, 99)
ES99_1=np.mean(losses_1[losses_1 > VaR99_1])

EL_2=np.mean(losses_2)
VaR99_2=np.percentile(losses_2, 99)
ES99_2=np.mean(losses_2[losses_2 > VaR99_2])

print("Portfolio 1:")
print(f"EL: {EL_1:,.2f} VaR99: {VaR99_1:,.2f} ES99: {ES99_1:,.2f}")
print(f"Portfolio 2:")
print(f"EL: {EL_2:,.2f} VaR99: {VaR99_2:,.2f} ES99: {ES99_2:,.2f}")


# PLOT LOSS DISTRIBUTIONS [FIGURE 3]
plt.figure(figsize=(10, 5))
plt.hist(losses_1, bins=100, alpha=0.5, label="Portfolio 1")
plt.hist(losses_2, bins=100, alpha=0.5, label="Portfolio 2")
plt.axvline(VaR99_1,linewidth=2, color="blue",   linestyle="--", label="VaR99 P1")
plt.axvline(VaR99_2, color="orange", linestyle="--", label="VaR99 P2",linewidth=2)
plt.xlabel("Portfolio Loss")
plt.ylabel("Frequency")
plt.title("Loss Distribution in Gaussian One-Factor Model (rho=0.2)")
plt.legend()
plt.tight_layout()
plt.show()

#Check my result thanks to the analytical property: EL=PD*LGD*EAD)
EL_an_1=np.sum(pd_1*lgd_1*ead_1)
EL_an_2=np.sum(pd_2*lgd_2*ead_2)
print(f"Analytical EL1:{EL_an_1:,.2f}")
print(f"Analytical EL2:{EL_an_2:,.2f}") #Very close, good/valid simulation!

#Sensitivity analysis to different rhos:
#as ρ increases, VaR and ES should increase significantly while EL stays roughly the same
rho_values=[0.1, 0.3]
sensitivity_results=[]

for rho in rho_values:
    losses_1_sens= np.zeros(n_simulations)
    losses_2_sens =np.zeros(n_simulations)

    for i in range(n_simulations):
        F=np.random.normal(0, 1)
        eps_1=np.random.normal(0, 1, len(pd_1))
        eps_2=np.random.normal(0, 1, len(pd_2))

        A_1= np.sqrt(rho)*F+np.sqrt(1-rho)*eps_1
        A_2=np.sqrt(rho)*F+ np.sqrt(1-rho)*eps_2

        defaults_1=A_1<threshold_1
        defaults_2=A_2 < threshold_2

        losses_1_sens[i]=np.sum(ead_1[defaults_1]*lgd_1[defaults_1])
        losses_2_sens[i]=np.sum(ead_2[defaults_2]*lgd_2[defaults_2])

    #for only this rho
    sensitivity_results.append({
        "rho":      rho,
        "EL_1":     np.mean(losses_1_sens),
        "VaR99_1":  np.percentile(losses_1_sens, 99),
        "ES99_1":   np.mean(losses_1_sens[losses_1_sens > np.percentile(losses_1_sens, 99)]),
        "EL_2":     np.mean(losses_2_sens),
        "VaR99_2":  np.percentile(losses_2_sens, 99),
        "ES99_2":   np.mean(losses_2_sens[losses_2_sens > np.percentile(losses_2_sens, 99)]),
    })

sensitivity_df = pd.DataFrame(sensitivity_results) #nicer display of results. Confirms intuition
print(sensitivity_df.to_string(index=False))       #see report for detailled explanation

# BLOCK III BETA-BINOMIAL-MODEL
#We need to use the default history to fit the beta distribution of rho! Table:
# Count total observed and total defaulted by summing them per year
yearly =default_history.groupby(default_history["date"].dt.year).agg(total=("default_event", "count"), defaults=("default_event", "sum")
)
yearly["default_rate"]= yearly["defaults"] / yearly["total"]
print(yearly)
print(f"Over {len(yearly)} years")

#Fit the beta distribution to those default dates
default_rates = yearly["default_rate"].replace(0, 0.0001) #As in lecture: replace 0s with small value so Beta dist. can handle them

alpha, beta_param, loc, scale = beta.fit(default_rates, floc=0, fscale=1) #fixes the distribution between 0 and 1 because beta distribution property

print(f"Alpha: {alpha:.4f}")
print(f"Beta:  {beta_param:.4f}")

#Again use actual (historic) vs fitted data to check:
print(f"Mean of fitted Beta: {alpha / (alpha + beta_param):.4f}")
print(f"Mean of historical default rates: {yearly['default_rate'].mean():.4f}") #very good

#Simulation
losses_bb_1=np.zeros(n_simulations)
losses_bb_2=np.zeros(n_simulations)

for i in range(n_simulations):
    p =beta.rvs(alpha, beta_param) #draws one rho for each simulation from our beta distribution (random variate sample)

    # Each obligor defaults with probability p (iid)
    defaults_1=np.random.binomial(1, p, len(pd_1))
    defaults_2=np.random.binomial(1, p, len(pd_2))

    #total portfolio loss
    losses_bb_1[i]=np.sum(ead_1[defaults_1==1]*lgd_1[defaults_1==1])
    losses_bb_2[i]=np.sum(ead_2[defaults_2==1]*lgd_2[defaults_2==1])

#Risk-measures
EL_bb_1   =np.mean(losses_bb_1)
VaR99_bb_1=np.percentile(losses_bb_1, 99)
ES99_bb_1=np.mean(losses_bb_1[losses_bb_1>VaR99_bb_1])

EL_bb_2 =np.mean(losses_bb_2)
VaR99_bb_2=np.percentile(losses_bb_2, 99)
ES99_bb_2= np.mean(losses_bb_2[losses_bb_2>VaR99_bb_2])

print("Portfolio 1:")
print(f"EL: {EL_bb_1:,.2f}  VaR99: {VaR99_bb_1:,.2f}  ES99: {ES99_bb_1:,.2f}")
print("Portfolio 2:")
print(f"EL: {EL_bb_2:,.2f}  VaR99: {VaR99_bb_2:,.2f}  ES99: {ES99_bb_2:,.2f}")

#Final comparison One-Factor and beta binomial models
comparison = pd.DataFrame({
    "Metric": ["EL", "VaR 99%", "ES 99%"],
    "P1 Gaussian": [EL_1, VaR99_1, ES99_1],
    "P2 Gaussian": [EL_2, VaR99_2, ES99_2],
    "P1 Beta-Binomial": [EL_bb_1, VaR99_bb_1, ES99_bb_1],
    "P2 Beta-Binomial": [EL_bb_2, VaR99_bb_2, ES99_bb_2],
})
print(comparison.to_string(index=False))

#plot FIGURE 4
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(losses_1, bins=100, alpha=0.5, label="One-Factor")
axes[0].hist(losses_bb_1, bins=100, alpha=0.5, label="Beta-Binomial")
axes[0].axvline(VaR99_1, color="blue", linestyle="--", label="VaR99 One-Factor",linewidth=2)
axes[0].axvline(VaR99_bb_1, linewidth=2, color="pink", linestyle="--", label="VaR99 BB")
axes[0].set_title("Port.1 Model Comparison")
axes[0].set_xlabel("Loss")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(losses_2, bins=100, alpha=0.5, label="One Factor")
axes[1].hist(losses_bb_2, bins=100, alpha=0.5, label="Beta-Binomial")
axes[1].axvline(VaR99_2, color="blue",linewidth=2, linestyle="--", label="VaR99 One-Factor")
axes[1].axvline(VaR99_bb_2, linewidth=2, color="pink", linestyle="--", label="VaR99 BB")
axes[1].set_title("Portf. 2 Model Comparison")
axes[1].set_xlabel("Loss")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.show()


# Split default history by industry
# First merge default_history with entities_info to get industry per entity
default_history_with_industry = default_history.merge(
    entities_info[["entity_code", "industry"]],on="entity_code",how="left"
)

# Separate by industry
default_history_A = default_history_with_industry[default_history_with_industry["industry"] == "Industry A"]
default_history_B = default_history_with_industry[default_history_with_industry["industry"] == "Industry B"]

# Compute yearly default rates per industry
yearly_A = default_history_A.groupby(default_history_A["date"].dt.year).agg(total=("default_event", "count"),
                                defaults=("default_event", "sum")
)
yearly_A["default_rate"] = yearly_A["defaults"] / yearly_A["total"]

yearly_B = default_history_B.groupby(default_history_B["date"].dt.year).agg(
    total=("default_event", "count"),
    defaults=("default_event", "sum")
)
yearly_B["default_rate"] = yearly_B["defaults"] / yearly_B["total"]

print("Industry A yearly defaults:")
print(yearly_A)
print("\nIndustry B yearly defaults:")
print(yearly_B)

#Fit separate Beta distributions per portfolio
rates_1=yearly_A["default_rate"].replace(0, 0.0001)
rates_2=yearly_B["default_rate"].replace(0, 0.0001)

alpha_1, beta_1, _, _=beta.fit(rates_1, floc=0, fscale=1)
alpha_2, beta_2, _, _=beta.fit(rates_2, floc=0, fscale=1)

print(f"Portfolio 1:")
print(f"Alpha: {alpha_1:.4f}  Beta: {beta_1:.4f}  Mean: {alpha_1/(alpha_1+beta_1):.4f}")
print(f"Portfolio 2:")
print(f"Alpha: {alpha_2:.4f}  Beta: {beta_2:.4f}  Mean: {alpha_2/(alpha_2+beta_2):.4f}")

# Beta-Binomial simulation with portfolio-specific parameters
losses_bb_1_new =np.zeros(n_simulations)
losses_bb_2_new = np.zeros(n_simulations)

for i in range(n_simulations):
    p_1_new = beta.rvs(alpha_1, beta_1)
    p_2_new= beta.rvs(alpha_2, beta_2)

    defaults_1_new = np.random.binomial(1, p_1_new, len(pd_1))
    defaults_2_new = np.random.binomial(1, p_2_new, len(pd_2))

    losses_bb_1_new[i]= np.sum(ead_1[defaults_1_new == 1]*lgd_1[defaults_1_new== 1])
    losses_bb_2_new[i] = np.sum(ead_2[defaults_2_new == 1]* lgd_2[defaults_2_new== 1])

# Risk measures
EL_bb_1_new =np.mean(losses_bb_1_new)
VaR99_bb_1_new=np.percentile(losses_bb_1_new, 99)
ES99_bb_1_new=np.mean(losses_bb_1_new[losses_bb_1_new>VaR99_bb_1_new])

EL_bb_2_new =np.mean(losses_bb_2_new)
VaR99_bb_2_new=np.percentile(losses_bb_2_new, 99)
ES99_bb_2_new=np.mean(losses_bb_2_new[losses_bb_2_new>VaR99_bb_2_new])

print("Portfolio 1 with correction:")
print(f"EL: {EL_bb_1_new:,.2f}  VaR99: {VaR99_bb_1_new:,.2f}  ES99: {ES99_bb_1_new:,.2f}")
print("Portfolio 2 with correction:")
print(f"EL: {EL_bb_2_new:,.2f}  VaR99: {VaR99_bb_2_new:,.2f}  ES99: {ES99_bb_2_new:,.2f}")

#Final comparison table
comparison = pd.DataFrame({
    "Metric": ["EL", "VaR 99%", "ES 99%"],
    "P1 Gaussian": [EL_1, VaR99_1, ES99_1],
    "P2 Gaussian": [EL_2, VaR99_2, ES99_2],
    "P1 BB (old)": [EL_bb_1, VaR99_bb_1, ES99_bb_1],
    "P2 BB (old)": [EL_bb_2, VaR99_bb_2, ES99_bb_2],
    "P1 BB (corrected)": [EL_bb_1_new, VaR99_bb_1_new, ES99_bb_1_new],
    "P2 BB (corrected)": [EL_bb_2_new, VaR99_bb_2_new, ES99_bb_2_new],
})
print(comparison.to_string(index=False))


#plot by industries [FIGURE 5]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(losses_1, bins=100, alpha=0.5, label="One-Factor")
axes[0].hist(losses_bb_1_new, bins=100, alpha=0.5, label="Beta-Binomial")
axes[0].axvline(VaR99_1, color="blue",linewidth=2, linestyle="--", label="VaR99 One-Factor")
axes[0].axvline(VaR99_bb_1_new, linewidth=2,color="red", linestyle="--", label="VaR99 BB")
axes[0].set_title("Portf.1 Comparison w/ individual p")
axes[0].set_xlabel("Loss")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(losses_2, bins=100, alpha=0.5, label="One-Factor")
axes[1].hist(losses_bb_2_new, bins=100, alpha=0.5, label="Beta-Binomial")
axes[1].axvline(VaR99_2, color="blue", linestyle="--", label="VaR99 One-Factor",linewidth=2)
axes[1].axvline(VaR99_bb_2_new, color="red",linewidth=2, linestyle="--", label="VaR99 BB")
axes[1].set_title("Portf.2 Comparison w/ individual p")
axes[1].set_xlabel("Loss")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.show()