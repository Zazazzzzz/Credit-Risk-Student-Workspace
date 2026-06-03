import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt

# Helper function for entity code

def fix_entity_code(code):
    code = str(code).strip().upper()
    if len(code) > 1 and code[0] in ["A", "B"] and code[1] != "-":
        code = code[0] + "-" + code[1:]
    return code

# ============================================================
# Part 1 - Data Cleaning
# ============================================================

# Change only these paths when running the project on another computer
DIR_BASE = Path(__file__).resolve().parent
DIR_DATA = DIR_BASE / "data"
DIR_RESULTS = DIR_BASE / "results"
DIR_RESULTS.mkdir(parents=True, exist_ok=True)

# 1. Clean entities_info.csv
entities = pd.read_csv(DIR_DATA / "entities_info.csv")

# Standardize entity codes
entities["entity_code"] = entities["entity_code"].apply(fix_entity_code)

# Standardize industry labels
industry_map = {
    "IND. A": "Industry A",
    "INDUSTRYA": "Industry A",
    "INDUSTRY A": "Industry A",
    "IND. B": "Industry B",
    "INDUSTRYB": "Industry B",
    "INDUSTRY B": "Industry B",
}

entities["industry"] = entities["industry"].astype(str).str.strip().str.upper()
entities["industry"] = entities["industry"].replace(industry_map)

# Standardize region
entities["region"] = entities["region"].astype(str).str.strip().str.title()
entities["region"] = entities["region"].replace({"Nan": np.nan}).fillna("Unknown")

# Remove duplicate firm rows
entities = entities.drop_duplicates(subset=["entity_code"])

# Convert LGD and base PD hint to numeric
entities["lgd"] = pd.to_numeric(entities["lgd"], errors="coerce")
entities["base_pd_hint"] = pd.to_numeric(entities["base_pd_hint"], errors="coerce")

# Fill missing LGD by industry median, then overall median
entities["lgd"] = entities.groupby("industry")["lgd"].transform(lambda x: x.fillna(x.median()))
entities["lgd"] = entities["lgd"].fillna(entities["lgd"].median())

# Fill missing base PD hint by industry median, then overall median
entities["base_pd_hint"] = entities.groupby("industry")["base_pd_hint"].transform(lambda x: x.fillna(x.median()))
entities["base_pd_hint"] = entities["base_pd_hint"].fillna(entities["base_pd_hint"].median())

# Drop columns that are not needed for modeling
entities_clean = entities.drop(columns=["analyst_note", "region_note", "irrelevant_score", "size_bucket"], errors="ignore")

# Save cleaned entity information
entities_clean.to_csv(DIR_RESULTS / "entities_info_clean.csv", index=False)


# 2. Clean default_history_20y.csv
default_history = pd.read_csv(DIR_DATA / "default_history_20y.csv")

# Standardize entity codes
default_history["entity_code"] = default_history["entity_code"].apply(fix_entity_code)

# Convert date column to datetime format
# default_history dates are in formats such as 12/31/2006, so dayfirst=False
default_history["date"] = pd.to_datetime(default_history["date"], errors="coerce", dayfirst=False)

# Convert default_event to numeric
default_history["default_event"] = pd.to_numeric(default_history["default_event"], errors="coerce")

# Fill missing default_event with 0
# Missing default records are treated as no observed default
default_history["default_event"] = default_history["default_event"].fillna(0)

# Remove rows with invalid dates or missing entity codes
default_history = default_history.dropna(subset=["date", "entity_code"])

# Remove duplicate observations for the same firm and date
default_history = default_history.drop_duplicates(subset=["entity_code", "date"])

# Save cleaned default history
default_history.to_csv(DIR_RESULTS / "default_history_20y_clean.csv", index=False)


# 3. Clean industry_index_returns.csv
industry_returns = pd.read_csv(DIR_DATA / "industry_index_returns.csv")

# Standardize industry labels
industry_returns["industry"] = industry_returns["industry"].astype(str).str.strip().str.upper()
industry_returns["industry"] = industry_returns["industry"].replace(industry_map)

# Convert date column
industry_returns["date"] = pd.to_datetime(industry_returns["date"], errors="coerce", dayfirst=True)

# Convert index_return to numeric
industry_returns["index_return"] = pd.to_numeric(industry_returns["index_return"], errors="coerce")

# Remove invalid rows
industry_returns = industry_returns.dropna(subset=["date", "industry"])

# Sort before filling missing returns
industry_returns = industry_returns.sort_values(["industry", "date"])

# Remove duplicate observations for the same industry and date
industry_returns = industry_returns.drop_duplicates(subset=["industry", "date"])

# Fill missing index returns within each industry
industry_returns["index_return"] = industry_returns.groupby("industry")["index_return"].transform(lambda x: x.interpolate().ffill().bfill())

# Save cleaned industry returns
industry_returns.to_csv(DIR_RESULTS / "industry_index_returns_clean.csv", index=False)


# 4. Clean stock_returns.csv
stock_returns = pd.read_csv(DIR_DATA / "stock_returns.csv")

# Standardize entity codes
stock_returns["entity_code"] = stock_returns["entity_code"].apply(fix_entity_code)

# Convert date column
stock_returns["date"] = pd.to_datetime(stock_returns["date"], errors="coerce", dayfirst=True)

# Convert stock_return to numeric
stock_returns["stock_return"] = pd.to_numeric(stock_returns["stock_return"], errors="coerce")

# Remove invalid rows
stock_returns = stock_returns.dropna(subset=["date", "entity_code"])

# Sort before filling missing returns
stock_returns = stock_returns.sort_values(["entity_code", "date"])

# Remove duplicate observations for the same firm and date
stock_returns = stock_returns.drop_duplicates(subset=["entity_code", "date"])

# Fill missing stock returns within each firm
stock_returns["stock_return"] = stock_returns.groupby("entity_code")["stock_return"].transform(lambda x: x.interpolate().ffill().bfill())

# Save cleaned stock returns
stock_returns.to_csv(DIR_RESULTS / "stock_returns_clean.csv", index=False)


# 5. Clean portfolio_1.csv and portfolio_2.csv
def clean_portfolio(file_name, output_name):
    # Read portfolio file
    portfolio = pd.read_csv(DIR_DATA / file_name)

    # Standardize entity codes
    portfolio["entity_code"] = portfolio["entity_code"].apply(fix_entity_code)

    # Convert EAD and booking_date
    portfolio["EAD"] = pd.to_numeric(portfolio["EAD"], errors="coerce")
    portfolio["booking_date"] = pd.to_datetime(portfolio["booking_date"], errors="coerce")

    # Remove duplicates and fill missing EAD
    portfolio = portfolio.drop_duplicates()
    portfolio["EAD"] = portfolio["EAD"].fillna(portfolio["EAD"].median())

    # Drop non-modeling column and save cleaned file
    portfolio = portfolio.drop(columns=["internal_comment"], errors="ignore")
    portfolio.to_csv(DIR_RESULTS / output_name, index=False)

    return portfolio

portfolio_1 = clean_portfolio("portfolio_1.csv", "portfolio_1_clean.csv")
portfolio_2 = clean_portfolio("portfolio_2.csv", "portfolio_2_clean.csv")


# 6. Merge data into modeling tables

# Merge Portfolio 1 with firm characteristics
portfolio_1_modeling_table = portfolio_1.merge(entities_clean, on="entity_code", how="left")

# Merge Portfolio 2 with firm characteristics
portfolio_2_modeling_table = portfolio_2.merge(entities_clean, on="entity_code", how="left")

# Add portfolio identifier
portfolio_1_modeling_table["portfolio"] = "Portfolio 1"
portfolio_2_modeling_table["portfolio"] = "Portfolio 2"

# Combine Portfolio 1 and Portfolio 2 into one modeling table
all_portfolio_modeling_table = pd.concat([portfolio_1_modeling_table, portfolio_2_modeling_table], ignore_index=True)

# Save Portfolio 1 modeling table
portfolio_1_modeling_table.to_csv(DIR_RESULTS / "portfolio_1_modeling_table.csv", index=False)

# Save Portfolio 2 modeling table
portfolio_2_modeling_table.to_csv(DIR_RESULTS / "portfolio_2_modeling_table.csv", index=False)

# Save combined modeling table
all_portfolio_modeling_table.to_csv(DIR_RESULTS / "all_portfolio_modeling_table.csv", index=False)


# ============================================================
# Part 2: Model Setup and Parameter Estimation
# ============================================================

all_portfolios = all_portfolio_modeling_table.copy()

# Use base_pd_hint as one-year PD
all_portfolios["pd_1y"] = all_portfolios["base_pd_hint"]

# Basel II corporate asset correlation formula: rho = 0.24 - 0.12 * ((1 - exp(-50 * PD)) / (1 - exp(-50)))
all_portfolios["rho"] = 0.24 - 0.12 * ((1 - np.exp(-50 * all_portfolios["pd_1y"])) / (1 - np.exp(-50)))

# Individual expected loss: EL_i = EAD_i * LGD_i * PD_i
all_portfolios["expected_loss_i"] = all_portfolios["EAD"] * all_portfolios["lgd"] * all_portfolios["pd_1y"]

# Create parameter summary by portfolio
parameter_summary = all_portfolios.groupby("portfolio").size().reset_index(name="number_of_obligors")
parameter_summary["total_EAD"] = all_portfolios.groupby("portfolio")["EAD"].sum().values
parameter_summary["average_PD"] = all_portfolios.groupby("portfolio")["pd_1y"].mean().values
parameter_summary["average_LGD"] = all_portfolios.groupby("portfolio")["lgd"].mean().values
parameter_summary["average_rho"] = all_portfolios.groupby("portfolio")["rho"].mean().values
parameter_summary["expected_loss"] = all_portfolios.groupby("portfolio")["expected_loss_i"].sum().values
parameter_summary["expected_loss_ratio"] = parameter_summary["expected_loss"] / parameter_summary["total_EAD"]

# Save full parameter table
all_portfolios.to_csv(DIR_RESULTS / "all_portfolio_parameters.csv", index=False)

# Save parameter summary
parameter_summary.to_csv(DIR_RESULTS / "part2_parameter_summary.csv", index=False)

print("Part 2 parameter summary:")
print(parameter_summary.to_string(index=False))


# ============================================================
# Part 3: Simulation and Risk Measurement
# ============================================================

# 1. Simulation settings
N_SIM = 10000
np.random.seed(42)

def simulate_portfolio_loss(portfolio_data, portfolio_name):
    # Get model inputs
    EAD = portfolio_data["EAD"].values
    LGD = portfolio_data["lgd"].values
    PD = portfolio_data["pd_1y"].values
    rho = portfolio_data["rho"].values

    # Basel II rho is asset correlation, so factor loading is sqrt(rho)
    factor_loading = np.sqrt(rho)

    # Default threshold
    default_threshold = norm.ppf(PD)

    # Simulate systematic factor and idiosyncratic shocks
    F = np.random.normal(size=(N_SIM, 1))
    epsilon = np.random.normal(size=(N_SIM, len(portfolio_data)))

    # Simulate latent credit quality
    X = factor_loading * F + np.sqrt(1 - rho) * epsilon

    # Default occurs when X is below the threshold
    default_indicator = X <= default_threshold

    # Calculate portfolio losses
    loss_given_default = EAD * LGD
    losses = (default_indicator * loss_given_default).sum(axis=1)

    # Calculate risk measures
    expected_loss = losses.mean()
    var_95 = np.quantile(losses, 0.95)
    var_99 = np.quantile(losses, 0.99)
    es_95 = losses[losses >= var_95].mean()
    es_99 = losses[losses >= var_99].mean()

    # Save simulated losses
    simulated_losses = pd.DataFrame({
        "portfolio": portfolio_name,
        "simulation_id": range(1, N_SIM + 1),
        "loss": losses
    })

    # Save risk measures
    risk_measures = {
        "portfolio": portfolio_name,
        "simulated_expected_loss": expected_loss,
        "VaR_95": var_95,
        "VaR_99": var_99,
        "ES_95": es_95,
        "ES_99": es_99,
        "mean_loss_ratio": expected_loss / EAD.sum()
    }

    return simulated_losses, risk_measures


# 2. Run simulation
portfolio_1_data = all_portfolios[all_portfolios["portfolio"] == "Portfolio 1"].copy()
portfolio_2_data = all_portfolios[all_portfolios["portfolio"] == "Portfolio 2"].copy()
losses_1, risk_1 = simulate_portfolio_loss(portfolio_1_data, "Portfolio 1")
losses_2, risk_2 = simulate_portfolio_loss(portfolio_2_data, "Portfolio 2")


# 3. Plot loss distributions
losses_1_million = losses_1["loss"] / 1_000_000
losses_2_million = losses_2["loss"] / 1_000_000

# Use the same bins and axes for fair comparison
max_loss = max(losses_1_million.max(), losses_2_million.max())
bins = np.linspace(0, max_loss, 50)

density_1, _ = np.histogram(losses_1_million, bins=bins, density=True)
density_2, _ = np.histogram(losses_2_million, bins=bins, density=True)
max_density = max(density_1.max(), density_2.max())

# Colors similar to lecture slides
color_d = np.array([210, 150, 150]) / 255
color_i = "silver"

# Portfolio 1 plot
plt.figure(figsize=(7, 5))
plt.hist(losses_1_million, bins=bins, color=color_i, density=True, alpha=0.7, label="Portfolio 1")
plt.xlabel("Portfolio Loss (million)")
plt.ylabel("Density")
plt.title("Loss Distribution - Portfolio 1")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, max_loss)
plt.ylim(0, max_density * 1.1)
plt.tight_layout()
plt.savefig(DIR_RESULTS / "loss_distribution_Portfolio_1.png", dpi=300, bbox_inches="tight")
plt.close()

# Portfolio 2 plot
plt.figure(figsize=(7, 5))
plt.hist(losses_2_million, bins=bins, color=color_d, density=True, alpha=0.9, label="Portfolio 2")
plt.xlabel("Portfolio Loss (million)")
plt.ylabel("Density")
plt.title("Loss Distribution - Portfolio 2")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, max_loss)
plt.ylim(0, max_density * 1.1)
plt.tight_layout()
plt.savefig(DIR_RESULTS / "loss_distribution_Portfolio_2.png", dpi=300, bbox_inches="tight")
plt.close()

# Combined comparison plot
# This comparison plot is used in the report
plt.figure(figsize=(10, 6))
plt.hist(losses_2_million, bins=bins, color=color_d, density=True, alpha=0.9, label="Portfolio 2")
plt.hist(losses_1_million, bins=bins, color=color_i, density=True, alpha=0.5, label="Portfolio 1")
plt.xlabel("Portfolio Loss (million)")
plt.ylabel("Density")
plt.title("Simulated Loss Distribution Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, max_loss)
plt.ylim(0, max_density * 1.1)
plt.tight_layout()
plt.savefig(DIR_RESULTS / "loss_distribution_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# 4. Save Part 3 outputs
simulated_losses_all = pd.concat([losses_1, losses_2], ignore_index=True)
risk_measures = pd.DataFrame([risk_1, risk_2])
simulated_losses_all.to_csv(DIR_RESULTS / "part3_simulated_losses.csv", index=False)
risk_measures.to_csv(DIR_RESULTS / "part3_risk_measures.csv", index=False)


# 5. Print results
print("Part 3 risk measures:")
print(risk_measures.to_string(index=False))
