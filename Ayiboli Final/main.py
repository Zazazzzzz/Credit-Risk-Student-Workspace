from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ============================================================
# 0. Settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DIR_DATA = BASE_DIR / "Data"
DIR_RESULTS = BASE_DIR / "results"
DIR_RESULTS.mkdir(exist_ok=True)

N_SIM = 50_000          # number of Monte Carlo simulation runs
SEED = 42               # random seed for reproducibility
CONF_LEVELS = [0.95, 0.99, 0.999]  # confidence levels for VaR and ES


# ============================================================
# 1. Helper functions
# ============================================================

def clean_entity_code(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    x = x.replace("_", "-").replace(" ", "")
    # insert missing hyphen, e.g. BFVUAOK -> B-FVUAOK
    if "-" not in x and len(x) == 7 and x[0] in ["A", "B"]:
        x = x[0] + "-" + x[1:]
    return x


def clean_industry(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower().replace(".", "").replace(" ", "")
    if x in ["industrya", "inda"]:
        return "Industry A"
    if x in ["industryb", "indb"]:
        return "Industry B"
    return np.nan


def clean_region(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower().replace(" ", "")
    if x == "europe":
        return "Europe"
    if x == "asia":
        return "Asia"
    if x == "northamerica":
        return "North America"
    if x == "southamerica":
        return "South America"
    return x.title()


def first_non_missing(x):
    x = x.dropna()
    return x.iloc[0] if len(x) > 0 else np.nan


def save_csv(df, name):
    df.to_csv(DIR_RESULTS / name, index=False)


# ============================================================
# 2. Load raw data
# ============================================================

entities = pd.read_csv(DIR_DATA / "entities_info.csv")
defaults = pd.read_csv(DIR_DATA / "default_history_20y.csv")
industry_returns = pd.read_csv(DIR_DATA / "industry_index_returns.csv")
stock_returns = pd.read_csv(DIR_DATA / "stock_returns.csv")

portfolio_1 = pd.read_csv(DIR_DATA / "portfolio_1.csv")
portfolio_2 = pd.read_csv(DIR_DATA / "portfolio_2.csv")
portfolio_3 = pd.read_csv(DIR_DATA / "portfolio_3.csv")


# # ===================================================================================================
# 3. Part 1: Data cleaning
# # ===================================================================================================

# Entity information
entities["entity_code_clean"] = entities["entity_code"].apply(clean_entity_code)
entities["industry_clean"] = entities["industry"].apply(clean_industry)
entities["region_clean"] = entities["region"].apply(clean_region)
entities["lgd"] = pd.to_numeric(entities["lgd"], errors="coerce")
entities["base_pd_hint"] = pd.to_numeric(entities["base_pd_hint"], errors="coerce")

cleaned_entities = (
    entities
    .dropna(subset=["entity_code_clean"])
    .groupby("entity_code_clean", as_index=False)
    .agg({
        "entity_name": first_non_missing,
        "industry_clean": first_non_missing,
        "region_clean": first_non_missing,
        "lgd": first_non_missing,
        "base_pd_hint": first_non_missing,
        "size_bucket": first_non_missing,
    })
)
# Report: Data section (entity-level cleaning assumptions)
save_csv(cleaned_entities, "cleaned_entities.csv")


# Default history and historical PD
defaults["entity_code_clean"] = defaults["entity_code"].apply(clean_entity_code)
defaults["date_clean"] = pd.to_datetime(defaults["date"], errors="coerce", format="mixed")
defaults["default_event_clean"] = pd.to_numeric(
    defaults["default_event"], errors="coerce"
).fillna(0)

cleaned_defaults = (
    defaults
    .dropna(subset=["entity_code_clean", "date_clean"])
    .groupby(["entity_code_clean", "date_clean"], as_index=False)
    .agg({"default_event_clean": "max"})
)
# Report: Data section (default history cleaning)
save_csv(cleaned_defaults, "cleaned_default_history.csv")

historical_pd_entity = (
    cleaned_defaults
    .groupby("entity_code_clean", as_index=False)
    .agg(years_observed=("date_clean", "count"),
         default_count=("default_event_clean", "sum"))
)
historical_pd_entity["historical_pd"] = (
    historical_pd_entity["default_count"] / historical_pd_entity["years_observed"]
)
# Report: Data section (cross-check for base_pd_hint; not used as PD_clean)
save_csv(historical_pd_entity, "historical_pd_by_entity.csv")

# Pre-compute industry medians for imputation inside create_modeling_table
industry_lgd_median = cleaned_entities.groupby("industry_clean")["lgd"].median()
industry_pd_median = cleaned_entities.groupby("industry_clean")["base_pd_hint"].median()
overall_lgd = cleaned_entities["lgd"].median()
overall_pd = cleaned_entities["base_pd_hint"].median()


def create_modeling_table(portfolio, portfolio_name):
    table = portfolio.copy()
    table["entity_code_clean"] = table["entity_code"].apply(clean_entity_code)
    table["EAD"] = pd.to_numeric(table["EAD"], errors="coerce")
    table["booking_date_clean"] = pd.to_datetime(
        table["booking_date"], errors="coerce", format="mixed"
    )

    table = (
        table
        .dropna(subset=["entity_code_clean"])
        .groupby("entity_code_clean", as_index=False)
        .agg({
            "EAD": lambda x: x.sum(min_count=1),
            "booking_date_clean": first_non_missing,
        })
    )

    table = table.merge(cleaned_entities, on="entity_code_clean", how="left")
    table = table.merge(
        historical_pd_entity[["entity_code_clean", "historical_pd"]],
        on="entity_code_clean", how="left"
    )

    # EAD: replace missing with portfolio median
    table["EAD_clean"] = table["EAD"].fillna(table["EAD"].median())

    # LGD: original value, then industry median, then overall median
    table["LGD_clean"] = table["lgd"].copy()
    table["LGD_clean"] = table["LGD_clean"].fillna(
        table["industry_clean"].map(industry_lgd_median)
    )
    table["LGD_clean"] = table["LGD_clean"].fillna(overall_lgd)

    # PD: base_pd_hint is preferred over historical_pd because many firms
    # have zero observed defaults, making historical_pd unreliable.
    # Fallback: industry median of base_pd_hint, then overall median.
    table["PD_clean"] = table["base_pd_hint"].copy()
    table["PD_clean"] = table["PD_clean"].fillna(
        table["industry_clean"].map(industry_pd_median)
    )
    table["PD_clean"] = table["PD_clean"].fillna(overall_pd).clip(0.0001, 0.50)

    table["portfolio"] = portfolio_name

    columns = [
        "portfolio", "entity_code_clean", "entity_name", "industry_clean",
        "region_clean", "size_bucket", "EAD_clean", "LGD_clean", "PD_clean",
        "historical_pd", "booking_date_clean",
    ]
    return table[columns]


modeling_table_1 = create_modeling_table(portfolio_1, "Portfolio 1")
modeling_table_2 = create_modeling_table(portfolio_2, "Portfolio 2")
modeling_table_3 = create_modeling_table(portfolio_3, "Portfolio 3")

# Report: Data section (modeling tables, Table 1)
save_csv(modeling_table_1, "modeling_table_portfolio_1.csv")
save_csv(modeling_table_2, "modeling_table_portfolio_2.csv")
save_csv(modeling_table_3, "modeling_table_portfolio_3.csv")


# Stock and industry return data
cleaned_stocks = stock_returns.copy()
cleaned_stocks["entity_code_clean"] = cleaned_stocks["entity_code"].apply(clean_entity_code)
cleaned_stocks["date_clean"] = pd.to_datetime(cleaned_stocks["date"], errors="coerce", format="mixed")
cleaned_stocks["stock_return"] = pd.to_numeric(cleaned_stocks["stock_return"], errors="coerce")
cleaned_stocks = cleaned_stocks.dropna(subset=["entity_code_clean", "date_clean", "stock_return"])
# Report: Data section (stock returns, used for rho estimation in Part 2)
save_csv(cleaned_stocks, "cleaned_stock_returns.csv")

cleaned_industry = industry_returns.copy()
cleaned_industry["industry_clean"] = cleaned_industry["industry"].apply(clean_industry)
cleaned_industry["date_clean"] = pd.to_datetime(cleaned_industry["date"], errors="coerce", format="mixed")
cleaned_industry["index_return"] = pd.to_numeric(cleaned_industry["index_return"], errors="coerce")
cleaned_industry = cleaned_industry.dropna(subset=["industry_clean", "date_clean", "index_return"])
# Report: Data section (industry index returns, used for rho estimation in Part 2)
save_csv(cleaned_industry, "cleaned_industry_index_returns.csv")


data_quality_summary = pd.DataFrame({
    "dataset": [
        "entities", "default_history", "portfolio_1", "portfolio_2",
        "portfolio_3", "stock_returns", "industry_index_returns",
    ],
    "raw_rows": [
        len(entities), len(defaults), len(portfolio_1), len(portfolio_2),
        len(portfolio_3), len(stock_returns), len(industry_returns),
    ],
    "cleaned_rows": [
        len(cleaned_entities), len(cleaned_defaults), len(modeling_table_1),
        len(modeling_table_2), len(modeling_table_3), len(cleaned_stocks),
        len(cleaned_industry),
    ],
})
# Report: Data section, Table 1 (data quality summary)
save_csv(data_quality_summary, "data_quality_summary.csv")

print("Part 1 completed.")


# # ===================================================================================================
# 4. Part 2: Model setup and parameter estimation
# # ===================================================================================================

def estimate_rho(entity_code, industry_name):
    # rho is estimated as the absolute correlation between a firm's stock
    # return and its own industry index return.
    entity_ret = (
        cleaned_stocks[cleaned_stocks["entity_code_clean"] == entity_code]
        .groupby("date_clean")["stock_return"]
        .mean()
    )

    industry_ret = (
        cleaned_industry[cleaned_industry["industry_clean"] == industry_name]
        .groupby("date_clean")["index_return"]
        .mean()
    )

    joint = (
        entity_ret.to_frame("stock_return")
        .join(industry_ret.to_frame("index_return"), how="inner")
        .dropna()
    )

    # Fewer than 10 overlapping observations make the OLS estimate unreliable.
    # Fall back to rho = 0.30, a conservative default corresponding to an
    # implied asset correlation of rho^2 = 9%, below the Basel II corporate
    # benchmark range of 12-24%.
    if len(joint) < 10:
        return 0.30

    _, _, r_value, _, _ = scipy_stats.linregress(
        joint["index_return"], joint["stock_return"]
    )

    return abs(r_value)


def add_rho(table):
    table = table.copy()
    table["rho"] = [
        estimate_rho(row["entity_code_clean"], row["industry_clean"])
        for _, row in table.iterrows()
    ]
    return table


print("Estimating factor loadings for Portfolio 1...")
model_inputs_1 = add_rho(modeling_table_1)
print("Estimating factor loadings for Portfolio 2...")
model_inputs_2 = add_rho(modeling_table_2)

# Report: Methodology section (per-entity model inputs after estimation)
save_csv(model_inputs_1, "model_inputs_portfolio_1.csv")
save_csv(model_inputs_2, "model_inputs_portfolio_2.csv")


parameter_summary = pd.DataFrame({
    "portfolio": ["Portfolio 1", "Portfolio 2"],
    "number_of_obligors": [len(model_inputs_1), len(model_inputs_2)],
    "total_EAD": [
        model_inputs_1["EAD_clean"].sum(),
        model_inputs_2["EAD_clean"].sum(),
    ],
    "weighted_PD": [
        np.average(model_inputs_1["PD_clean"], weights=model_inputs_1["EAD_clean"]),
        np.average(model_inputs_2["PD_clean"], weights=model_inputs_2["EAD_clean"]),
    ],
    "weighted_LGD": [
        np.average(model_inputs_1["LGD_clean"], weights=model_inputs_1["EAD_clean"]),
        np.average(model_inputs_2["LGD_clean"], weights=model_inputs_2["EAD_clean"]),
    ],
    "mean_rho": [
        model_inputs_1["rho"].mean(),
        model_inputs_2["rho"].mean(),
    ],
})
parameter_summary["approx_asset_correlation"] = parameter_summary["mean_rho"] ** 2
# Report: Methodology section, Table 2 (portfolio-level parameter summary)
save_csv(parameter_summary, "parameter_summary.csv")

print("Part 2 completed.")


# ===================================================================================================
# 5. Part 3: Simulation and risk metrics
# ===================================================================================================

def portfolio_arrays(table):
    return (
        table["EAD_clean"].to_numpy(),
        table["LGD_clean"].to_numpy(),
        table["PD_clean"].to_numpy(),
        table["rho"].to_numpy(),
    )


def simulate_independent(table, n_sim=N_SIM, seed=SEED):
    # Each obligor defaults independently with probability PD_i.
    # When rho=0 in the One-Factor model, the systematic factor F drops out
    # and X_i = eps_i, so each obligor defaults independently with probability
    # PD_i. This Bernoulli draw is the direct equivalent of setting rho=0
    # in the correlated model (Portfolio_Threshold.py lines 80-81).
    ead, lgd, pd_values, _ = portfolio_arrays(table)
    rng = np.random.default_rng(seed)
    defaults_sim = rng.binomial(1, pd_values, size=(n_sim, len(table)))
    losses = defaults_sim * ead * lgd
    return losses.sum(axis=1) / ead.sum()


def simulate_one_factor(table, n_sim=N_SIM, seed=SEED):
    # One-Factor Gaussian Threshold: X_i = rho*F + sqrt(1-rho^2)*eps_i
    # Default when X_i <= Phi^{-1}(PD_i).
    # Simulation structure follows Credit Metrics.py (lines 163-175).
    # common_factor has shape (n_sim, 1) so that a single draw of F is
    # shared across all obligors in each scenario via NumPy broadcasting;
    # see Portfolio_Threshold.py (lines 52-68) for a detailed explanation.
    ead, lgd, pd_values, rho = portfolio_arrays(table)
    rng = np.random.default_rng(seed)
    common_factor = rng.normal(size=(n_sim, 1))
    idiosyncratic = rng.normal(size=(n_sim, len(table)))
    credit_quality = rho * common_factor + np.sqrt(1 - rho ** 2) * idiosyncratic
    default_threshold = scipy_stats.norm.ppf(pd_values)
    defaults_sim = credit_quality < default_threshold
    losses = defaults_sim * ead * lgd
    return losses.sum(axis=1) / ead.sum()


def risk_metrics(loss_rates, portfolio, model):
    # Risk measures follow Credit Metrics.py (lines 182-204).
    row = {
        "portfolio": portfolio,
        "model": model,
        "EL": np.mean(loss_rates),
        "Std": np.std(loss_rates),
    }
    for level in CONF_LEVELS:
        var = np.quantile(loss_rates, level)
        # ES = E[L | L > VaR]: strict inequality excludes the VaR observation.
        es = loss_rates[loss_rates > var].mean()
        row[f"VaR_{int(level * 1000)}"] = var
        row[f"ES_{int(level * 1000)}"] = es
    return row


print(f"Running simulations (N = {N_SIM:,}, seed = {SEED})...")

simulation_results = {
    ("Portfolio 1", "Independent Bernoulli"): simulate_independent(model_inputs_1, seed=SEED),
    ("Portfolio 2", "Independent Bernoulli"): simulate_independent(model_inputs_2, seed=SEED + 1),
    ("Portfolio 1", "One-Factor Gaussian"): simulate_one_factor(model_inputs_1, seed=SEED + 2),
    ("Portfolio 2", "One-Factor Gaussian"): simulate_one_factor(model_inputs_2, seed=SEED + 3),
}

risk_measures = pd.DataFrame([
    risk_metrics(loss_rates, portfolio, model)
    for (portfolio, model), loss_rates in simulation_results.items()
])
# Report: Results and Discussion section, Table 3 (EL, VaR, ES for all scenarios)
save_csv(risk_measures.round(6), "risk_measures.csv")

print("Part 3 completed.")


# # ===================================================================================================
# 6. Part 4: Plots for report
# # ===================================================================================================

BINS = 60
rm = risk_measures.set_index(["portfolio", "model"])


def plot_loss_distribution(model, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Loss Distribution: {model}", fontweight="bold")
    for ax, portfolio in zip(axes, ["Portfolio 1", "Portfolio 2"]):
        values = simulation_results[(portfolio, model)]
        el = rm.loc[(portfolio, model), "EL"]
        var99 = rm.loc[(portfolio, model), "VaR_990"]
        ax.hist(values, bins=BINS, density=True, alpha=0.65, color="steelblue")
        ax.axvline(el, color="green", ls="--", lw=1.5, label=f"EL = {el:.4f}")
        ax.axvline(var99, color="darkred", ls="--", lw=1.5, label=f"VaR 99% = {var99:.4f}")
        ax.set_title(portfolio)
        ax.set_xlabel("Portfolio loss rate")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / filename, dpi=150)
    plt.close()


def plot_model_comparison(portfolio, filename):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"Independent Bernoulli": "darkorange", "One-Factor Gaussian": "steelblue"}
    for model, color in colors.items():
        values = simulation_results[(portfolio, model)]
        var99 = rm.loc[(portfolio, model), "VaR_990"]
        ax.hist(values, bins=BINS, density=True, alpha=0.55, color=color, label=model)
        ax.axvline(var99, color=color, ls="--", lw=2.0,
                   label=f"VaR 99% = {var99:.4f}")
    ax.set_title(f"{portfolio} - Impact of Default Correlation")
    ax.set_xlabel("Portfolio loss rate")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(DIR_RESULTS / filename, dpi=150)
    plt.close()


# Saved to results/ (not included in report)
plot_loss_distribution("Independent Bernoulli", "loss_dist_indep_bernoulli.png")
# Saved to results/ (not included in report)
plot_loss_distribution("One-Factor Gaussian", "loss_dist_one_factor.png")
# Report: Results and Discussion, Figure 1 (Section 4.2 – Portfolio 1 model comparison)
plot_model_comparison("Portfolio 1", "model_comparison_p1.png")
# Saved to results/ (not included in report)
plot_model_comparison("Portfolio 2", "model_comparison_p2.png")


# Bar chart: VaR 99% across all scenarios
plot_data = risk_measures.copy()
plot_data["label"] = plot_data["portfolio"] + "\n" + plot_data["model"]

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#f4a261", "#e76f51", "#457b9d", "#1d3557"]
bars = ax.bar(plot_data["label"], plot_data["VaR_990"] * 100,
              color=bar_colors, alpha=0.85)
for bar, val in zip(bars, plot_data["VaR_990"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("99% VaR Comparison - All Scenarios")
ax.set_ylabel("Loss rate (%)")
plt.tight_layout()
# Saved to results/ (not included in report)
plt.savefig(DIR_RESULTS / "risk_measures_bar.png", dpi=150)
plt.close()


# Portfolio comparison under the One-Factor model
of_rows = risk_measures[risk_measures["model"] == "One-Factor Gaussian"].copy()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(of_rows["portfolio"], of_rows["VaR_990"] * 100,
       color=["steelblue", "crimson"], alpha=0.85)
ax.set_title("Portfolio Comparison - One-Factor Gaussian Model")
ax.set_ylabel("99% VaR loss rate (%)")
plt.tight_layout()
# Saved to results/ (not included in report)
plt.savefig(DIR_RESULTS / "portfolio_comparison.png", dpi=150)
plt.close()

print("Part 4 completed.")
print(f"All outputs saved in: {DIR_RESULTS}")
