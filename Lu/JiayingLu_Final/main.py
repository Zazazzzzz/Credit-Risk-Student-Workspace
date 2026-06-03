# ============================================================
# Portfolio 2 and portfolio 1
# 1. Read raw data
# 2. Inspect raw datasets
# 3. Clean entity information
# 4. Clean each portfolio
# 5. Merge portfolio data with entity information
# 6. Handle missing values
# 7. Estimate PD and rho
# 8. Simulate one-year loss distribution
# 9. Compute EL, VaR, and ES
# 10. Save cleaned tables, result tables, and plots
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# 1. Set file paths
# ============================================================

# This Python file should be placed inside your "Lu" folder.
current_folder = Path(__file__).resolve().parent

# The project folder is one level above "Lu".
project_folder = current_folder.parent

# The raw data folder.
data_folder = project_folder / "Final Assignment" / "Data"

# The output folder.
results_folder = current_folder / "results"
results_folder.mkdir(parents=True, exist_ok=True)

print("Current folder:", current_folder)
print("Project folder:", project_folder)
print("Data folder:", data_folder)
print("Results folder:", results_folder)


# ============================================================
# 2. Read raw CSV files
# ============================================================

portfolio1_raw = pd.read_csv(data_folder / "portfolio_1.csv")
portfolio2_raw = pd.read_csv(data_folder / "portfolio_2.csv")
entities_raw = pd.read_csv(data_folder / "entities_info.csv")
stock_returns_raw = pd.read_csv(data_folder / "stock_returns.csv")
industry_returns_raw = pd.read_csv(data_folder / "industry_index_returns.csv")
default_history_raw = pd.read_csv(data_folder / "default_history_20y.csv")


# ============================================================
# 3. Initial inspection function
# ============================================================

def inspect_data(df, name):
    """
    Print basic information about a raw dataset.
    """
    print("\n" + "=" * 60)
    print(f"Dataset name: {name}")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nNumber of duplicated rows:")
    print(df.duplicated().sum())


inspect_data(portfolio1_raw, "portfolio_1.csv")
inspect_data(portfolio2_raw, "portfolio_2.csv")
inspect_data(entities_raw, "entities_info.csv")
inspect_data(stock_returns_raw, "stock_returns.csv")
inspect_data(industry_returns_raw, "industry_index_returns.csv")
inspect_data(default_history_raw, "default_history_20y.csv")


# ============================================================
# 4. Helper functions
# ============================================================

def standardize_entity_code(x):
    """
    Standardize entity code.

    Examples:
    a-h8wkws    -> A-H8WKWS
    A - W20E3T  -> A-W20E3T
    BWMAEKB     -> B-WMAEKB
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()

    # Remove spaces around hyphen.
    x = re.sub(r"\s*-\s*", "-", x)

    # Remove other internal spaces.
    x = re.sub(r"\s+", "", x)

    # If the code has no hyphen, insert a hyphen after the first letter.
    # Example: BWMAEKB -> B-WMAEKB
    if "-" not in x and re.match(r"^[A-Z][A-Z0-9]{6}$", x):
        x = x[0] + "-" + x[1:]

    return x


def standardize_industry(x):
    """
    Standardize industry labels.

    Examples:
    Ind. A, IndustryA, industry a -> Industry A
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()
    x = x.replace(".", "")
    x = re.sub(r"\s+", " ", x)

    if x in ["ind a", "industry a", "industrya"]:
        return "Industry A"

    if x in ["ind b", "industry b", "industryb"]:
        return "Industry B"

    if x in ["ind c", "industry c", "industryc"]:
        return "Industry C"

    if x in ["ind d", "industry d", "industryd"]:
        return "Industry D"

    if x in ["ind e", "industry e", "industrye"]:
        return "Industry E"

    return x.title()


def standardize_size_bucket(x):
    """
    Standardize size bucket labels.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)

    return x.title()


def convert_rate_to_decimal(series):
    """
    Convert PD or LGD values into decimal format.

    Examples:
    0.4  -> 0.4
    40   -> 0.4
    40%  -> 0.4
    """
    series = series.astype(str).str.strip()

    has_percent = series.str.contains("%", regex=False)

    clean_series = (
        series
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    numeric_series = pd.to_numeric(clean_series, errors="coerce")

    # If the original value contains %, divide by 100.
    numeric_series = numeric_series.where(~has_percent, numeric_series / 100)

    # If a rate is written as 40 instead of 0.40, convert it to 0.40.
    numeric_series = numeric_series.where(~(numeric_series > 1), numeric_series / 100)

    return numeric_series


def parse_mixed_dates(date_series):
    """
    Parse mixed date formats.

    Examples:
    2023-01-02
    04-01-2023
    03/01/2023
    """
    s = date_series.astype(str).str.strip()

    dates = pd.Series(pd.NaT, index=s.index)

    # ISO format: 2023-01-02
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")

    dates.loc[iso_mask] = pd.to_datetime(
        s.loc[iso_mask],
        errors="coerce"
    )

    # Other formats such as 04-01-2023 or 03/01/2023.
    dates.loc[~iso_mask] = pd.to_datetime(
        s.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    return dates


# ============================================================
# 5. Clean entities_info.csv
# ============================================================

entities = entities_raw.copy()

entities["entity_code"] = entities["entity_code"].apply(standardize_entity_code)
entities["industry"] = entities["industry"].apply(standardize_industry)
entities["size_bucket"] = entities["size_bucket"].apply(standardize_size_bucket)

entities["lgd"] = convert_rate_to_decimal(entities["lgd"])
entities["pd"] = convert_rate_to_decimal(entities["base_pd_hint"])

entity_duplicates_before = entities.duplicated().sum()
entities = entities.drop_duplicates()

print("\nDuplicated rows removed from entities_info:")
print(entity_duplicates_before)

print("\nCleaned industry labels:")
print(entities["industry"].value_counts())

print("\nMissing values before integration:")
print(entities[["industry", "size_bucket", "lgd", "pd"]].isna().sum())

entities_clean = entities[
    [
        "entity_code",
        "entity_name",
        "industry",
        "region",
        "size_bucket",
        "lgd",
        "pd"
    ]
].copy()

entity_code_duplicates_before = entities_clean.duplicated(subset=["entity_code"]).sum()
entities_clean = entities_clean.drop_duplicates(subset=["entity_code"], keep="first")

print("\nDuplicated entity_code rows removed from entities_clean:")
print(entity_code_duplicates_before)


# ============================================================
# 6. Estimate historical PD from default_history_20y.csv
# ============================================================

default_history = default_history_raw.copy()

default_history["entity_code"] = default_history["entity_code"].apply(standardize_entity_code)
default_history["date"] = parse_mixed_dates(default_history["date"])
default_history["default_event"] = pd.to_numeric(
    default_history["default_event"],
    errors="coerce"
)

default_history_duplicates_before = default_history.duplicated().sum()
default_history = default_history.drop_duplicates()

default_history = default_history.dropna(subset=["default_event"])
default_history["default_event"] = (default_history["default_event"] > 0).astype(int)

historical_pd_entity = (
    default_history
    .groupby("entity_code", as_index=False)
    .agg(
        historical_pd=("default_event", "mean"),
        default_obs_count=("default_event", "count")
    )
)

print("\nDuplicated rows removed from default_history:")
print(default_history_duplicates_before)

print("\nHistorical PD table:")
print(historical_pd_entity.head())


# ============================================================
# 7. Estimate rho using stock returns and industry returns
# ============================================================

stock_returns = stock_returns_raw.copy()

stock_returns["entity_code"] = stock_returns["entity_code"].apply(standardize_entity_code)
stock_returns["date"] = parse_mixed_dates(stock_returns["date"])
stock_returns["stock_return"] = pd.to_numeric(
    stock_returns["stock_return"],
    errors="coerce"
)

stock_returns = stock_returns.drop_duplicates()

industry_returns = industry_returns_raw.copy()

industry_returns["industry"] = industry_returns["industry"].apply(standardize_industry)
industry_returns["date"] = parse_mixed_dates(industry_returns["date"])
industry_returns["index_return"] = pd.to_numeric(
    industry_returns["index_return"],
    errors="coerce"
)

industry_returns = industry_returns.drop_duplicates()

stock_returns = stock_returns.merge(
    entities_clean[["entity_code", "industry"]],
    on="entity_code",
    how="left"
)

regression_data = stock_returns.merge(
    industry_returns,
    on=["date", "industry"],
    how="left"
)


def estimate_rho(group):
    """
    Estimate rho from the correlation between entity stock return
    and the corresponding industry index return.

    Here rho is treated as the factor loading in:
    X_i = rho_i F + sqrt(1 - rho_i^2) epsilon_i
    """
    group = group.dropna(subset=["stock_return", "index_return"])

    if len(group) < 30:
        return pd.Series({
            "return_correlation": np.nan,
            "rho": np.nan,
            "return_obs_count": len(group)
        })

    corr = group["stock_return"].corr(group["index_return"])

    return pd.Series({
        "return_correlation": corr,
        "rho": abs(corr),
        "return_obs_count": len(group)
    })


rho_table = (
    regression_data
    .groupby("entity_code")
    .apply(estimate_rho)
    .reset_index()
)

print("\nEstimated rho table:")
print(rho_table.head())


# ============================================================
# 8. Function: build modelling table for one portfolio
# ============================================================

def build_modeling_table(portfolio_raw, portfolio_name):
    """
    Clean one portfolio, merge it with entity information,
    handle missing values, and create a modelling table.
    """
    print("\n" + "=" * 70)
    print(f"Building modelling table for {portfolio_name}")
    print("=" * 70)

    portfolio = portfolio_raw.copy()

    portfolio["entity_code"] = portfolio["entity_code"].apply(standardize_entity_code)
    portfolio["EAD"] = pd.to_numeric(portfolio["EAD"], errors="coerce")
    portfolio["booking_date"] = pd.to_datetime(
        portfolio["booking_date"],
        errors="coerce"
    )

    portfolio_duplicates_before = portfolio.duplicated().sum()
    portfolio = portfolio.drop_duplicates()

    print(f"\nDuplicated rows removed from {portfolio_name}:")
    print(portfolio_duplicates_before)

    portfolio["ead_missing_flag"] = portfolio["EAD"].isna().astype(int)

    print(f"\nRows with missing EAD before imputation in {portfolio_name}:")
    print(portfolio[portfolio["ead_missing_flag"] == 1])

    modeling_table = portfolio.merge(
        entities_clean,
        on="entity_code",
        how="left",
        indicator=True
    )

    unmatched_entities = modeling_table[modeling_table["_merge"] == "left_only"]

    print(f"\nNumber of unmatched entities after merge for {portfolio_name}:")
    print(len(unmatched_entities))

    print(f"\nUnmatched entities in {portfolio_name}:")
    print(unmatched_entities["entity_code"].tolist())

    modeling_table = modeling_table.drop(columns=["_merge"])

    print(f"\nMissing values after integration and before imputation for {portfolio_name}:")
    print(modeling_table[["EAD", "industry", "size_bucket", "lgd", "pd"]].isna().sum())

    modeling_table["ead_missing_flag"] = modeling_table["EAD"].isna().astype(int)
    modeling_table["lgd_missing_flag"] = modeling_table["lgd"].isna().astype(int)
    modeling_table["pd_missing_flag"] = modeling_table["pd"].isna().astype(int)

    print(f"\nMissing flags before imputation for {portfolio_name}:")
    print(modeling_table[[
        "ead_missing_flag",
        "lgd_missing_flag",
        "pd_missing_flag"
    ]].sum())

    # --------------------------------------------------------
    # EAD imputation: size_bucket average, then overall average
    # --------------------------------------------------------
    ead_missing_before = modeling_table["EAD"].isna().sum()

    modeling_table["EAD"] = modeling_table["EAD"].fillna(
        modeling_table.groupby("size_bucket")["EAD"].transform("mean")
    )

    modeling_table["EAD"] = modeling_table["EAD"].fillna(
        modeling_table["EAD"].mean()
    )

    ead_missing_after = modeling_table["EAD"].isna().sum()

    print(f"\nEAD missing before imputation for {portfolio_name}:", ead_missing_before)
    print(f"EAD missing after imputation for {portfolio_name}:", ead_missing_after)

    print(f"\nRows with originally missing EAD in {portfolio_name}:")
    print(modeling_table.loc[
        modeling_table["ead_missing_flag"] == 1,
        ["entity_code", "EAD", "size_bucket", "ead_missing_flag"]
    ])

    # --------------------------------------------------------
    # LGD imputation: industry average, then overall average
    # --------------------------------------------------------
    lgd_missing_before = modeling_table["lgd"].isna().sum()

    modeling_table["lgd"] = modeling_table["lgd"].fillna(
        modeling_table.groupby("industry")["lgd"].transform("mean")
    )

    modeling_table["lgd"] = modeling_table["lgd"].fillna(
        modeling_table["lgd"].mean()
    )

    modeling_table["lgd"] = modeling_table["lgd"].clip(lower=0, upper=1)

    lgd_missing_after = modeling_table["lgd"].isna().sum()

    print(f"\nLGD missing before imputation for {portfolio_name}:", lgd_missing_before)
    print(f"LGD missing after imputation for {portfolio_name}:", lgd_missing_after)

    print(f"\nRows with originally missing LGD in {portfolio_name}:")
    print(modeling_table.loc[
        modeling_table["lgd_missing_flag"] == 1,
        ["entity_code", "lgd", "industry", "lgd_missing_flag"]
    ])

    # --------------------------------------------------------
    # PD imputation:
    # 1. Use base PD if available.
    # 2. If missing, use historical default frequency.
    # 3. If still missing, use industry average PD.
    # 4. If still missing, use minimum positive PD.
    # 5. If PD equals zero, clip it to a very small positive value.
    # --------------------------------------------------------
    modeling_table = modeling_table.merge(
        historical_pd_entity,
        on="entity_code",
        how="left"
    )

    pd_missing_before = modeling_table["pd"].isna().sum()

    modeling_table["pd"] = modeling_table["pd"].fillna(
        modeling_table["historical_pd"]
    )

    modeling_table["pd"] = modeling_table["pd"].fillna(
        modeling_table.groupby("industry")["pd"].transform("mean")
    )

    min_positive_pd = modeling_table.loc[
        modeling_table["pd"] > 0,
        "pd"
    ].min()

    modeling_table["pd"] = modeling_table["pd"].fillna(min_positive_pd)

    # Avoid zero PD because norm.ppf(0) is invalid.
    modeling_table["pd"] = modeling_table["pd"].clip(lower=1e-6, upper=1)

    pd_missing_after = modeling_table["pd"].isna().sum()

    print(f"\nPD missing before imputation for {portfolio_name}:", pd_missing_before)
    print(f"PD missing after imputation for {portfolio_name}:", pd_missing_after)
    print(f"Minimum positive PD used as fallback for {portfolio_name}:", min_positive_pd)

    print(f"\nRows with originally missing PD in {portfolio_name}:")
    print(modeling_table.loc[
        modeling_table["pd_missing_flag"] == 1,
        ["entity_code", "pd", "historical_pd", "industry", "pd_missing_flag"]
    ])

    # --------------------------------------------------------
    # Merge rho
    # --------------------------------------------------------
    modeling_table = modeling_table.merge(
        rho_table,
        on="entity_code",
        how="left"
    )

    rho_missing_before = modeling_table["rho"].isna().sum()

    modeling_table["rho"] = modeling_table["rho"].fillna(
        modeling_table.groupby("industry")["rho"].transform("mean")
    )

    modeling_table["rho"] = modeling_table["rho"].fillna(
        modeling_table["rho"].mean()
    )

    modeling_table["rho"] = modeling_table["rho"].clip(lower=0, upper=0.9999)

    rho_missing_after = modeling_table["rho"].isna().sum()

    print(f"\nrho missing before imputation for {portfolio_name}:", rho_missing_before)
    print(f"rho missing after imputation for {portfolio_name}:", rho_missing_after)

    print(f"\nMissing values after all treatments for {portfolio_name}:")
    print(modeling_table[["EAD", "pd", "lgd", "rho"]].isna().sum())

    # Keep only positive EAD rows.
    modeling_table = modeling_table[modeling_table["EAD"] > 0].copy()

    total_ead = modeling_table["EAD"].sum()
    modeling_table["weight"] = modeling_table["EAD"] / total_ead

    print(f"\nTotal EAD for {portfolio_name}:")
    print(total_ead)

    print(f"\nSum of weights for {portfolio_name}:")
    print(modeling_table["weight"].sum())

    return modeling_table


# ============================================================
# 9. Function: simulate one portfolio
# ============================================================

def simulate_portfolio_loss(modeling_table, portfolio_name, num_trials=100000, random_seed=42):
    """
    Simulate one-year loss distribution under a one-factor Gaussian threshold model.
    """
    print("\n" + "=" * 70)
    print(f"Simulating one-year loss distribution for {portfolio_name}")
    print("=" * 70)

    modeling_table = modeling_table.copy()

    modeling_table["pd"] = modeling_table["pd"].clip(
        lower=1e-6,
        upper=1 - 1e-6
    )

    modeling_table["lgd"] = modeling_table["lgd"].clip(
        lower=0,
        upper=1
    )

    modeling_table["rho"] = modeling_table["rho"].clip(
        lower=0,
        upper=0.9999
    )

    print(f"\nFinal check before simulation for {portfolio_name}:")
    print(modeling_table[["EAD", "weight", "pd", "lgd", "rho"]].isna().sum())

    total_ead = modeling_table["EAD"].sum()

    print(f"\nTotal EAD for {portfolio_name}:")
    print(total_ead)

    print(f"\nSum of weights for {portfolio_name}:")
    print(modeling_table["weight"].sum())

    ead_values = modeling_table["EAD"].to_numpy()
    lgd_values = modeling_table["lgd"].to_numpy()
    pd_values = modeling_table["pd"].to_numpy()
    rho_values = modeling_table["rho"].to_numpy()

    num_obligors = len(modeling_table)

    print(f"\nNumber of obligors in {portfolio_name}:")
    print(num_obligors)

    # Default threshold: d_i = Phi^{-1}(PD_i)
    default_thresholds = norm.ppf(pd_values)

    print(f"\nDefault thresholds for {portfolio_name}:")
    print(default_thresholds)

    np.random.seed(random_seed)

    # Common systematic factor.
    F = np.random.normal(0, 1, size=(num_trials, 1))

    # Idiosyncratic shocks.
    epsilon = np.random.normal(0, 1, size=(num_trials, num_obligors))

    # Latent credit variable:
    # X_i = rho_i F + sqrt(1 - rho_i^2) epsilon_i
    X = rho_values * F + np.sqrt(1 - rho_values ** 2) * epsilon

    # Default indicator.
    default_matrix = (X <= default_thresholds).astype(int)

    print(f"\nDefault matrix shape for {portfolio_name}:")
    print(default_matrix.shape)

    # Loss_i = EAD_i * LGD_i * Default_i
    loss_matrix = default_matrix * ead_values * lgd_values

    portfolio_losses = loss_matrix.sum(axis=1)
    portfolio_loss_rates = portfolio_losses / total_ead
    default_counts = default_matrix.sum(axis=1)

    analytical_el_amount = np.sum(ead_values * lgd_values * pd_values)
    analytical_el_rate = analytical_el_amount / total_ead

    simulated_el_amount = portfolio_losses.mean()
    simulated_el_rate = portfolio_loss_rates.mean()

    risk_levels = [0.95, 0.99]
    risk_results = []

    for alpha in risk_levels:
        var_amount = np.quantile(portfolio_losses, alpha)
        var_rate = np.quantile(portfolio_loss_rates, alpha)

        tail_mask = portfolio_losses >= var_amount

        es_amount = portfolio_losses[tail_mask].mean()
        es_rate = portfolio_loss_rates[tail_mask].mean()

        risk_results.append({
            "portfolio": portfolio_name,
            "confidence_level": alpha,
            "VaR_amount": var_amount,
            "VaR_rate": var_rate,
            "ES_amount": es_amount,
            "ES_rate": es_rate
        })

    risk_results_df = pd.DataFrame(risk_results)

    summary_results = pd.DataFrame({
        "portfolio": [portfolio_name] * 7,
        "metric": [
            "total_EAD",
            "analytical_expected_loss_amount",
            "analytical_expected_loss_rate",
            "simulated_expected_loss_amount",
            "simulated_expected_loss_rate",
            "mean_number_of_defaults",
            "max_number_of_defaults"
        ],
        "value": [
            total_ead,
            analytical_el_amount,
            analytical_el_rate,
            simulated_el_amount,
            simulated_el_rate,
            default_counts.mean(),
            default_counts.max()
        ]
    })

    print(f"\nSummary results for {portfolio_name}:")
    print(summary_results)

    print(f"\nRisk measures for {portfolio_name}:")
    print(risk_results_df)

    safe_name = portfolio_name.lower().replace(" ", "")

    modeling_table.to_csv(
        results_folder / f"{safe_name}_final_modeling_table.csv",
        index=False
    )

    summary_results.to_csv(
        results_folder / f"{safe_name}_summary_results.csv",
        index=False
    )

    risk_results_df.to_csv(
        results_folder / f"{safe_name}_risk_measures.csv",
        index=False
    )

    simulation_outputs = pd.DataFrame({
        "portfolio_loss": portfolio_losses,
        "portfolio_loss_rate": portfolio_loss_rates,
        "number_of_defaults": default_counts
    })

    simulation_outputs.to_csv(
        results_folder / f"{safe_name}_simulation_outputs.csv",
        index=False
    )

    print(f"\nSaved files for {portfolio_name}:")
    print(results_folder / f"{safe_name}_final_modeling_table.csv")
    print(results_folder / f"{safe_name}_summary_results.csv")
    print(results_folder / f"{safe_name}_risk_measures.csv")
    print(results_folder / f"{safe_name}_simulation_outputs.csv")

    # --------------------------------------------------------
    # Plot 1: Loss amount distribution
    #  Additional output, saved in results folder
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.hist(
        portfolio_losses,
        bins=50,
        density=True
    )

    plt.xlabel("Portfolio Loss")
    plt.ylabel("Density")
    plt.title(f"Simulated One-Year Loss Distribution of {portfolio_name}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    amount_plot_path = results_folder / f"{safe_name}_loss_distribution.png"
    # Additional output:
    # This figure shows the simulated loss amount distribution.
    # It is saved in the results folder but is not the main figure used in the report.
    plt.savefig(amount_plot_path, dpi=300)
    plt.close()

    print("\nLoss amount distribution plot saved to:")
    print(amount_plot_path)

    # --------------------------------------------------------
    # Plot 2 loss rate distribution with VaR lines
    # This is the main report figure
    # --------------------------------------------------------
    var_95_rate = risk_results_df.loc[
        risk_results_df["confidence_level"] == 0.95,
        "VaR_rate"
    ].iloc[0]

    var_99_rate = risk_results_df.loc[
        risk_results_df["confidence_level"] == 0.99,
        "VaR_rate"
    ].iloc[0]

    plt.figure(figsize=(10, 6))

    plt.hist(
        portfolio_loss_rates * 100,
        bins=50,
        density=True
    )

    plt.axvline(
        var_95_rate * 100,
        linestyle="--",
        linewidth=2,
        label=f"95% VaR = {var_95_rate:.2%}"
    )

    plt.axvline(
        var_99_rate * 100,
        linestyle="--",
        linewidth=2,
        label=f"99% VaR = {var_99_rate:.2%}"
    )

    plt.xlabel("Portfolio Loss Rate (%)")
    plt.ylabel("Density")
    plt.title(f"Simulated One-Year Loss Rate Distribution of {portfolio_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    rate_plot_path = results_folder / f"{safe_name}_loss_rate_distribution.png"

    # Main report figures:
    # If portfolio_name == "Portfolio1":
    #     Figure 1. Simulated One-Year Loss Rate Distribution of Portfolio 1
    #     Saved as: results/portfolio1_loss_rate_distribution.png
    # If portfolio_name == "Portfolio2":
    #     Figure 2. Simulated One-Year Loss Rate Distribution of Portfolio 2
    #     Saved as: results/portfolio2_loss_rate_distribution.png
    plt.savefig(rate_plot_path, dpi=300)
    plt.close()

    print("\nLoss rate distribution plot saved to:")
    print(rate_plot_path)

    return {
        "modeling_table": modeling_table,
        "summary_results": summary_results,
        "risk_results": risk_results_df,
        "portfolio_losses": portfolio_losses,
        "portfolio_loss_rates": portfolio_loss_rates,
        "default_counts": default_counts
    }

# ============================================================
# 10. Run Portfolio 1 and Portfolio 2
# ============================================================

modeling_table_p1 = build_modeling_table(
    portfolio_raw=portfolio1_raw,
    portfolio_name="Portfolio1"
)

results_p1 = simulate_portfolio_loss(
    modeling_table=modeling_table_p1,
    portfolio_name="Portfolio1",
    num_trials=100000,
    random_seed=42
)

modeling_table_p2 = build_modeling_table(
    portfolio_raw=portfolio2_raw,
    portfolio_name="Portfolio2"
)

results_p2 = simulate_portfolio_loss(
    modeling_table=modeling_table_p2,
    portfolio_name="Portfolio2",
    num_trials=100000,
    random_seed=42
)

# ============================================================
# 11. Compare Portfolio 1 and Portfolio 2
# ============================================================

summary_comparison = pd.concat(
    [
        results_p1["summary_results"],
        results_p2["summary_results"]
    ],
    ignore_index=True
)

risk_comparison = pd.concat(
    [
        results_p1["risk_results"],
        results_p2["risk_results"]
    ],
    ignore_index=True
)

summary_comparison.to_csv(
    results_folder / "portfolio_summary_comparison.csv",
    index=False
)

risk_comparison.to_csv(
    results_folder / "portfolio_risk_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("Portfolio comparison completed")
print("=" * 70)

print("\nSummary comparison:")
print(summary_comparison)

print("\nRisk comparison:")
print(risk_comparison)

print("\nComparison files saved to:")
print(results_folder / "portfolio_summary_comparison.csv")
print(results_folder / "portfolio_risk_comparison.csv")

# ============================================================
# 12. Final message
# ============================================================

print("\n" + "=" * 70)
print("Full pipeline completed successfully")
print("=" * 70)

print("\nMain output folder:")
print(results_folder)

print("\nMain output files:")
print("- portfolio1_final_modeling_table.csv")
print("- portfolio1_summary_results.csv")
print("- portfolio1_risk_measures.csv")
print("- portfolio1_loss_distribution.png")
print("- portfolio1_loss_rate_distribution.png")
print("- portfolio2_final_modeling_table.csv")
print("- portfolio2_summary_results.csv")
print("- portfolio2_risk_measures.csv")
print("- portfolio2_loss_distribution.png")
print("- portfolio2_loss_rate_distribution.png")
print("- portfolio_summary_comparison.csv")
print("- portfolio_risk_comparison.csv")
