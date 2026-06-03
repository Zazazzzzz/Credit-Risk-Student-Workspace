import pandas as pd
import re
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------
# Part 1 -Data cleaning
# ----------------------

csv_files = [
    "default_history_20y.csv",
    "entities_info.csv",
    "portfolio_1.csv",
    "portfolio_2.csv",
    "portfolio_3.csv",
    "stock_returns.csv",
    "industry_index_returns.csv"
]

# Standardize entity_code
"""
Convert entity_code into the format:
UPPERCASE LETTER - UPPERCASE LETTERS/NUMBERS
'a-h8wkws'   -> 'A-H8WKWS'
"""
def standardize_entity_code(code):
    if pd.isna(code):
        return code

    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9]", "", code) # remove all characters that are not A-Z or 0-9

    if len(code) >= 2:
        return code[0] + "-" + code[1:]

    return code


# Standardize industry
def standardize_industry(industry):
    if pd.isna(industry):
        return industry

    industry = str(industry).strip().upper()

    if "A" in industry:
        return "A"

    if "B" in industry:
        return "B"

    return pd.NA


for file in csv_files:
    df = pd.read_csv(file)

    if "entity_code" in df.columns:
        df["entity_code"] = df["entity_code"].apply(standardize_entity_code)

    if "industry" in df.columns:
        df["industry"] = df["industry"].apply(standardize_industry)

    output_file = file.replace(".csv", "_cleaned.csv")
    df.to_csv(output_file, index=False)

    print(f"Cleaned and saved: {output_file}")

    if "entity_code" in df.columns:
        print(df["entity_code"].head())


    if "industry" in df.columns:
        print(df["industry"].head())

    print("-" * 50)


# ----------------------
# Table merging
# ----------------------
"""
The following code prepares and merges the datasets.

First, it reads the cleaned stock return data, entity information and industry index return data, 
The code standardizes date formats and text keys such as entity_code and industry to ensure consistent 
merging across datasets.
Duplicate observations are removed to avoid many-to-many merge problems.

Next, the code merges
1. stock_returns_cleaned.csv with entities_info_cleaned.csv using entity_code
2. the resulting dataset with industry_index_returns_cleaned.csv using both date and industry

This creates a combined dataset containing:
stock returns,industry returns,and entity characteristics
for each entity-date observation.
The merged dataset is then sorted by entity_code and date and saved as:
merged_stock_entity_industry_returns.csv

Afterwards, the code merges this return dataset with each portfolio file 
(portfolio_1, portfolio_2, and portfolio_3) using entity_code.

To ensure that only information available after the loan exposure starts is used, 
the code keeps only observations where: date≥booking_date
This step removes return observations that occurred before the portfolio exposure was booked.

Finally, the merged portfolio datasets are sorted and saved as:
portfolio_1_merged_returns_after_booking.csv
portfolio_2_merged_returns_after_booking.csv
portfolio_3_merged_returns_after_booking.csv
"""


stock_returns = pd.read_csv("stock_returns_cleaned.csv")
entities_info = pd.read_csv("entities_info_cleaned.csv")
industry_returns = pd.read_csv("industry_index_returns_cleaned.csv")

# Standardize date
stock_returns["date"] = pd.to_datetime(
    stock_returns["date"],
    dayfirst=True,
    errors="coerce"
)

industry_returns["date"] = pd.to_datetime(
    industry_returns["date"],
    dayfirst=True,
    errors="coerce"
)

# Standardize text keys
stock_returns["entity_code"] = stock_returns["entity_code"].astype(str).str.strip().str.upper()
entities_info["entity_code"] = entities_info["entity_code"].astype(str).str.strip().str.upper()

entities_info["industry"] = entities_info["industry"].astype(str).str.strip().str.upper()
industry_returns["industry"] = industry_returns["industry"].astype(str).str.strip().str.upper()

# Remove duplicates in key tables
entities_info = entities_info.drop_duplicates(
    subset=["entity_code"],
    keep="first"
)

industry_returns = industry_returns.drop_duplicates(
    subset=["date", "industry"],
    keep="first"
)

stock_returns = stock_returns.drop_duplicates(
    subset=["entity_code", "date"],
    keep="first"
)

# Merge stock_returns with entities_info by entity_code
merged_df = stock_returns.merge(
    entities_info,
    on="entity_code",
    how="left",
    validate="many_to_one"
)

# Merge industry_index_returns by date + industry
merged_df = merged_df.merge(
    industry_returns,
    on=["date", "industry"],
    how="left",
    validate="many_to_one"
)

merged_df = merged_df.rename(columns={
    "index_return": "industry_return"
})

# Check missing values after merge
print("Missing stock_return:")
print(merged_df["stock_return"].isna().sum())

print("\nMissing industry_return:")
print(merged_df["industry_return"].isna().sum())

print("\nMissing industry:")
print(merged_df["industry"].isna().sum())

# Sort
merged_df = merged_df.sort_values(
    by=["entity_code", "date"],
    ascending=[True, True]
)

merged_df = merged_df.reset_index(drop=True)

merged_df.to_csv("merged_stock_entity_industry_returns.csv", index=False)

print("Final merged dataset saved: merged_stock_entity_industry_returns.csv")

# Merge Portfolios with merged_stock_entity_industry_returns
merged_returns = pd.read_csv("merged_stock_entity_industry_returns.csv")
portfolio_files = [
    "portfolio_1_cleaned.csv",
    "portfolio_2_cleaned.csv",
    "portfolio_3_cleaned.csv"
]

merged_returns["date"] = pd.to_datetime(
    merged_returns["date"],
    dayfirst=True,
    errors="coerce"
)

# Convert date columns
for file in portfolio_files:
    portfolio = pd.read_csv(file)

    portfolio["booking_date"] = pd.to_datetime(
        portfolio["booking_date"],
        dayfirst=True,
        errors="coerce"
    )

    # Merge by entity_code
    merged_portfolio = merged_returns.merge(
        portfolio,
        on="entity_code",
        how="inner"
    )
    # Keep only rows where return date is on or after booking_date
    merged_portfolio = merged_portfolio[
        merged_portfolio["date"] >= merged_portfolio["booking_date"]
    ]

    # Sort by entity_code and date
    merged_portfolio = merged_portfolio.sort_values(
        by=["entity_code", "date"]
    ).reset_index(drop=True)

    # Save output
    output_file = file.replace("_cleaned.csv", "_merged_returns_after_booking.csv")

    merged_portfolio.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(merged_portfolio.head())
    print("-" * 50)


# ----------------------
# Part 2 - Model Setup and Parameter Estimation
# ----------------------

# Data Preperation
# Save the unique entity_name and entity_code in each portfolio for further simulations
portfolio_files = {
    "portfolio_1": "portfolio_1_merged_returns_after_booking.csv",
    "portfolio_2": "portfolio_2_merged_returns_after_booking.csv",
    "portfolio_3": "portfolio_3_merged_returns_after_booking.csv"
}

for portfolio_name, file in portfolio_files.items():
    df = pd.read_csv(file)

    entities_df = (
        df[["entity_code", "entity_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(by="entity_name")
        .reset_index(drop=True)
    )

    output_file = f"{portfolio_name}_entities.csv"

    entities_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Number of unique entities: {len(entities_df)}")
    print("-" * 50)

# --------------------------------------------------
# Prepare rho
'''
The following code estimates the factor loading parameter rho for each entity in Portfolio 1, 
Portfolio 2, and Portfolio 3.

It uses a factor regression:
    stock_return_i,t = alpha_i + beta_i*industry_return_t + residual_i,t
    
For each entity, the code regresses its stock return on the corresponding industry return. 
The regression R^2 measures how much of the entity’s return variation is explained by the common 
industry factor. Then, the entity-specific rho is estimated as:

rho_i = sqrt(R^2)

Entities with fewer than 5 valid observations are excluded from the regression 
because the available data is considered insufficient for reliable estimation.

After estimating entity-level ρ, the code combines the results from all three portfolios 
into a single dataset and calculates the average ρ separately for Industry A and Industry B. 
These industry-average values are then used to fill missing ρ values for entities 
that could not be estimated individually due to insufficient observations.

Finally, the code generates several output files, including:
portfolio_1_rho_summary.csv
portfolio_2_rho_summary.csv
portfolio_3_rho_summary.csv
all_portfolios_rho_summary.csv
portfolio_1_entities_with_rho_global_filled.csv
portfolio_2_entities_with_rho_global_filled.csv
portfolio_3_entities_with_rho_global_filled.csv

The final files contain both the original estimated ρ values and the adjusted final ρ values (rho_final) 
that are used in the Gaussian Threshold Model simulation.
'''

all_rho_results = []

for portfolio_name, file in portfolio_files.items():

    print(f"\nProcessing {portfolio_name} ...")

    df = pd.read_csv(file)

    # Check required columns
    required_cols = ["entity_code", "entity_name", "stock_return", "industry_return"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing columns in {file}: {missing_cols}")
        continue

    df["stock_return"] = pd.to_numeric(
        df["stock_return"],
        errors="coerce"
    )

    df["industry_return"] = pd.to_numeric(
        df["industry_return"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna(
        subset=["stock_return", "industry_return"]
    )
    print(f"Rows after dropping missing returns: {len(df)}")
    print(f"Number of entities: {df['entity_code'].nunique()}")

    rho_results = []
    # Estimate rho for each entity

    for entity, group in df.groupby("entity_code"):
        entity_name = group["entity_name"].iloc[0]
        industry = group["industry"].iloc[0]

        # Count valid observations for regression
        valid_group = group.dropna(subset=["stock_return", "industry_return"])
        n_obs = len(valid_group)

        # Need enough data
        if n_obs < 5:
            rho_results.append({
                "portfolio": portfolio_name,
                "entity_code": entity,
                "entity_name": entity_name,
                "industry": industry,
                "rho": np.nan,
                "n_obs": n_obs,
                "reason": "not enough valid observations"
            })
            continue

        X = valid_group["industry_return"]
        y = valid_group["stock_return"]

        # Add intercept alpha_i
        X = sm.add_constant(X)

        # OLS regression
        model = sm.OLS(y, X).fit()

        # R-squared
        r_squared = model.rsquared

        # rho = sqrt(R^2)
        rho = np.sqrt(max(r_squared, 0))

        rho_results.append({
            "portfolio": portfolio_name,
            "entity_code": entity,
            "entity_name": entity_name,
            "industry": industry,
            "rho": rho,
            "n_obs": n_obs,
            "reason": "rho estimated successfully"
        })

    rho_df = pd.DataFrame(rho_results)

    # Save entity-code, industry, rho summary table
    rho_summary_file = f"{portfolio_name}_rho_summary.csv"
    rho_df.to_csv(rho_summary_file, index=False)

    print(f"Saved rho summary: {rho_summary_file}")
    all_rho_results.append(rho_df)

# Combine all portfolio rho summary files
all_rho_df = pd.concat(
    all_rho_results,
    ignore_index=True
)

all_rho_df["rho"] = pd.to_numeric(
    all_rho_df["rho"],
    errors="coerce"
)

all_rho_df["industry"] = (
    all_rho_df["industry"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Calculate industry average rho from successfully estimated rho
industry_avg_rho = (
    all_rho_df
    .dropna(subset=["rho"])
    .groupby("industry", as_index=False)
    .agg(industry_avg_rho=("rho", "mean"))
)

print(industry_avg_rho)

all_rho_df = all_rho_df.merge(
    industry_avg_rho,
    on="industry",
    how="left"
)

# Fill missing rho
all_rho_df["rho_final"] = all_rho_df["rho"]

missing_rho_mask = all_rho_df["rho_final"].isna()

all_rho_df.loc[missing_rho_mask, "rho_final"] = all_rho_df.loc[
    missing_rho_mask,
    "industry_avg_rho"
]

print("\nGlobal industry average rho:")
print(industry_avg_rho)

# Fill missing rho in each portfolio using global industry average
all_rho_df["rho_final"] = all_rho_df["rho"]

missing_rho_mask = all_rho_df["rho_final"].isna()

all_rho_df.loc[missing_rho_mask, "rho_final"] = all_rho_df.loc[
    missing_rho_mask,
    "industry_avg_rho"
]

filled_mask = missing_rho_mask & all_rho_df["rho_final"].notna()

all_rho_df.loc[filled_mask, "reason"] = (
    all_rho_df.loc[filled_mask, "reason"].astype(str)
    + "; rho filled by global industry average"
)

still_missing_mask = all_rho_df["rho_final"].isna()

all_rho_df.loc[still_missing_mask, "reason"] = (
    all_rho_df.loc[still_missing_mask, "reason"].astype(str)
    + "; no available industry average rho"
)

for portfolio_name in portfolio_files.keys():

    rho_portfolio = all_rho_df[
        all_rho_df["portfolio"] == portfolio_name
    ].copy()

    entity_file = f"{portfolio_name}_entities.csv"
    entities_df = pd.read_csv(entity_file)

    entities_with_rho = entities_df.merge(
        rho_portfolio[
            [
                "entity_code",
                "industry",
                "rho",
                "industry_avg_rho",
                "rho_final",
                "n_obs",
                "reason"
            ]
        ],
        on="entity_code",
        how="left"
    )

    output_file = f"{portfolio_name}_entities_with_rho_global_filled.csv"

    entities_with_rho.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

# --------------------------------------------------
# Prepare EAD
"""
This code fills missing EAD values for entities across Portfolio 1, Portfolio 2, and Portfolio 3.

It first extracts one row per entity from each portfolio, keeping:
entity_code, entity_name, size_bucket, EAD and portfolio

Then, it combines all entities from the three portfolios into one dataset 
and calculates the average EAD for each size_bucket.

For entities with missing EAD, the code fills the missing value using the average 
EAD of the corresponding size_bucket.

It also creates new columns:
EAD_original, EAD_filled and EAD_fill_note
to record whether the original EAD was available or whether it was filled 
by the size-bucket average.

Finally, the code saves:
all_portfolios_entities_EAD_filled.csv
portfolio_1_entities_EAD_filled.csv
portfolio_2_entities_EAD_filled.csv
portfolio_3_entities_EAD_filled.csv

"""

portfolio_files = {
    "portfolio_1": "portfolio_1_merged_returns_after_booking.csv",
    "portfolio_2": "portfolio_2_merged_returns_after_booking.csv",
    "portfolio_3": "portfolio_3_merged_returns_after_booking.csv"
}

all_entities = []

for portfolio_name, file in portfolio_files.items():

    df = pd.read_csv(file)

    # Keep one row per entity
    entity_df = (
        df[["entity_code", "entity_name", "size_bucket", "EAD"]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    entity_df["portfolio"] = portfolio_name

    entity_df["EAD"] = pd.to_numeric(
        entity_df["EAD"],
        errors="coerce"
    )

    all_entities.append(entity_df)

# Combine all portfolios
all_entities_df = pd.concat(
    all_entities,
    ignore_index=True
)

size_bucket_avg_ead = (
    all_entities_df
    .dropna(subset=["EAD"])
    .groupby("size_bucket", as_index=False)
    .agg(size_bucket_avg_EAD=("EAD", "mean"))
)

print("Average EAD by size_bucket:")
print(size_bucket_avg_ead)

size_bucket_avg_ead.to_csv(
    "size_bucket_average_EAD.csv",
    index=False
)

all_entities_df = all_entities_df.merge(
    size_bucket_avg_ead,
    on="size_bucket",
    how="left"
)

all_entities_df["EAD_original"] = all_entities_df["EAD"]

missing_ead_mask = all_entities_df["EAD"].isna()

all_entities_df["EAD_filled"] = all_entities_df["EAD"]

all_entities_df.loc[missing_ead_mask, "EAD_filled"] = all_entities_df.loc[
    missing_ead_mask,
    "size_bucket_avg_EAD"
]

# Add note column
all_entities_df["EAD_fill_note"] = "original EAD available"

all_entities_df.loc[
    missing_ead_mask & all_entities_df["EAD_filled"].notna(),
    "EAD_fill_note"
] = "EAD missing; filled by size_bucket average"

all_entities_df.loc[
    missing_ead_mask & all_entities_df["EAD_filled"].isna(),
    "EAD_fill_note"
] = "EAD missing; no size_bucket average available"

all_entities_df.to_csv(
    "all_portfolios_entities_EAD_filled.csv",
    index=False
)

print("Saved: all_portfolios_entities_EAD_filled.csv")

for portfolio_name in portfolio_files.keys():

    portfolio_filled = all_entities_df[
        all_entities_df["portfolio"] == portfolio_name
    ].copy()

    output_file = f"{portfolio_name}_entities_EAD_filled.csv"

    portfolio_filled.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")

# --------------------------------------------------
# Prepare base pd hint
"""
This code handles missing base_pd_hint values in the three portfolio datasets.

It converts base_pd_hint into numeric format, identifies missing values, 
and fills them with the minimum valid probability value: 1e-10

This ensures that base_pd_hint can later be safely used to calculate the default threshold:
d_i = Phi^(-1)(p_i)

The code also creates a new column, base_pd_hint_note, 
to indicate whether the PD value was originally available or filled manually.

Finally, it saves new files with the suffix:
_base_pd_filled.csv

"""

files = [
    "portfolio_1_merged_returns_after_booking.csv",
    "portfolio_2_merged_returns_after_booking.csv",
    "portfolio_3_merged_returns_after_booking.csv"
]

min_pd = 1e-10

for file in files:

    df = pd.read_csv(file)

    # Convert base_pd_hint to numeric
    df["base_pd_hint"] = pd.to_numeric(
        df["base_pd_hint"],
        errors="coerce"
    )

    # Create note column
    df["base_pd_hint_note"] = "original base_pd_hint available"

    # Find missing base_pd_hint
    missing_pd_mask = df["base_pd_hint"].isna()

    # Fill missing base_pd_hint with the minimum valid value
    df.loc[missing_pd_mask, "base_pd_hint"] = min_pd

    df.loc[
        missing_pd_mask,
        "base_pd_hint_note"
    ] = "base_pd_hint missing; filled with minimum valid value 1e-10"

    # Apply clipping rule
    df["base_pd_hint"] = df["base_pd_hint"].clip(
        1e-10,
        1 - 1e-10
    )

    # Save new file
    output_file = file.replace(
        ".csv",
        "_base_pd_filled.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Number of filled base_pd_hint values: {missing_pd_mask.sum()}")
    print("-" * 50)

# --------------------------------------------------
# prepare lgd
"""
This code handles missing LGD values across Portfolio 1, Portfolio 2, and Portfolio 3.

It first extracts one row per entity, keeping:
entity_code, entity_name, industry, lgd and portfolio

Then, it combines all entities from the three portfolios and calculates:
industry average LGD and global average LGD

For entities with missing lgd, the code first fills the missing value 
using the corresponding industry average LGD. 
If the industry average is also unavailable, it uses the global average LGD.

The code also creates new columns:
lgd_original, lgd_final and lgd_fill_note
to record whether the LGD was original or filled.

Finally, it saves one combined file and three separate portfolio-level files:
all_portfolios_entities_lgd_filled.csv
portfolio_1_entities_lgd_filled.csv
portfolio_2_entities_lgd_filled.csv
portfolio_3_entities_lgd_filled.csv

"""

portfolio_files = {
    "portfolio_1": "portfolio_1_merged_returns_after_booking.csv",
    "portfolio_2": "portfolio_2_merged_returns_after_booking.csv",
    "portfolio_3": "portfolio_3_merged_returns_after_booking.csv"
}

all_lgd_data = []

for portfolio_name, file in portfolio_files.items():

    df = pd.read_csv(file)

    entity_lgd = (
        df[["entity_code", "entity_name", "industry", "lgd"]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    entity_lgd["portfolio"] = portfolio_name

    entity_lgd["lgd"] = pd.to_numeric(
        entity_lgd["lgd"],
        errors="coerce"
    )

    all_lgd_data.append(entity_lgd)

all_lgd_df = pd.concat(
    all_lgd_data,
    ignore_index=True
)

industry_avg_lgd = (
    all_lgd_df
    .dropna(subset=["lgd"])
    .groupby("industry", as_index=False)
    .agg(industry_avg_lgd=("lgd", "mean"))
)

global_avg_lgd = all_lgd_df["lgd"].mean(skipna=True)

print("Industry average LGD:")
print(industry_avg_lgd)

print(f"\nGlobal average LGD: {global_avg_lgd}")

all_lgd_df = all_lgd_df.merge(
    industry_avg_lgd,
    on="industry",
    how="left"
)

# Fill missing LGD
all_lgd_df["lgd_original"] = all_lgd_df["lgd"]

all_lgd_df["lgd_final"] = all_lgd_df["lgd"]

all_lgd_df["lgd_fill_note"] = "original lgd available"

missing_lgd_mask = all_lgd_df["lgd_final"].isna()

# Fill by industry average first
all_lgd_df.loc[
    missing_lgd_mask,
    "lgd_final"
] = all_lgd_df.loc[
    missing_lgd_mask,
    "industry_avg_lgd"
]

filled_by_industry_mask = (
    missing_lgd_mask
    & all_lgd_df["lgd_final"].notna()
)

all_lgd_df.loc[
    filled_by_industry_mask,
    "lgd_fill_note"
] = "lgd missing; filled by industry average"

# If still missing, fill by global average
still_missing_mask = all_lgd_df["lgd_final"].isna()

all_lgd_df.loc[
    still_missing_mask,
    "lgd_final"
] = global_avg_lgd

all_lgd_df.loc[
    still_missing_mask & all_lgd_df["lgd_final"].notna(),
    "lgd_fill_note"
] = "lgd missing; filled by global average"

# Save combined LGD filled table
all_lgd_df.to_csv(
    "all_portfolios_entities_lgd_filled.csv",
    index=False
)

print("Saved: all_portfolios_entities_lgd_filled.csv")

for portfolio_name in portfolio_files.keys():

    portfolio_lgd_filled = all_lgd_df[
        all_lgd_df["portfolio"] == portfolio_name
    ].copy()

    output_file = f"{portfolio_name}_entities_lgd_filled.csv"

    portfolio_lgd_filled.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")


# ----------------------
# Part 3 - Simulation and Risk Measurement
# ----------------------
"""
This code prepares the final parameter table for the Gaussian Threshold Model simulation.

For each portfolio, it reads the cleaned input files containing:
EAD, base_pd_hint, rho_final and lgd_final

Then, it creates one row per entity and merges the required simulation parameters by entity_code.
The final params table contains:
entity_code, entity_name, industry, EAD, base_pd_hint, rho_final and lgd,
where:
EAD comes from the EAD-filled file,
base_pd_hint comes from the PD-filled file,
rho_final comes from the rho-filled file,
lgd comes from the LGD-filled file.

Finally, the code converts these key variables into numeric format 
so they can be used in the one-year Gaussian Threshold Model loss simulation.
"""
np.random.seed(42)

portfolio_files = {
    "portfolio_1": {
        "base_file": "portfolio_1_merged_returns_after_booking.csv",
        "ead_file": "portfolio_1_entities_EAD_filled.csv",
        "pd_file": "portfolio_1_merged_returns_after_booking_base_pd_filled.csv",
        "rho_file": "portfolio_1_entities_with_rho_global_filled.csv",
        "lgd_file": "portfolio_1_entities_lgd_filled.csv"
    },
    "portfolio_2": {
        "base_file": "portfolio_2_merged_returns_after_booking.csv",
        "ead_file": "portfolio_2_entities_EAD_filled.csv",
        "pd_file": "portfolio_2_merged_returns_after_booking_base_pd_filled.csv",
        "rho_file": "portfolio_2_entities_with_rho_global_filled.csv",
        "lgd_file": "portfolio_2_entities_lgd_filled.csv"
    },
    "portfolio_3": {
        "base_file": "portfolio_3_merged_returns_after_booking.csv",
        "ead_file": "portfolio_3_entities_EAD_filled.csv",
        "pd_file": "portfolio_3_merged_returns_after_booking_base_pd_filled.csv",
        "rho_file": "portfolio_3_entities_with_rho_global_filled.csv",
        "lgd_file": "portfolio_3_entities_lgd_filled.csv"
    }
}

n_sim = 10000
alpha = 0.99

simulation_portfolios = ["portfolio_1", "portfolio_2"]
all_comparison_results = []

for portfolio_name, files in portfolio_files.items():
    if portfolio_name not in simulation_portfolios:
        print(f"Skipping simulation for {portfolio_name}")
        continue

    print(f"\nRunning simulation for {portfolio_name}...")

    base_df = pd.read_csv(files["base_file"])
    ead_df = pd.read_csv(files["ead_file"])
    pd_df = pd.read_csv(files["pd_file"])
    rho_df = pd.read_csv(files["rho_file"])
    lgd_df = pd.read_csv(files["lgd_file"])

    params = (
        base_df[[
            "entity_code",
            "entity_name",
            "industry"
        ]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    ead_cols = [
        "entity_code",
        "EAD_filled",
        "EAD_fill_note"
    ]

    ead_source = ead_df[ead_cols].drop_duplicates(
        subset=["entity_code"]
    )

    params = params.merge(
        ead_source,
        on="entity_code",
        how="left"
    )

    params = params.rename(columns={
        "EAD_filled": "EAD"
    })

    pd_source = (
        pd_df[[
            "entity_code",
            "base_pd_hint",
            "base_pd_hint_note"
        ]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    params = params.merge(
        pd_source,
        on="entity_code",
        how="left"
    )

    rho_source = (
        rho_df[[
            "entity_code",
            "rho_final",
            "n_obs",
            "reason"
        ]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    params = params.merge(
        rho_source,
        on="entity_code",
        how="left"
    )

    lgd_source = (
        lgd_df[[
            "entity_code",
            "lgd_final",
            "lgd_fill_note"
        ]]
        .drop_duplicates(subset=["entity_code"])
        .copy()
    )

    params = params.merge(
        lgd_source,
        on="entity_code",
        how="left"
    )

    params = params.rename(columns={
        "lgd_final": "lgd"
    })

    numeric_cols = ["EAD", "lgd", "base_pd_hint", "rho_final"]

    for col in numeric_cols:
        params[col] = pd.to_numeric(
            params[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # Check which entities will be dropped
    """
    This code checks whether all entities have the required parameters for the 
    Gaussian Threshold Model simulation:EAD, lgd, base_pd_hint, rho_final.
    
    It identifies entities with missing values in any of these columns and records 
    why they would be dropped, for example:missing EAD, missing rho_final, missing base_pd_hint
    
    The dropped entities are printed and saved as:
    portfolio_name_dropped_entities.csv
    
    The full simulation parameter table is also saved as:
    portfolio_name_simulation_parameters.csv
    
    Finally, the code removes entities with missing required parameters, clips base_pd_hint and 
    rho_final into valid ranges, and reports the final number of entities used in the simulation.
    
    """

    required_param_cols = ["EAD", "lgd", "base_pd_hint", "rho_final"]

    missing_mask = params[required_param_cols].isna().any(axis=1)

    dropped_entities = params.loc[
        missing_mask,
        [
            "entity_code",
            "entity_name",
            "industry",
            "EAD",
            "lgd",
            "base_pd_hint",
            "rho_final"
        ]
    ].copy()


    def get_drop_reason(row):
        missing_cols = []

        for col in required_param_cols:
            if pd.isna(row[col]):
                missing_cols.append(col)

        return "missing " + ", ".join(missing_cols)

    if len(dropped_entities) > 0:
        dropped_entities["drop_reason"] = dropped_entities.apply(
            get_drop_reason,
            axis=1
        )

        print("\nEntities dropped due to missing parameters:")
        print(dropped_entities[[
            "entity_code",
            "entity_name",
            "industry",
            "drop_reason"
        ]])

        dropped_entities.to_csv(
            f"{portfolio_name}_dropped_entities.csv",
            index=False
        )

        print(f"Saved dropped entity list: {portfolio_name}_dropped_entities.csv")

    else:
        print("\nNo entities dropped.")

    if len(dropped_entities) > 0:
        print(dropped_entities)

    params.to_csv(
        f"{portfolio_name}_simulation_parameters.csv",
        index=False
    )

    print(f"Saved simulation parameters: {portfolio_name}_simulation_parameters.csv")

    params = params.dropna(
        subset=["EAD", "lgd", "base_pd_hint", "rho_final"]
    ).reset_index(drop=True)

    params["base_pd_hint"] = params["base_pd_hint"].clip(1e-10, 1 - 1e-10)
    params["rho_final"] = params["rho_final"].clip(0, 0.999999)

    m = len(params)

    print(f"Number of entities used: {m}")

    # --------------------------------------------------
    # Simulation
    """
    This code runs the one-year Gaussian Threshold Model simulation. 
    It extracts the model inputs: EAD, LGD, base_pd_hint, rho_final.
    
    Then it calculates the default threshold:
    threshold = Phi^(-1)(p)
    
    For each simulation, the code draws:
    1. one common systematic factor F
    2. one idiosyncratic shock epsilon_i for each entity
    
    It then computes the latent credit variable:
    X_i = rho_i * F + sqrt(1 - rho_i^2) * epsilon_i
    
    An entity defaults if: 
    X_i <= threshold_i 
    
    The total simulated portfolio loss is calculated as:
    Loss = sum(EAD_i * LGD_i * default_i)
    
    Finally, the code computes and saves the risk measures: 
    Expected Loss, Standard Deviation, VaR 99%, ES 99%
    and outputs: 
    portfolio_name_simulation_parameters.csv
    portfolio_name_simulated_loss_distribution.csv
    portfolio_name_risk_measures.csv
    
    """


    EAD = params["EAD"].values
    LGD = params["lgd"].values
    p = params["base_pd_hint"].values
    rho = params["rho_final"].values
    threshold = norm.ppf(p)

    F = np.random.normal(0, 1, size=n_sim)
    epsilon = np.random.normal(0, 1, size=(n_sim, m))

    X = rho * F[:, None] + np.sqrt(1 - rho**2) * epsilon
    defaults = (X <= threshold).astype(int)
    loss_sim = np.sum(EAD * LGD * defaults, axis=1)

    EL = loss_sim.mean()
    loss_std = loss_sim.std()
    VaR_99 = np.quantile(loss_sim, alpha)
    ES_99 = loss_sim[loss_sim >= VaR_99].mean()

    result = pd.DataFrame({
        "portfolio": [portfolio_name],
        "n_entities": [m],
        "n_simulations": [n_sim],
        "Expected_Loss": [EL],
        "Standard_Deviation": [loss_std],
        "VaR_99": [VaR_99],
        "ES_99": [ES_99]
    })

    params["threshold"] = threshold

    params.to_csv(
        f"{portfolio_name}_simulation_parameters.csv",
        index=False
    )

    loss_df = pd.DataFrame({
        "simulation_id": np.arange(1, n_sim + 1),
        "portfolio_loss": loss_sim
    })

    loss_df.to_csv(
        f"{portfolio_name}_simulated_loss_distribution.csv",
        index=False
    )

    result.to_csv(
        f"{portfolio_name}_risk_measures.csv",
        index=False
    )

    print(result)

    # --------------------------------------------------
    # Analytical conditional expected loss function
    """
    This section uses the analytical conditional expected loss function.
    Here draws only the systematic factor F and calculates the conditional expected
    portfolio loss:

        l(F) = sum_i EAD_i * LGD_i * p_i(F)

    where:

        p_i(F) = Phi((d_i - rho_i * F) / sqrt(1 - rho_i^2))

    This gives a smooth systematic-factor-driven loss distribution.
    """


    def conditional_pd(f):
        return norm.cdf(
            (threshold - rho * f) / np.sqrt(1 - rho ** 2)
        )


    def analytical_conditional_loss(f):
        return np.sum(
            EAD * LGD * conditional_pd(f)
        )


    # Simulate systematic factor only
    F_ana = np.random.normal(
        0,
        1,
        size=n_sim
    )

    analytical_loss_sim = np.array([
        analytical_conditional_loss(f)
        for f in F_ana
    ])

    # Analytical conditional risk measures
    EL_ana_sim = analytical_loss_sim.mean()
    loss_std_ana_sim = analytical_loss_sim.std()
    VaR_99_ana_sim = np.quantile(
        analytical_loss_sim,
        alpha
    )
    ES_99_ana_sim = analytical_loss_sim[
        analytical_loss_sim >= VaR_99_ana_sim
        ].mean()

    analytical_result = pd.DataFrame({
        "portfolio": [portfolio_name],
        "n_entities": [m],
        "n_simulations": [n_sim],
        "Method": ["Analytical Conditional Expected Loss"],
        "Expected_Loss": [EL_ana_sim],
        "Standard_Deviation": [loss_std_ana_sim],
        "VaR_99": [VaR_99_ana_sim],
        "ES_99": [ES_99_ana_sim]
    })

    print("\nAnalytical conditional expected loss results:")
    print(analytical_result)

    # Save analytical conditional loss distribution
    analytical_loss_df = pd.DataFrame({
        "simulation_id": np.arange(1, n_sim + 1),
        "systematic_factor_F": F_ana,
        "analytical_conditional_loss": analytical_loss_sim
    })

    analytical_loss_df.to_csv(
        f"{portfolio_name}_analytical_conditional_loss_distribution.csv",
        index=False
    )

    analytical_result.to_csv(
        f"{portfolio_name}_analytical_conditional_risk_measures.csv",
        index=False
    )

    # Compare original Monte Carlo simulation with analytical conditional simulation
    comparison_result = pd.DataFrame({
        "portfolio": [portfolio_name, portfolio_name],
        "Method": [
            "Default Indicator Simulation",
            "Analytical Conditional Expected Loss"
        ],
        "Expected_Loss": [
            EL,
            EL_ana_sim
        ],
        "Standard_Deviation": [
            loss_std,
            loss_std_ana_sim
        ],
        "VaR_99": [
            VaR_99,
            VaR_99_ana_sim
        ],
        "ES_99": [
            ES_99,
            ES_99_ana_sim
        ]
    })

    comparison_result.to_csv(
        f"{portfolio_name}_simulation_vs_analytical_comparison.csv",
        index=False
    )

    print("\nComparison between simulation and analytical conditional method:")
    print(comparison_result)

    # --------------------------------------------------
    # Compare simulation and analytical results
    # at both 99% and 99.9% confidence levels

    alpha_99 = 0.99
    alpha_999 = 0.999

    # Default-indicator simulation risk measures
    VaR_99_sim = np.quantile(loss_sim, alpha_99)
    VaR_999_sim = np.quantile(loss_sim, alpha_999)

    ES_99_sim = loss_sim[loss_sim >= VaR_99_sim].mean()
    ES_999_sim = loss_sim[loss_sim >= VaR_999_sim].mean()

    # Analytical conditional expected loss risk measures
    VaR_99_ana = np.quantile(analytical_loss_sim, alpha_99)
    VaR_999_ana = np.quantile(analytical_loss_sim, alpha_999)

    ES_99_ana = analytical_loss_sim[
        analytical_loss_sim >= VaR_99_ana
        ].mean()

    ES_999_ana = analytical_loss_sim[
        analytical_loss_sim >= VaR_999_ana
        ].mean()

    # Average rho used in this portfolio
    avg_rho = params["rho_final"].mean()

    comparison_result = pd.DataFrame({
        "portfolio": [
            portfolio_name,
            portfolio_name
        ],
        "Method": [
            "Default Indicator Simulation",
            "Analytical Conditional Expected Loss"
        ],
        "Expected Loss": [
            EL,
            EL_ana_sim
        ],
        "VaR 99.9%": [
            VaR_999_sim,
            VaR_999_ana
        ],
        "VaR 99%": [
            VaR_99_sim,
            VaR_99_ana
        ],
        "ES 99.9%": [
            ES_999_sim,
            ES_999_ana
        ],
        "ES 99%": [
            ES_99_sim,
            ES_99_ana
        ],
        "rho": [
            avg_rho,
            avg_rho
        ]
    })

    comparison_result.to_csv(
        f"{portfolio_name}_simulation_vs_analytical_comparison_full.csv",
        index=False
    )

    print("\nComparison of simulation and analytical results:")
    print(comparison_result)
    all_comparison_results.append(comparison_result)

    # --------------------------------------------------
    # Plot

    """
    This code visualizes and compares the simulated one-year portfolio loss distributions generated 
    by the Gaussian Threshold Model.
    
    First, for each portfolio, the code plots the simulated loss distribution as a histogram 
    and adds vertical reference lines for:
    Expected Loss (EL)
    Value-at-Risk at 99% (VaR 99%)
    Expected Shortfall at 99% (ES 99%)
    
    The figures are saved individually as: portfolio_name_loss_distribution.png
    
    """
    loss_sim_million = loss_sim / 1e6
    EL_million = EL / 1e6
    VaR_99_million = VaR_99 / 1e6
    ES_99_million = ES_99 / 1e6

    plt.figure(figsize=(10, 6))

    plt.hist(
        loss_sim_million,
        bins=50,
        density=True,
        color="lightblue",
        edgecolor="grey",
        alpha=0.7
    )

    plt.axvline(EL_million, color="red", linestyle="--", label=f"EL = {EL:.2f}")
    plt.axvline(VaR_99_million, color="yellow", linestyle="--", label=f"VaR 99% = {VaR_99:.2f}")
    plt.axvline(ES_99_million, color="lightblue", linestyle="--", label=f"ES 99% = {ES_99:.2f}")

    plt.xlabel("Portfolio Loss (million)")
    plt.ylabel("Density")
    plt.title(f"Simulated One-Year Loss Distribution: {portfolio_name}")
    plt.legend()
    plt.tight_layout()

    plot_file = f"{portfolio_name}_loss_distribution_million.png"
    plt.savefig(plot_file, dpi=300)
    plt.show()

    print(f"Saved parameter file: {portfolio_name}_simulation_parameters.csv")
    print(f"Saved loss distribution: {portfolio_name}_simulated_loss_distribution.csv")
    print(f"Saved risk measures: {portfolio_name}_risk_measures.csv")
    print(f"Saved plot: {plot_file}")
    print("-" * 60)

    # Combine bins for comparison
    """
    This code compares Portfolio 1 and Portfolio 2 directly by plotting both 
    simulated loss distributions on the same figure. 
    The losses are converted into million units to ensure consistent and readable axis scaling.

    The comparison figure is saved as:
    portfolio_1_2_loss_distribution_comparison_million.png
    """

    loss_p1 = pd.read_csv("portfolio_1_simulated_loss_distribution.csv")
    loss_p2 = pd.read_csv("portfolio_2_simulated_loss_distribution.csv")

    # Convert losses to millions
    loss_1 = loss_p1["portfolio_loss"].values / 1e6
    loss_2 = loss_p2["portfolio_loss"].values / 1e6

    x_min = 0
    x_max = max(loss_1.max(), loss_2.max())
    bins = np.linspace(x_min, x_max, 50)

    plt.figure(figsize=(10, 6))

    plt.hist(
        loss_1,
        bins=bins,
        color="pink",
        density=True,
        alpha=0.5,
        edgecolor="grey",
        label="Portfolio 1"
    )

    plt.hist(
        loss_2,
        bins=bins,
        density=True,
        alpha=0.5,
        edgecolor="grey",
        label="Portfolio 2"
    )

    plt.xlabel("Portfolio Loss (million)")
    plt.ylabel("Density")
    plt.title("Comparison of Simulated One-Year Loss Distributions")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "portfolio_1_2_loss_distribution_comparison_million.png",
        dpi=300
    )

    plt.show()

    # Plot analytical conditional loss distribution

    analytical_loss_million = analytical_loss_sim / 1e6
    EL_ana_million = EL_ana_sim / 1e6
    VaR_99_ana_million = VaR_99_ana_sim / 1e6
    ES_99_ana_million = ES_99_ana_sim / 1e6

    plt.figure(figsize=(10, 6))

    plt.hist(
        analytical_loss_million,
        bins=50,
        density=True,
        color="lightgreen",
        edgecolor="grey",
        alpha=0.7
    )

    plt.axvline(
        EL_ana_million,
        color="red",
        linestyle="--",
        label=f"EL = {EL_ana_million:.2f}M"
    )

    plt.axvline(
        VaR_99_ana_million,
        color="orange",
        linestyle="--",
        label=f"VaR 99% = {VaR_99_ana_million:.2f}M"
    )

    plt.axvline(
        ES_99_ana_million,
        color="blue",
        linestyle="--",
        label=f"ES 99% = {ES_99_ana_million:.2f}M"
    )

    plt.xlabel("Total Portfolio Loss (million)")
    plt.ylabel("Density")
    plt.title(
        f"Analytical Conditional Expected Loss Distribution: {portfolio_name}"
    )

    plt.legend()
    plt.tight_layout()

    analytical_plot_file = (
        f"{portfolio_name}_analytical_conditional_loss_distribution_million.png"
    )

    plt.savefig(analytical_plot_file, dpi=300)
    plt.show()

    print(f"Saved analytical conditional plot: {analytical_plot_file}")

all_comparison_df = pd.concat(
    all_comparison_results,
    ignore_index=True
)

all_comparison_df.to_csv(
    "portfolio_1_2_simulation_vs_analytical_comparison_full.csv",
    index=False
)

print("\nSaved combined comparison file:")
print(all_comparison_df)

# --------------------------------------------------
# Compare simulation loss distribution and analytical
# conditional loss distribution for Portfolio 1 and 2

comparison_portfolios = ["portfolio_1", "portfolio_2"]

for portfolio_name in comparison_portfolios:

    # Read default-indicator simulation losses
    sim_loss_df = pd.read_csv(
        f"{portfolio_name}_simulated_loss_distribution.csv"
    )

    # Read analytical conditional expected losses
    ana_loss_df = pd.read_csv(
        f"{portfolio_name}_analytical_conditional_loss_distribution.csv"
    )

    # Convert losses to million units
    sim_loss = sim_loss_df["portfolio_loss"].values / 1e6
    ana_loss = ana_loss_df["analytical_conditional_loss"].values / 1e6

    # Use common bins
    x_min = 0
    x_max = max(sim_loss.max(), ana_loss.max())

    bins = np.linspace(
        x_min,
        x_max,
        50
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        sim_loss,
        bins=bins,
        density=True,
        alpha=0.5,
        color="lightblue",
        edgecolor="grey",
        label="Default Indicator Simulation"
    )

    plt.hist(
        ana_loss,
        bins=bins,
        density=True,
        alpha=0.5,
        color="lightgreen",
        edgecolor="grey",
        label="Analytical Conditional Expected Loss"
    )

    plt.xlabel("Total Portfolio Loss (million)")
    plt.ylabel("Density")
    plt.title(
        f"Simulation vs Analytical Conditional Loss Distribution: {portfolio_name}"
    )

    plt.legend()
    plt.tight_layout()

    output_file = (
        f"{portfolio_name}_simulation_vs_analytical_loss_distribution_million.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.show()

    print(f"Saved comparison plot: {output_file}")