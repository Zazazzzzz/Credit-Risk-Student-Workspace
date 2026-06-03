# ================================================================
# File:        01_data_cleaning.py
# Author:      Shazia Ishaq
# Course:      Introduction to Credit Risk — University of Freiburg
# Description: Part 1 — Data Loading and Cleaning
#
# INPUTS:  all raw CSV files from Final Assignment/data/
# OUTPUTS: cleaned_portfolio1.csv  → report Table 1
#          cleaned_portfolio2.csv  → report Table 1
#          step1_cleaned.pkl       → passed to 02_parameter_estimation.py
# ================================================================

import os
import pickle
import pandas as pd
import numpy as np

# ================================================================
# DIRECTORIES — change only DIR_DATA if data moves
# ================================================================
DIR_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'Final Assignment', 'data'
)
DIR_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results'
)
os.makedirs(DIR_RESULTS, exist_ok=True)

print("=" * 65)
print("   PART 1: DATA LOADING AND CLEANING")
print("=" * 65)
print(f"\nData:    {os.path.normpath(DIR_DATA)}")
print(f"Results: {os.path.normpath(DIR_RESULTS)}")

# ================================================================
# STEP 1: LOAD ALL RAW DATA FILES
# ================================================================
# We load all six files exactly as provided — no modifications yet.
# This allows us to first understand what problems exist.

print("\n" + "-" * 65)
print("STEP 1: Loading raw data files")
print("-" * 65)

entities = pd.read_csv(os.path.join(DIR_DATA, 'entities_info.csv'))
port1 = pd.read_csv(os.path.join(DIR_DATA, 'portfolio_1.csv'))
port2 = pd.read_csv(os.path.join(DIR_DATA, 'portfolio_2.csv'))
returns = pd.read_csv(os.path.join(DIR_DATA, 'stock_returns.csv'))
idx_ret = pd.read_csv(os.path.join(DIR_DATA, 'industry_index_returns.csv'))
defaults = pd.read_csv(os.path.join(DIR_DATA, 'default_history_20y.csv'))

print(f"\n  {'File':<30} {'Rows':>6}  {'Cols':>5}")
print(f"  {'-' * 44}")
for name, df in [
    ('entities_info.csv', entities),
    ('portfolio_1.csv', port1),
    ('portfolio_2.csv', port2),
    ('stock_returns.csv', returns),
    ('industry_index_returns.csv', idx_ret),
    ('default_history_20y.csv', defaults),
]:
    print(f"  {name:<30} {df.shape[0]:>6}  {df.shape[1]:>5}")

# ================================================================
# STEP 2: IDENTIFY AND DOCUMENT PROBLEMS
# ================================================================
# The data is intentionally messy. We document every problem
# found so it can be described accurately in the report.

print("\n" + "-" * 65)
print("STEP 2: Data quality issues identified")
print("-" * 65)
print(f"  Entity codes:   mixed case + extra spaces")
print(f"  Industry labels:{entities['industry'].unique()[:4].tolist()}")
print(f"  Region labels:  {entities['region'].dropna().unique()[:4].tolist()}")
print(f"  Date formats:   mixed (YYYY-MM-DD and DD-MM-YYYY)")
print(f"  Missing LGD:    {entities['lgd'].isna().sum()}")
print(f"  Missing PD:     {entities['base_pd_hint'].isna().sum()}")
print(f"  Missing returns:{returns['stock_return'].isna().sum()}")
print(f"  Missing default:{defaults['default_event'].isna().sum()}")
print(f"  P1 duplicates:  {port1.duplicated(subset=['entity_code']).sum()}")
print(f"  P2 duplicates:  {port2.duplicated(subset=['entity_code']).sum()}")

# ================================================================
# STEP 3: CLEAN ENTITY CODES
# ================================================================
# Problem: same company has codes like 'a-h8wkws', ' A-H8WKWS '
# Fix: strip whitespace, convert to uppercase
# Reason: codes are the join key — must match exactly across files

print("\n" + "-" * 65)
print("STEP 3: Cleaning entity codes")
print("-" * 65)


def clean_entity_code(code):
    """Remove whitespace and convert to uppercase."""
    if pd.isna(code):
        return np.nan
    return str(code).strip().upper()


for df in [entities, port1, port2, returns, defaults]:
    df['entity_code'] = df['entity_code'].apply(clean_entity_code)

print("  Done: spaces removed, all codes uppercased")

# ================================================================
# STEP 4: STANDARDIZE INDUSTRY LABELS
# ================================================================
# Problem: 'Industry A', 'industry a', 'Ind. A', 'IndustryA'
#          all refer to the same industry
# Fix: map all variants to 'Industry A' or 'Industry B'
# Reason: needed to match firms to their industry index for rho

print("\n" + "-" * 65)
print("STEP 4: Standardizing industry labels")
print("-" * 65)


def clean_industry_label(label):
    """Standardize industry label to 'Industry A' or 'Industry B'."""
    if pd.isna(label):
        return np.nan
    cleaned = str(label).strip().lower().replace(' ', '').replace('.', '')
    if 'a' in cleaned and 'b' not in cleaned:
        return 'Industry A'
    if 'b' in cleaned:
        return 'Industry B'
    return np.nan


entities['industry'] = entities['industry'].apply(clean_industry_label)
idx_ret['industry'] = idx_ret['industry'].apply(clean_industry_label)
print(f"  Unique industries after cleaning: "
      f"{entities['industry'].dropna().unique().tolist()}")

# ================================================================
# STEP 5: STANDARDIZE REGION LABELS
# ================================================================
# Problem: 'europe', 'Europe', 'SouthAmerica', 'NorthAmerica'
# Fix: map to consistent capitalized names
# Reason: needed for descriptive statistics in the report

print("\n" + "-" * 65)
print("STEP 5: Standardizing region labels")
print("-" * 65)


def clean_region_label(region):
    """Map region variants to standardized names."""
    if pd.isna(region):
        return 'Unknown'
    r = str(region).strip().lower().replace(' ', '')
    mapping = {
        'europe': 'Europe',
        'asia': 'Asia',
        'northamerica': 'North America',
        'southamerica': 'South America',
    }
    return mapping.get(r, 'Unknown')


entities['region'] = entities['region'].apply(clean_region_label)
print(f"  Unique regions: {entities['region'].unique().tolist()}")

# ================================================================
# STEP 6: FIX DATE FORMATS
# ================================================================
# Problem: '2023-01-02' and '04-01-2023' mixed in same column
# Fix: parse with format='mixed' → consistent datetime format
# Reason: needed for merging return data by date

print("\n" + "-" * 65)
print("STEP 6: Fixing date formats")
print("-" * 65)

returns['date'] = pd.to_datetime(returns['date'],
                                 format='mixed', dayfirst=False)
idx_ret['date'] = pd.to_datetime(idx_ret['date'],
                                 format='mixed', dayfirst=False)
defaults['date'] = pd.to_datetime(defaults['date'],
                                  format='mixed', dayfirst=False)
print(f"  Returns date range: {returns['date'].min().date()} "
      f"to {returns['date'].max().date()}")

# ================================================================
# STEP 7: HANDLE MISSING VALUES
# ================================================================
# LGD missing (6 values):
#   Fill with median — robust estimate, industry-standard range
# PD missing (5 values):
#   Fill with median — consistent with peer group estimate
# Stock returns missing (2242 values):
#   Drop rows — cannot interpolate financial returns
# Default events missing (47 values):
#   Fill with 0 — missing record implies no default observed

print("\n" + "-" * 65)
print("STEP 7: Handling missing values")
print("-" * 65)

lgd_median = entities['lgd'].median()
pd_median = entities['base_pd_hint'].median()

entities['lgd'] = entities['lgd'].fillna(lgd_median)
entities['base_pd_hint'] = entities['base_pd_hint'].fillna(pd_median)
entities['region'] = entities['region'].fillna('Unknown')

n_before = len(returns)
returns = returns.dropna(subset=['stock_return'])
defaults['default_event'] = defaults['default_event'].fillna(0)

print(f"  LGD: filled with median = {lgd_median:.4f}")
print(f"  PD:  filled with median = {pd_median:.4f}")
print(f"  Stock returns: dropped {n_before - len(returns)} missing rows")
print(f"  Default events: NaN treated as 0")

# ================================================================
# STEP 8: REMOVE DUPLICATES
# ================================================================
# Keep first occurrence of each entity code.
# Duplicates would inflate portfolio sizes and distort risk measures.

print("\n" + "-" * 65)
print("STEP 8: Removing duplicates")
print("-" * 65)

for name, df, cols in [
    ('Portfolio 1', port1, ['entity_code']),
    ('Portfolio 2', port2, ['entity_code']),
    ('Entities', entities, ['entity_code']),
    ('Stock returns', returns, ['date', 'entity_code']),
]:
    n_before = len(df)
    df.drop_duplicates(subset=cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  {name:<20}: {n_before - len(df)} removed "
          f"→ {len(df)} remain")

# ================================================================
# STEP 9: MERGE INTO MODELING TABLES
# ================================================================
# Join each portfolio's EAD with entity characteristics.
# Calculate EL = PD x LGD x EAD for each obligor.
# This is the fundamental credit risk formula.

print("\n" + "-" * 65)
print("STEP 9: Building modeling tables")
print("-" * 65)


def build_modeling_table(portfolio_df, portfolio_name):
    """
    Merge portfolio EAD with entity PD, LGD, industry, region.
    Calculate Expected Loss: EL = PD x LGD x EAD.

    Unmatched entity codes are dropped and logged.
    Returns clean modeling DataFrame ready for simulation.
    """
    merged = portfolio_df.merge(
        entities[['entity_code', 'entity_name', 'industry',
                  'region', 'lgd', 'base_pd_hint']],
        on='entity_code', how='left'
    )
    n_unmatched = merged['base_pd_hint'].isna().sum()
    if n_unmatched > 0:
        unmatched = merged[merged['base_pd_hint'].isna()][
            'entity_code'].tolist()
        print(f"  {portfolio_name}: {n_unmatched} unmatched "
              f"→ {unmatched}")
    merged = merged.dropna(subset=['base_pd_hint', 'lgd', 'EAD'])
    merged = merged.rename(columns={
        'base_pd_hint': 'PD', 'lgd': 'LGD'})
    # EL = PD x LGD x EAD (fundamental credit risk formula)
    merged['EL'] = merged['PD'] * merged['LGD'] * merged['EAD']
    return merged.reset_index(drop=True)


table1 = build_modeling_table(port1, "Portfolio 1")
table2 = build_modeling_table(port2, "Portfolio 2")

# ================================================================
# STEP 10: SUMMARY AND SAVE
# ================================================================
print("\n" + "=" * 65)
print("CLEANING SUMMARY")
print("=" * 65)

for name, table in [("Portfolio 1", table1), ("Portfolio 2", table2)]:
    print(f"\n  {name}:")
    print(f"    Obligors:  {len(table)}")
    print(f"    Total EAD: EUR {table['EAD'].sum() / 1e6:.2f}M")
    print(f"    Mean PD:   {table['PD'].mean() * 100:.3f}%")
    print(f"    Mean LGD:  {table['LGD'].mean() * 100:.1f}%")
    print(f"    Total EL:  EUR {table['EL'].sum() / 1e6:.4f}M")

# Save CSV outputs → report Table 1
table1.to_csv(os.path.join(DIR_RESULTS, 'cleaned_portfolio1.csv'),
              index=False)
table2.to_csv(os.path.join(DIR_RESULTS, 'cleaned_portfolio2.csv'),
              index=False)
print("\n  Saved: cleaned_portfolio1.csv → report Table 1")
print("  Saved: cleaned_portfolio2.csv → report Table 1")

# Save cleaned objects for next script
pickle.dump({
    'table1': table1, 'table2': table2,
    'returns': returns, 'idx_ret': idx_ret,
    'defaults': defaults, 'entities': entities,
}, open(os.path.join(DIR_RESULTS, 'step1_cleaned.pkl'), 'wb'))
print("  Saved: step1_cleaned.pkl → used by 02_parameter_estimation.py")

print("\n" + "=" * 65)
print("PART 1 COMPLETE — Next: run 02_parameter_estimation.py")
print("=" * 65)
