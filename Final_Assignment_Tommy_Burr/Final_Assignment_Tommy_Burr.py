# =============================================================================
# step 1: define directory path
# =============================================================================

PATH = r"C:\Users\tommy\PycharmProjects\Credit-Risk-Student-Workspace\Final_Assignment_Tommy_Burr"

# =============================================================================
# TABLE OF CONTENTS:
#
# line   83- 121 - EDA: explorative data analysis
# line  122- 182 - list of problems in the data
# line  184- 676 - data cleaning
# line  677- 718 - asset correlation
# line  719- 973 - Model 1: BB-Model
# line  974-1233 - Model 2: Gaussian Thrshold / One Factor
# line 1234-1401 - Portfolio 3: Two-Factor Gaussian Model
# line 1402-1438 - final comparison
#
# =============================================================================


# =============================================================================
# step 2: import packages
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import betabinom
from scipy.stats import multivariate_normal   # for implied_default_correlation (Correlation.py)
from scipy.optimize import brentq             # for rho inversion (Correlation.py line 83)

warnings.filterwarnings("ignore")

np.random.seed(42)

# =============================================================================
# step 3: seaborn style setting
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
plt.rcParams["figure.dpi"] = 110

# =============================================================================
# step 4: load data
# =============================================================================

portfolio_1_raw = pd.read_csv(f"{PATH}/portfolio_1.csv")
portfolio_2_raw = pd.read_csv(f"{PATH}/portfolio_2.csv")
portfolio_3_raw = pd.read_csv(f"{PATH}/portfolio_3.csv")
entities_info_raw = pd.read_csv(f"{PATH}/entities_info.csv")
default_history_raw = pd.read_csv(f"{PATH}/default_history_20y.csv")
stock_returns_raw = pd.read_csv(f"{PATH}/stock_returns.csv")
industry_index_raw = pd.read_csv(f"{PATH}/industry_index_returns.csv")

print("=" * 60)
print("CREDIT RISK -- data upload structure")
print("=" * 60)

print(f"portfolio_1:       {portfolio_1_raw.shape[0]:>6} rows, {portfolio_1_raw.shape[1]} columns")
print(f"portfolio_2:       {portfolio_2_raw.shape[0]:>6} rows, {portfolio_2_raw.shape[1]} columns")
print(f"portfolio_3:       {portfolio_3_raw.shape[0]:>6} rows, {portfolio_3_raw.shape[1]} columns")
print(f"entities_info:     {entities_info_raw.shape[0]:>6} rows, {entities_info_raw.shape[1]} columns")
print(f"default_history:   {default_history_raw.shape[0]:>6} rowsn, {default_history_raw.shape[1]} columns")
print(f"stock_returns:     {stock_returns_raw.shape[0]:>6} rowsn, {stock_returns_raw.shape[1]} columns")
print(f"industry_index:    {industry_index_raw.shape[0]:>6} rows, {industry_index_raw.shape[1]} columns")

# =============================================================================
# step 5: copy for working
# =============================================================================

p1 = portfolio_1_raw.copy()
p2 = portfolio_2_raw.copy()
p3 = portfolio_3_raw.copy()
ei = entities_info_raw.copy()
dh = default_history_raw.copy()
sr = stock_returns_raw.copy()
ii = industry_index_raw.copy()

# =============================================================================
# step 6: explorative data analysis to get a first impression of the data
# =============================================================================

print("\n" + "=" * 60)
print("explorative data analysis for first impression of the data")
print("=" * 60)

print("basis information and missing values for each dataset ---")

datasets = {
    "portfolio_1": p1,
    "portfolio_2": p2,
    "portfolio_3": p3,
    "entities_info": ei,
    "default_history": dh,
    "stock_returns": sr,
    "industry_index": ii,
}

for name, df in datasets.items():
    n_nan = df.isna().sum().sum()
    n_dupl = df.duplicated().sum()
    print(f"  {name:<20} {df.shape[0]:>6} x {df.shape[1]:<2}  "
          f"| NaN total: {n_nan:>4}  | duplicates: {n_dupl}")

for name, df in datasets.items():
    print("\n" + "=" * 60)
    print(f" {name}")
    print("=" * 60)
    print(df.describe().round(2))

print("\n" + "=" * 60)
print("plots for data structure, missing values, and distributions")
print("=" * 60)
"""
[EDA plots omitted -- see original code]
"""

# =============================================================================
# data problems found during EDA
# =============================================================================

print("\n" + "=" * 60)
print("EDA: identified issues summary")
print("=" * 60)
print("""
FILE: entities_info.csv
  • entity_code: 3 duplicate entries after normalization (A-FKUFQE,
    B-TB1OZ6, B-MHTAKQ -- one uppercase, one lowercase)
  • industry: 8 different spellings for only 2 industries
    ("Industry A", "industry a", "Ind. A", "IndustryA", etc.)
  • region: 4 spellings for the same regions
    ("South America" / "SouthAmerica" / "south america", etc.)
  • lgd: 6 missing values (NaN)
  • base_pd_hint: 5 missing values (NaN)
  • irrelevant_score: according to its name not a relevant parameter -> ignore

FILE: portfolio_1.csv
  • entity_code: 4 entries in lowercase 
  • EAD: 2 missing values (NaN)

FILE: portfolio_2.csv
  • entity_code: 4 entries in lowercase 
  • EAD: 2 missing values (NaN)
  • BWMAEKB: duplicate entry (row 2 and 30, identical EAD)
  • 7 codes without hyphen (e.g. BWMAEKB instead of B-WMAEKB):
    BWMAEKB, BSQ4OLC, BJS2LBB, BY6J2QG, B3C5B0M, BFVUAOK, BBN8BND
    -> missing hyphen in entity_code
    -> no PD, no LGD, no default history -> not modelable

FILE: portfolio_3.csv
  • entity_code: 8 entries in lowercase 
  • EAD: 5 missing values (NaN)
  • 3 codes without hyphen (same root cause as portfolio_2)

FILE: default_history_20y.csv
  • default_event: 47 missing values (NaN) -- unclear whether default or not
  • 15 duplicate entity-year combinations (identical values)
  • source_flag: 27% manual/archived entries (lower reliability)

FILE: stock_returns.csv
  • date: TWO mixed formats (YYYY-MM-DD and DD-MM-YYYY)
  • entity_code: leading spaces for some codes + missing hyphen
  • stock_return: 2242 missing values 
  • 40 duplicates (entity + date) after correct parsing 

FILE: industry_index_returns.csv
  • date: TWO mixed formats (YYYY-MM-DD and DD/MM/YYYY)
  • industry: 8 spellings for 2 industries (same as entities_info)
  • index_return: 30 missing values (NaN)

OVERALL:
  • Portfolio 1 and Portfolio 2 each have 0 missing JOIN coverages
    in stock_returns (100%), but Portfolio 2 has 7 codes with missing
    hyphens 
  • Portfolio 3 is a mixed portfolio (Industry A + B) 
  • booking_date, internal_comment (portfolios), and region_note
    (entities_info) are not model parameters -> ignore
""")

# =============================================================================
# STEP 7: DATA CLEANING
# =============================================================================

# =============================================================================
# step 7.1: normalize entity codes
# =============================================================================

print("step 7.1: normalize entity codes")

for df, name in [(p1, "portfolio_1"), (p2, "portfolio_2"), (p3, "portfolio_3"),
                 (ei, "entities_info"), (dh, "default_history"), (sr, "stock_returns")]:
    original = df["entity_code"].copy()
    df["entity_code"] = df["entity_code"].str.strip().str.upper()
    changed_case = (original != df["entity_code"]).sum()

    # Fix missing hyphens: codes like "B3C5B0M" should be "B-3C5B0M".
    no_hyphen_mask = ~df["entity_code"].str.contains("-", na=False)
    n_hyphen_fix   = no_hyphen_mask.sum()
    if n_hyphen_fix > 0:
        df.loc[no_hyphen_mask, "entity_code"] = (
            df.loc[no_hyphen_mask, "entity_code"].str[0]
            + "-"
            + df.loc[no_hyphen_mask, "entity_code"].str[1:]
        )

    changed_total = (original != df["entity_code"]).sum()
    if changed_total > 0:
        parts = []
        if changed_case > 0:
            parts.append(f"{changed_case} case/whitespace")
        if n_hyphen_fix > 0:
            parts.append(f"{n_hyphen_fix} missing hyphens")
        print(f"  {name}: {changed_total} codes normalized ({', '.join(parts)})")
    else:
        print(f"  {name}: no changes required")

print("  -> strip() + upper() + hyphen fix applied to all entity_codes")

# =============================================================================
# step 7.2: normalize industry names
# =============================================================================

print("step 7.2: normalize industry names")

INDUSTRY_MAP = {
    "Industry A": "Industry A",
    "industry a": "Industry A",
    "Ind. A":     "Industry A",
    "IndustryA":  "Industry A",
    "Industry B": "Industry B",
    "industry b": "Industry B",
    "Ind. B":     "Industry B",
    "IndustryB":  "Industry B",
}

def normalize_industry(series):
    cleaned = series.str.strip()
    result  = cleaned.map(INDUSTRY_MAP)
    unknown = cleaned[result.isna() & cleaned.notna()].unique()
    if len(unknown) > 0:
        print(f"  WARNING: unknown industry values: {unknown}")
    return result

ei["industry_norm"] = normalize_industry(ei["industry"])
print(f"  entities_info: {ei['industry_norm'].notna().sum()}/{len(ei)} mapped")

ii["industry_norm"] = normalize_industry(ii["industry"])
print(f"  industry_index: {ii['industry_norm'].notna().sum()}/{len(ii)} mapped")

print(f"  Unique values after normalization: "
      f"{sorted(ei['industry_norm'].dropna().unique())}")

ei = ei.drop(columns=["industry"])
ii = ii.drop(columns=["industry"])

# =============================================================================
# step 7.2b: normalize region names
# =============================================================================

print("step 7.2b: normalize region names")

REGION_MAP = {
    "Europe":       "Europe",
    "europe":       "Europe",
    "Asia":         "Asia",
    "asia":         "Asia",
    "North America":"North America",
    "NorthAmerica": "North America",
    "South America":"South America",
    "SouthAmerica": "South America",
    "south america":"South America",
}

unknown_str = ei["region"][ei["region"].notna() & ~ei["region"].isin(REGION_MAP)].unique()
if len(unknown_str) > 0:
    print(f"  WARNING: unknown region strings (not in map): {unknown_str}")

ei["region"] = ei["region"].map(REGION_MAP)

n_nan = ei["region"].isna().sum()
print(f"  All regions normalized: {sorted(ei['region'].dropna().unique())}")
if n_nan > 0:
    print(f"  Note: {n_nan} entities have no region value (genuinely missing in raw data)")

# =============================================================================
# step 7.3: drop duplicates in entities_info
# =============================================================================

print("step 7.3: remove duplicates in entities_info")

duplicated_mask = ei["entity_code"].duplicated(keep=False)
dupl_codes = ei[duplicated_mask]["entity_code"].unique()
print(f"  Found duplicate entity_codes: {list(dupl_codes)}")

conflict_found = False
for code in dupl_codes:
    group = ei[ei["entity_code"] == code]
    if group["lgd"].nunique(dropna=False) > 1 or \
       group["base_pd_hint"].nunique(dropna=False) > 1:
        print(f"  CONFLICT for {code}: values differ!")
        print(group[["entity_code", "lgd", "base_pd_hint"]].to_string())
        conflict_found = True

if not conflict_found:
    print("  All duplicates have identical values -- safe to remove")

n_before = len(ei)
ei = ei.drop_duplicates(subset="entity_code", keep="first")
n_after = len(ei)
print(f"  Rows before: {n_before}, after: {n_after} ({n_before - n_after} removed)")

# =============================================================================
# step 7.4: clean date formats
# =============================================================================

print("step 7.4: clean date formats")

# ----- stock_returns -----
print("  stock_returns:")
mask_iso_sr = sr["date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
mask_dmy_sr = sr["date"].str.match(r"^\d{2}-\d{2}-\d{4}$")
print(f"    ISO format (YYYY-MM-DD): {mask_iso_sr.sum()} rows")
print(f"    DD-MM-YYYY format:       {mask_dmy_sr.sum()} rows")
print(f"    Unknown formats:         {(~mask_iso_sr & ~mask_dmy_sr).sum()} rows")

sr_iso = sr[mask_iso_sr].copy()
sr_dmy = sr[mask_dmy_sr].copy()
sr_iso["date_parsed"] = pd.to_datetime(sr_iso["date"], format="%Y-%m-%d")
sr_dmy["date_parsed"] = pd.to_datetime(sr_dmy["date"], format="%d-%m-%Y")
sr = pd.concat([sr_iso, sr_dmy], ignore_index=True)
print(f"    -> date_parsed created: {len(sr)} rows total")
print(f"    Date range: {sr['date_parsed'].min().date()} to {sr['date_parsed'].max().date()}")

# ----- industry_index_returns -----
print("  industry_index_returns:")
mask_iso_ii = ii["date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
mask_dmy_ii = ii["date"].str.contains("/")
print(f"    ISO format (YYYY-MM-DD): {mask_iso_ii.sum()} rows")
print(f"    DD/MM/YYYY format:       {mask_dmy_ii.sum()} rows")

ii_iso = ii[mask_iso_ii].copy()
ii_dmy = ii[mask_dmy_ii].copy()
ii_iso["date_parsed"] = pd.to_datetime(ii_iso["date"], format="%Y-%m-%d")
ii_dmy["date_parsed"] = pd.to_datetime(ii_dmy["date"], format="%d/%m/%Y")
ii = pd.concat([ii_iso, ii_dmy], ignore_index=True)
print(f"    -> date_parsed created: {len(ii)} rows total")
print(f"    Date range: {ii['date_parsed'].min().date()} to {ii['date_parsed'].max().date()}")

# ----- default_history -----
dh["date_parsed"] = pd.to_datetime(dh["date"], format="%m/%d/%Y")
dh["year"] = dh["date_parsed"].dt.year
print(f"  default_history: years {dh['year'].min()} to {dh['year'].max()}")

# =============================================================================
# step 7.5: drop duplicates in default_history
# =============================================================================

print("step 7.5: remove duplicates in default_history")

n_before = len(dh)
dup_mask = dh.duplicated(subset=["year", "entity_code"], keep=False)

if dup_mask.sum() > 0:
    conflicts = (dh[dup_mask]
                 .groupby(["year", "entity_code"])["default_event"]
                 .nunique())
    real_conflicts = conflicts[conflicts > 1]
    print(f"  Duplicates found: {dup_mask.sum()} rows ({dup_mask.sum() // 2} pairs)")
    if len(real_conflicts) > 0:
        print(f"  CONFLICTS (different values): {len(real_conflicts)}")
        print(real_conflicts)
    else:
        print("  No conflicts -- all duplicates have identical values")

dh = (dh
      .groupby(["year", "entity_code", "date", "date_parsed"], dropna=False)
      .agg(default_event=("default_event", "max"),
           source_flag=("source_flag", "first"))
      .reset_index())

n_after = len(dh)
print(f"  Rows before: {n_before}, after: {n_after} ({n_before - n_after} removed)")

# =============================================================================
# step 7.6: remove duplicates in stock_returns
# =============================================================================

print("step 7.6: remove duplicates in stock_returns")

n_before = len(sr)
dup_mask_sr = sr.duplicated(subset=["date_parsed", "entity_code"], keep=False)

if dup_mask_sr.sum() > 0:
    conflicts_sr = (sr[dup_mask_sr]
                    .groupby(["date_parsed", "entity_code"])["stock_return"]
                    .nunique())
    real_conflicts_sr = conflicts_sr[conflicts_sr > 1]
    print(f"  Duplicates: {dup_mask_sr.sum()} rows")
    if len(real_conflicts_sr) > 0:
        print(f"  CONFLICTS (different returns!): {len(real_conflicts_sr)}")
    else:
        print("  No conflicts -- all duplicates identical, safe to remove")

sr = sr.drop_duplicates(subset=["date_parsed", "entity_code"], keep="first")
n_after = len(sr)
print(f"  Rows before: {n_before}, after: {n_after} ({n_before - n_after} removed)")

# =============================================================================
# step 7.7: remove duplicates in all three portfolios
# =============================================================================

print("step 7.7: remove duplicates in all portfolios")

for df, name in [(p1, "portfolio_1"), (p2, "portfolio_2"), (p3, "portfolio_3")]:
    n_before = len(df)
    dupl = df[df["entity_code"].duplicated(keep=False)]
    if len(dupl) > 0:
        print(f"  {name}: duplicate found: "
              f"{dupl['entity_code'].unique().tolist()} "
              f"| EAD values: {dupl['EAD'].tolist()}")
    df.drop_duplicates(subset="entity_code", keep="first", inplace=True)
    n_after = len(df)
    print(f"  {name}: rows before: {n_before}, after: {n_after} "
          f"({n_before - n_after} removed)")

# =============================================================================
# step 7.8: exclude missing EAD values
# =============================================================================

print("step 7.8: exclude missing EAD values")

for df, name in [(p1, "portfolio_1"), (p2, "portfolio_2"), (p3, "portfolio_3")]:
    n_nan = df["EAD"].isna().sum()
    if n_nan > 0:
        excluded = df[df["EAD"].isna()][["entity_code", "EAD"]].to_string(index=False)
        print(f"  {name}: {n_nan} positions without EAD (excluded):")
        print(f"    {excluded}")

p1 = p1.dropna(subset=["EAD"])
p2 = p2.dropna(subset=["EAD"])
p3 = p3.dropna(subset=["EAD"])
print(f"  After exclusion: P1={len(p1)}, P2={len(p2)}, P3={len(p3)} positions")

# =============================================================================
# step 7.9: handle missing LGD
# =============================================================================

print("step 7.9: handle missing LGD values")

lgd_by_industry = (ei.groupby("industry_norm")["lgd"]
                   .mean()
                   .rename("lgd_industry_mean"))
print(f"  LGD averages by industry: {lgd_by_industry.to_dict()}")

missing_lgd = ei[ei["lgd"].isna()][["entity_code", "industry_norm", "lgd"]]
if len(missing_lgd) > 0:
    print(f"  {len(missing_lgd)} missing LGD values:")
    print(missing_lgd.to_string(index=False))

ei = ei.merge(lgd_by_industry, on="industry_norm", how="left")
ei["lgd_imputed"] = ei["lgd"].isna().astype(int)
ei["lgd"] = ei["lgd"].fillna(ei["lgd_industry_mean"])
ei = ei.drop(columns=["lgd_industry_mean"])
print(f"  {ei['lgd_imputed'].sum()} LGD values filled with industry average")

# =============================================================================
# step 7.10: handle missing base_pd_hint
# =============================================================================

print("step 7.10: handle missing base_pd_hint values")

pd_by_industry = (ei.groupby("industry_norm")["base_pd_hint"]
                  .mean()
                  .rename("pd_industry_mean"))
print(f"  PD averages by industry: {pd_by_industry.to_dict()}")

missing_pd = ei[ei["base_pd_hint"].isna()][["entity_code", "industry_norm", "base_pd_hint"]]
if len(missing_pd) > 0:
    print(f"  {len(missing_pd)} missing PD values:")
    print(missing_pd.to_string(index=False))

ei = ei.merge(pd_by_industry, on="industry_norm", how="left")
ei["pd_imputed"] = ei["base_pd_hint"].isna().astype(int)
ei["base_pd_hint"] = ei["base_pd_hint"].fillna(ei["pd_industry_mean"])
ei = ei.drop(columns=["pd_industry_mean"])
print(f"  {ei['pd_imputed'].sum()} PD values filled with industry average")

# =============================================================================
# step 7.11: handle missing default_event values
# =============================================================================

print("step 7.11: handle missing default_event values")

n_nan_de = dh["default_event"].isna().sum()
print(f"  Missing default_event values: {n_nan_de}")
print(f"  Share: {n_nan_de / len(dh) * 100:.1f}% of all observations")

dh_clean = dh.dropna(subset=["default_event"]).copy()
print(f"  After exclusion: {len(dh_clean)} observations (from {len(dh)})")

dh_sensitivity = dh.copy()
dh_sensitivity["default_event"] = dh_sensitivity["default_event"].fillna(0)
print("  Sensitivity version created: NaNs treated as 0")

dh = dh_clean

# =============================================================================
# step 7.12: handle missing price data
# =============================================================================

print("step 7.12: handle missing price data")

n_nan_sr = sr["stock_return"].isna().sum()
n_nan_ii = ii["index_return"].isna().sum()
print(f"  stock_returns:  {n_nan_sr} NaN ({n_nan_sr / len(sr) * 100:.1f}%)")
print(f"  industry_index: {n_nan_ii} NaN ({n_nan_ii / len(ii) * 100:.1f}%)")

# =============================================================================
# step 7.13: safety check -- exclude any remaining unmodelable companies
# =============================================================================

print("step 7.13: safety check for companies without master data")

ei_codes = set(ei["entity_code"])

for df_ref, name in [("p1", "portfolio_1"), ("p2", "portfolio_2"), ("p3", "portfolio_3")]:
    df = eval(df_ref)
    missing = sorted(set(df["entity_code"]) - ei_codes)
    if missing:
        print(f"  {name}: {len(missing)} entities still without master data (excluded):")
        for code in missing:
            ead = df[df["entity_code"] == code]["EAD"].values
            print(f"    {code}  EAD={ead}")
        mask = df["entity_code"].isin(ei_codes)
        if df_ref == "p1":
            p1 = df[mask].copy()
        elif df_ref == "p2":
            p2 = df[mask].copy()
        else:
            p3 = df[mask].copy()
    else:
        print(f"  {name}: all codes present in entities_info")

print(f"  Final portfolio sizes: P1={len(p1)}, P2={len(p2)}, P3={len(p3)} positions")

# =============================================================================
# step 7.14: quality check and summary
# =============================================================================

print("\n" + "=" * 60)
print("step 7.14: quality check and summary")
print("=" * 60)

print("\n[Portfolios after cleaning]")
for df, name in [(p1, "Portfolio 1"), (p2, "Portfolio 2"), (p3, "Portfolio 3")]:
    n_dup  = df["entity_code"].duplicated().sum()
    status = "OK" if n_dup == 0 else f"WARNING: {n_dup} duplicates!"
    print(f"  {name}: {len(df)} positions | "
          f"Total EAD: {df['EAD'].sum() / 1e6:.1f}M | "
          f"NaN EAD: {df['EAD'].isna().sum()} | "
          f"Duplicates: {status}")

print("\n[entities_info after cleaning]")
print(f"  Unique entities:     {ei['entity_code'].nunique()}")
print(f"  Missing LGD:         {ei['lgd'].isna().sum()}")
print(f"  Missing base_pd_hint:{ei['base_pd_hint'].isna().sum()}")
print(f"  Imputed LGD:         {ei['lgd_imputed'].sum()}")
print(f"  Imputed PD:          {ei['pd_imputed'].sum()}")
print(f"  Industries:          {sorted(ei['industry_norm'].dropna().unique())}")
print(f"  Regions:             {sorted(ei['region'].dropna().unique())}")

raw_cols_remaining = [c for c in ["industry"] if c in ei.columns]
if raw_cols_remaining:
    print(f"  WARNING: raw columns still present: {raw_cols_remaining}")
else:
    print("  Raw 'industry' column: removed (only 'industry_norm' remains)")

print("\n[default_history after cleaning]")
print(f"  Rows:           {len(dh)}")
print(f"  Years:          {dh['year'].min()} to {dh['year'].max()}")
print(f"  Entities:       {dh['entity_code'].nunique()}")
print(f"  Total defaults: {int(dh['default_event'].sum())}")
print(f"  Remaining NaN:  {dh['default_event'].isna().sum()}")

print("\n[stock_returns after cleaning]")
print(f"  Rows:       {len(sr)}")
print(f"  Entities:   {sr['entity_code'].nunique()}")
print(f"  NaN returns:{sr['stock_return'].isna().sum()} (excluded at regression step)")
print(f"  Date range: {sr['date_parsed'].min().date()} to {sr['date_parsed'].max().date()}")

print("\n[industry_index after cleaning]")
print(f"  Rows:        {len(ii)}")
print(f"  Industries:  {sorted(ii['industry_norm'].dropna().unique())}")
print(f"  NaN returns: {ii['index_return'].isna().sum()} (excluded at regression step)")
print(f"  Date range:  {ii['date_parsed'].min().date()} to {ii['date_parsed'].max().date()}")
raw_cols_ii = [c for c in ["industry"] if c in ii.columns]
if raw_cols_ii:
    print(f"  WARNING: raw columns still present: {raw_cols_ii}")
else:
    print("  Raw 'industry' column: removed (only 'industry_norm' remains)")

print("\n[JOIN check: portfolio vs entities_info]")
for df, name in [(p1, "Portfolio 1"), (p2, "Portfolio 2"), (p3, "Portfolio 3")]:
    codes   = list(df["entity_code"])
    unique  = set(codes)
    found   = unique & ei_codes
    missing = unique - ei_codes
    n_dup   = len(codes) - len(unique)
    suffix  = f" | {n_dup} duplicate rows" if n_dup else ""
    print(f"  {name}: {len(found)}/{len(unique)} unique codes in entities_info"
          f"{' | OK' if not missing else f' | MISSING: {missing}'}{suffix}")

print("\n[JOIN check: portfolio vs stock_returns]")
sr_codes = set(sr["entity_code"])
for df, name in [(p1, "Portfolio 1"), (p2, "Portfolio 2"), (p3, "Portfolio 3")]:
    unique  = set(df["entity_code"])
    found   = unique & sr_codes
    missing = unique - sr_codes
    print(f"  {name}: {len(found)}/{len(unique)} unique codes in stock_returns"
          f"{' | OK' if not missing else f' | MISSING: {missing}'}")

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("Cleaned DataFrames: p1, p2, p3, ei, dh, sr, ii")
print("Sensitivity version: dh_sensitivity (NaN = 0)")
print("=" * 60)

# --- Save all cleaned datasets as CSV ---
print("\nSaving cleaned datasets to CSV ...")

cleaned_datasets = {
    "portfolio_1_cleaned":           p1,
    "portfolio_2_cleaned":           p2,
    "portfolio_3_cleaned":           p3,
    "entities_info_cleaned":         ei,
    "default_history_cleaned":       dh,
    "stock_returns_cleaned":         sr,
    "industry_index_cleaned":        ii,
    "default_history_sensitivity_cleaned": dh_sensitivity,
}

for filename, df in cleaned_datasets.items():
    filepath = f"{PATH}/{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"  {filename}.csv  ({len(df)} rows, {len(df.columns)} cols)")

print("  -> All cleaned datasets saved.")


# =============================================================================
# STEP 8: CREATE WORKING COPIES
# =============================================================================

print("=" * 60)
print("STEP 8: WORKING COPIES")
print("=" * 60)

m1_p1 = p1.copy()   # Model 1 (Beta-Binomial)
m1_p2 = p2.copy()
m1_p3 = p3.copy()
m1_ei = ei.copy()
m1_dh = dh.copy()

m2_p1 = p1.copy()   # Model 2 (Gaussian One-Factor)
m2_p2 = p2.copy()
m2_p3 = p3.copy()
m2_ei = ei.copy()
m2_dh = dh.copy()   # used in Gaussian loop to estimate rho_y -> rho

print("  Working copies ready: m1_* (Beta-Binomial), m2_* (Gaussian)")


# =============================================================================
# STEP 8b: ASSET CORRELATION -- DEFAULT CORRELATION INVERSION (Correlation.py)
# =============================================================================
#
# Source: Correlation.py (lecture exercise)
#   Lines 44-75 : implied_default_correlation() -- computes rho_y given rho
#   Lines 80-83 : brentq numerical inversion to find rho given rho_y

print("\n" + "=" * 60)
print("STEP 8b: ASSET CORRELATION -- CORRELATION.PY APPROACH")
print("=" * 60)

# -----------------------------------------------------------------------
# implied_default_correlation() -- Correlation.py lines 44-75
# -----------------------------------------------------------------------
# Given a factor loading rho and unconditional PD p, returns the implied
# default correlation rho_y = Corr(Y_i, Y_j) under the Gaussian one-factor
# model.  Identical to Correlation.py: same formula, same variable names.

def implied_default_correlation(rho, p):
    """
    Gaussian one-factor model  [Correlation.py lines 44-75]:

        X_i = rho * F + sqrt(1 - rho^2) * epsilon_i

    Asset correlation = rho^2  [Correlation.py lines 60-61].
    Returns Corr(Y_i, Y_j) where Y_i = 1{X_i <= Phi^{-1}(p)}.
    """
    d          = norm.ppf(p)
    asset_corr = rho ** 2
    cov_matrix = [[1.0, asset_corr],
                  [asset_corr, 1.0]]
    joint_default_prob = multivariate_normal.cdf(
        [d, d], mean=[0, 0], cov=cov_matrix
    )
    return (joint_default_prob - p ** 2) / (p * (1 - p))

print("  implied_default_correlation() defined  [Correlation.py lines 44-75]")
print("  brentq inversion ready                 [Correlation.py line 83]")
print("  rho_y estimated per portfolio in Model 2 loop (same as BB model)")


# =============================================================================
# MODEL 1: BETA-BINOMIAL MODEL
# =============================================================================
#
#   PRIMARY SOURCE
#   Beta-binomial.py
#     Lines  43-117 : model derivation and calibration formulas
#     Lines 119-122 : s, alpha, beta formulas
#     Lines 151-154 : simulation (p_random / Q in lecture notation)
#     Lines 193-201 : analytical PMF via betabinom.pmf
#     Line  237     : Beta-Binomial variance formula (used for rho_y)
#
#   I also used:
#   Credit_Metrics.py lines 182-204 : VaR and ES from simulated losses
#
# rho_y estimation:
#   -- formula is from Beta-binomial.py line 237
#   Beta-binomial.py takes rho_y as a fixed input parameter (line 34).
#   Here rho_y is estimated from 20 years of historical default-rate data
#   by rearranging the BB variance formula (Beta-binomial.py line 237):
#
#     Var(D) = m * p * (1-p) * (1 + (m-1) * rho_y)
#
#   Dividing both sides by m^2 and solving:
#
#     rho_y = (Var(D/m) / (p*(1-p)/m) - 1) / (m-1)

print("\n" + "=" * 60)
print("MODEL 1: BETA-BINOMIAL MODEL -- ALL PORTFOLIOS")
print("=" * 60)

N = 100_000
np.random.seed(42)

portfolios_bb = {
    "Portfolio 1": m1_p1,
    "Portfolio 2": m1_p2,
    "Portfolio 3": m1_p3,
}

bb_results     = []
bb_loss_data   = []
bb_var_data    = []
bb_yearly_data = []

for portfolio_name, portfolio_raw in portfolios_bb.items():

    print("\n" + "=" * 60)
    print(f"BETA-BINOMIAL -- {portfolio_name.upper()}")
    print("=" * 60)

    portfolio = portfolio_raw.merge(
        m1_ei[["entity_code", "base_pd_hint", "lgd"]],
        on="entity_code", how="left"
    )

    m         = len(portfolio)
    total_EAD = portfolio["EAD"].sum()

    LGD_mean = np.average(portfolio["lgd"].values, weights=portfolio["EAD"].values)

    # p: unconditional default probability
    # -- Beta-binomial.py uses a single scalar p (line 33)
    # EAD-weighted portfolio average p
    p = np.sum(portfolio["EAD"] * portfolio["base_pd_hint"]) / total_EAD
    p = np.clip(p, 1e-6, 1 - 1e-6)   # numerical guard

    # --- rho_y: default correlation from historical default-rate data ---
    # -- Beta-binomial.py takes rho_y as a given (line 34)
    # Formula derived by rearranging the BB variance formula:
    #   rho_y = (Var(D/m) / (p*(1-p)/m) - 1) / (m-1)
    portfolio_codes = portfolio["entity_code"].unique()
    dh_portfolio = m1_dh[m1_dh["entity_code"].isin(portfolio_codes)].copy()

    yearly_defaults = (
        dh_portfolio
        .groupby("year")
        .agg(m_obs=("default_event", "count"), d_obs=("default_event", "sum"))
        .reset_index()
    )
    yearly_defaults["default_rate"] = yearly_defaults["d_obs"] / yearly_defaults["m_obs"]

    observed_variance    = yearly_defaults["default_rate"].var(ddof=1)
    average_m            = yearly_defaults["m_obs"].mean()
    independent_variance = p * (1 - p) / average_m

    rho_y_raw = (observed_variance / independent_variance - 1) / (average_m - 1)
    # Floor at 0.0001: ensures alpha and beta remain positive (rho_y > 0 required)
    rho_y = max(rho_y_raw, 0.0001)

    # Beta calibration (Beta-binomial.py lines 119-122)
    # s = alpha + beta = 1/rho_y - 1
    s     = 1 / rho_y - 1
    alpha = p * s
    beta  = (1 - p) * s

    print("\nParameters")
    print("-" * 40)
    print(f"  Obligors m:                 {m}")
    print(f"  Total EAD:                  {total_EAD:,.0f}")
    print(f"  Portfolio PD p:            {p:.6f}")
    print(f"  EAD-weighted LGD_mean:      {LGD_mean:.4f}")
    print(f"  Obs. default-rate variance: {observed_variance:.8f}")
    print(f"  Ind. default-rate variance: {independent_variance:.8f}")
    print(f"  Raw rho_y:                  {rho_y_raw:.6f}")
    print(f"  Final rho_y (floored):      {rho_y:.6f}")
    print(f"  s = 1/rho_y - 1:            {s:.4f}")
    print(f"  alpha:                      {alpha:.4f}")
    print(f"  beta:                       {beta:.4f}")

    # --- Simulation [Beta-binomial.py lines 151-154] --------------------
    # Notation: Q = common random default probability  [Lecture slide 94]
    #           M = number of defaults                 [Lecture slide 95]
    # Beta-binomial.py uses p_random (slide 97 switches to P notation);
    # this code uses Q and M to match the theoretical slides 94-95.

    # Step 1: Q ~ Beta(alpha, beta)  (Beta-binomial.py line 151: p_random)
    # Draw the common systematic default probability for each simulation trial.
    # All m obligors share the same Q in a given trial which creates dependence.
    Q = np.random.beta(alpha, beta, size=N)

    # Step 2: M | Q ~ Binomial(m, Q)  (Beta-binomial.py line 154: defaults_dep)
    # Given Q, each obligor defaults independently with probability Q.
    # M is the total number of defaults in the portfolio.
    M = np.random.binomial(n=m, p=Q, size=N)

    # Loss rate: L = (M / m) * LGD_mean
    # Consistent with analytical formula below: loss_k = (k/m) * LGD_mean
    # Beta-binomial.py does not compute a loss rate.
    loss_rate = (M / m) * LGD_mean

    # --- Simulated risk metrics [Credit_Metrics.py lines 200-204] -------
    EL_sim    = np.mean(loss_rate)
    VaR99_sim  = np.quantile(loss_rate, 0.99)   # Credit_Metrics.py line 202
    VaR999_sim = np.quantile(loss_rate, 0.999)  # Credit_Metrics.py line 201
    ES99_sim   = np.mean(loss_rate[loss_rate >= VaR99_sim])   # line 204
    ES999_sim  = np.mean(loss_rate[loss_rate >= VaR999_sim])  # line 203

    # --- Analytical PMF and risk metrics [Beta-binomial.py lines 193-201] ---
    # M ~ Beta-Binomial(m, alpha, beta)       [Lecture slide 95: P(M = k)]
    # P(M = k) = betabinom.pmf(k, m, alpha, beta)  [Beta-binomial.py line 201]
    k      = np.arange(0, m + 1)
    pmf    = betabinom.pmf(k, n=m, a=alpha, b=beta)
    loss_k = (k / m) * LGD_mean    # loss rate when exactly M = k defaults occur
    cdf    = np.cumsum(pmf)

    # Analytical EL = E[L] = E[M/m] * LGD_mean = p * LGD_mean
    EL_ana    = np.sum(loss_k * pmf)

    # Analytical VaR: smallest loss_k where CDF >= confidence level
    VaR99_ana  = loss_k[np.where(cdf >= 0.99)[0][0]]
    VaR999_ana = loss_k[np.where(cdf >= 0.999)[0][0]]

    # Analytical ES: discrete analog of ES definition from Credit_Metrics.py
    # Beta-binomial.py does not compute ES.
    tail_99  = loss_k >= VaR99_ana
    tail_999 = loss_k >= VaR999_ana
    ES99_ana  = np.sum(loss_k[tail_99]  * pmf[tail_99])  / np.sum(pmf[tail_99])
    ES999_ana = np.sum(loss_k[tail_999] * pmf[tail_999]) / np.sum(pmf[tail_999])

    print("\nRisk Metrics -- Simulation vs Analytical")
    print("-" * 40)
    print(f"  {'Metric':<20} {'Simulation':>12} {'Analytical':>12}")
    for label, s_val, a_val in [
        ("Expected Loss",  EL_sim,    EL_ana),
        ("VaR 99%",        VaR99_sim,  VaR99_ana),
        ("VaR 99.9%",      VaR999_sim, VaR999_ana),
        ("ES 99%",         ES99_sim,   ES99_ana),
        ("ES 99.9%",       ES999_sim,  ES999_ana),
    ]:
        print(f"  {label:<20} {s_val:>12.6f} {a_val:>12.6f}")

    for method, EL, v99, v999, e99, e999 in [
        ("Simulation", EL_sim,  VaR99_sim,  VaR999_sim,  ES99_sim,  ES999_sim),
        ("Analytical", EL_ana,  VaR99_ana,  VaR999_ana,  ES99_ana,  ES999_ana),
    ]:
        bb_results.append({
            "Portfolio": portfolio_name, "Method": method,
            "Expected Loss": EL, "VaR 99%": v99, "VaR 99.9%": v999,
            "ES 99%": e99, "ES 99.9%": e999,
        })

    bb_loss_data.append(pd.DataFrame({"Portfolio": portfolio_name, "Loss Rate": loss_rate}))
    bb_var_data.append({"Portfolio": portfolio_name,
                         "VaR 99%": VaR99_sim, "VaR 99.9%": VaR999_sim})
    yearly_copy = yearly_defaults.copy()
    yearly_copy["Portfolio"] = portfolio_name
    bb_yearly_data.append(yearly_copy)


beta_binomial_results_all = pd.DataFrame(bb_results)
bb_loss_plot_df   = pd.concat(bb_loss_data,   ignore_index=True)
bb_var_plot_df    = pd.DataFrame(bb_var_data)
bb_yearly_plot_df = pd.concat(bb_yearly_data, ignore_index=True)

print("\n" + "=" * 60)
print("BETA-BINOMIAL -- RESULTS TABLE (ALL PORTFOLIOS)")
print("=" * 60)
print(beta_binomial_results_all.round(6).to_string(index=False))


# =============================================================================
# PLOTS -- BETA-BINOMIAL MODEL
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Beta-Binomial: Simulated Loss Distribution", fontsize=14, fontweight="bold")

for ax, pname in zip(axes, portfolios_bb.keys()):
    data    = bb_loss_plot_df[bb_loss_plot_df["Portfolio"] == pname]
    var_row = bb_var_plot_df[bb_var_plot_df["Portfolio"] == pname].iloc[0]
    sns.histplot(data=data, x="Loss Rate", bins=40, kde=True, stat="density", ax=ax)
    ax.axvline(var_row["VaR 99%"],   linestyle="--", linewidth=1.5,
               label=f"VaR 99%: {var_row['VaR 99%']:.2%}")
    ax.axvline(var_row["VaR 99.9%"], linestyle=":",  linewidth=2,
               label=f"VaR 99.9%: {var_row['VaR 99.9%']:.2%}")
    ax.set_title(pname); ax.set_xlabel("Loss Rate"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

metrics_long_bb = beta_binomial_results_all.melt(
    id_vars=["Portfolio", "Method"],
    value_vars=["Expected Loss", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"],
    var_name="Metric", value_name="Loss Rate"
)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# =============================================================================
# THIS PLOT IS UESD AS FIGURE 1
# =============================================================================
fig.suptitle("Figure 1: Beta-Binomial: Simulation vs Analytical", fontsize=14, fontweight="bold")
for ax, pname in zip(axes, portfolios_bb.keys()):
    data = metrics_long_bb[metrics_long_bb["Portfolio"] == pname]
    sns.barplot(data=data, x="Metric", y="Loss Rate", hue="Method", ax=ax)
    ax.set_title(pname); ax.set_xlabel(""); ax.set_ylabel("Loss Rate")
    ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{PATH}/Figure_1_BB_Simulation_vs_Analytical.png", dpi=200, bbox_inches="tight")
print("  -> Figure 1 saved to Figure_1_BB_Simulation_vs_Analytical.png")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Historical Annual Default Rates (basis for rho_y)",
             fontsize=14, fontweight="bold")
for ax, pname in zip(axes, portfolios_bb.keys()):
    data = bb_yearly_plot_df[bb_yearly_plot_df["Portfolio"] == pname].copy()
    sns.lineplot(data=data, x="year", y="default_rate", marker="o", ax=ax)
    ax.axhline(data["default_rate"].mean(), linestyle="--",
               label=f"Mean: {data['default_rate'].mean():.2%}")
    ax.set_title(pname); ax.set_xlabel("Year"); ax.set_ylabel("Default Rate")
    ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# =============================================================================
# MODEL 2: GAUSSIAN ONE-FACTOR THRESHOLD MODEL
# =============================================================================
#
#   PRIMARY SOURCES
#   Portfolio_Threshold.py
#     Lines  38-42 : parameters (num_trials, num_obligors, prob_default,
#                    correlation)
#     Lines  47-76 : simulate_correlated_defaults() -- F, epsilon, X,
#                    threshold, defaults
#   Credit_Metrics.py
#     Lines  18-40 : model description (X_i = rho*F + sqrt(1-rho^2)*eps)
#     Lines 126-130: default threshold d = Phi^{-1}(pd)
#     Lines 163-175: simulation (F, epsilon, X, defaults, loss_sim)
#     Lines 182-204: empirical VaR and ES from simulated losses
#     Lines 252-263: analytical conditional loss function l(f)
#     Lines 277-282: analytical VaR = l(Phi^{-1}(1-alpha))
#     Lines 304-312: analytical ES via u-grid integration
#     Line  323    : analytical EL = sum_i w_i * LGD_i * pd_i
#
#   Correlation.py
#     Lines  44-75: implied_default_correlation() -- rho_y given rho
#     Lines  60-61: asset correlation = rho^2
#     Line   83   : brentq numerical inversion rho_y -> rho
#
# Factor loading rho -- Correlation.py approach:
#   1. Estimate rho_y from 20 years of annual default-rate variance
#      (same formula as Model 1, derived from Beta-binomial.py line 237)
#   2. Numerically invert rho_y = f(rho, p) via brentq [Correlation.py line 83]
#      to find the factor loading rho that produces the observed rho_y.


print("\n" + "=" * 60)
print("MODEL 2: GAUSSIAN ONE-FACTOR THRESHOLD MODEL -- ALL PORTFOLIOS")
print("=" * 60)

np.random.seed(42)

portfolios_g = {
    "Portfolio 1": m2_p1,
    "Portfolio 2": m2_p2,
    "Portfolio 3": m2_p3,
}

g_results    = []
g_parameters = []
g_loss_data  = []
g_var_data   = []

for portfolio_name, portfolio_raw in portfolios_g.items():

    print("\n" + "=" * 60)
    print(f"GAUSSIAN ONE-FACTOR -- {portfolio_name.upper()}")
    print("=" * 60)

    portfolio = portfolio_raw.merge(
        m2_ei[["entity_code", "base_pd_hint", "lgd", "industry_norm"]],
        on="entity_code", how="left"
    )

    m         = len(portfolio)
    total_EAD = portfolio["EAD"].sum()
    # Exposure weight w_i = EAD_i / EAD_total  [Credit_Metrics.py line 80]
    portfolio["exposure_weight"] = portfolio["EAD"] / total_EAD

    # --- p: unconditional default probability -------------------------
    # EAD-weighted portfolio average PD.
    p = np.sum(portfolio["EAD"] * portfolio["base_pd_hint"]) / total_EAD
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # --- Default threshold d = Phi^{-1}(p)  [Credit_Metrics.py lines 126-130]
    # Obligor i defaults if X_i <= d.
    default_threshold = norm.ppf(p)

    # --- rho_y: default correlation from historical default-rate data -------
    # Identical approach to Model 1 (BB): same formula, same data source.
    # Formula derived from Beta-binomial.py line 237 (BB variance formula):
    #   rho_y = (Var(D/m) / (p*(1-p)/m) - 1) / (m-1)
    portfolio_codes = portfolio["entity_code"].unique()
    dh_portfolio    = m2_dh[m2_dh["entity_code"].isin(portfolio_codes)].copy()

    yearly_defaults_g = (
        dh_portfolio
        .groupby("year")
        .agg(m_obs=("default_event", "count"), d_obs=("default_event", "sum"))
        .reset_index()
    )
    yearly_defaults_g["default_rate"] = (yearly_defaults_g["d_obs"]
                                          / yearly_defaults_g["m_obs"])

    observed_variance_g    = yearly_defaults_g["default_rate"].var(ddof=1)
    average_m_g            = yearly_defaults_g["m_obs"].mean()
    independent_variance_g = p * (1 - p) / average_m_g

    rho_y_raw_g = (observed_variance_g / independent_variance_g - 1) / (average_m_g - 1)
    # Floor at 0.0001 (same guard as BB model -- ensures brentq has a valid target)
    rho_y_g = max(rho_y_raw_g, 0.0001)

    # --- rho: factor loading via brentq inversion  [Correlation.py lines 80-83] ---
    # Find rho such that implied_default_correlation(rho, p) == rho_y_g.
    # implied_default_correlation() is defined in Step 8b (Correlation.py lines 44-75).
    rho = brentq(
        lambda r: implied_default_correlation(r, p) - rho_y_g,
        0.0, 0.999999
    )
    # Asset correlation = rho^2  [Correlation.py lines 60-61]
    asset_correlation = rho ** 2

    print("\nParameters")
    print("-" * 40)
    print(f"  Obligors m:                  {m}")
    print(f"  Total EAD:                   {total_EAD:,.0f}")
    print(f"  Portfolio PD p:             {p:.6f}")
    print(f"  Default threshold d:         {default_threshold:.6f}")
    print(f"  Raw rho_y:                   {rho_y_raw_g:.6f}")
    print(f"  Final rho_y (floored):       {rho_y_g:.6f}")
    print(f"  rho (Correlation.py brentq): {rho:.6f}")
    print(f"  Asset correlation rho^2:     {asset_correlation:.6f}")

    # --- Simulation [Credit_Metrics.py lines 163-175] ------------------
    exposure = portfolio["exposure_weight"].values
    lgd      = portfolio["lgd"].values

    # Common systematic factor F ~ N(0,1)  [Portfolio_Threshold.py line 47]
    F       = np.random.normal(0, 1, size=N)

    # Idiosyncratic shocks epsilon_i ~ N(0,1)  [Portfolio_Threshold.py line 49]
    epsilon = np.random.normal(0, 1, size=(N, m))

    # Latent variable X_i  [Credit_Metrics.py line 169]
    X = rho * F[:, None] + np.sqrt(1 - rho ** 2) * epsilon

    # Default indicators Y_i  [Credit_Metrics.py line 172]
    defaults_sim = (X <= default_threshold).astype(int)

    # Portfolio loss rate per simulation  [Credit_Metrics.py line 175]
    loss_rate = np.sum(exposure * lgd * defaults_sim, axis=1)

    # --- Simulated risk metrics [Credit_Metrics.py lines 200-204] ------
    EL_sim    = np.mean(loss_rate)
    VaR99_sim  = np.quantile(loss_rate, 0.99)   # Credit_Metrics.py line 202
    VaR999_sim = np.quantile(loss_rate, 0.999)  # Credit_Metrics.py line 201
    ES99_sim   = np.mean(loss_rate[loss_rate >= VaR99_sim])   # line 204
    ES999_sim  = np.mean(loss_rate[loss_rate >= VaR999_sim])  # line 203

    # --- Analytical EL [Credit_Metrics.py line 323] --------------------
    # EL = sum_i w_i * LGD_i * p
    weighted_lgd_sum = np.sum(exposure * lgd)
    EL_ana           = weighted_lgd_sum * p

    # --- Conditional loss function l(f) [Credit_Metrics.py lines 252-263] ---
    # p_i(f) = Phi( (d - rho*f) / sqrt(1-rho^2) )  [line 256]
    # l(f)   = sum_i w_i * LGD_i * p_i(f)           [lines 259-263]
    def conditional_pd(f):
        return norm.cdf((default_threshold - rho * f) / np.sqrt(1 - rho ** 2))

    def analytical_loss_function(f):
        return weighted_lgd_sum * conditional_pd(f)

    # --- Analytical VaR [Credit_Metrics.py lines 277-282] --------------
    # l(F) is decreasing in F, so VaR_alpha = l( Phi^{-1}(1-alpha) )
    VaR99_ana  = analytical_loss_function(norm.ppf(1 - 0.99))
    VaR999_ana = analytical_loss_function(norm.ppf(1 - 0.999))

    # --- Analytical ES [Credit_Metrics.py lines 304-312] ---------------
    # ES_alpha = mean{ l(Phi^{-1}(u)) : u in (0, 1-alpha] }
    u_grid_99  = np.linspace(1e-10, 1 - 0.99,  100_000)
    u_grid_999 = np.linspace(1e-10, 1 - 0.999, 100_000)
    ES99_ana   = analytical_loss_function(norm.ppf(u_grid_99)).mean()
    ES999_ana  = analytical_loss_function(norm.ppf(u_grid_999)).mean()

    print("\nRisk Metrics -- Simulation vs Analytical")
    print("-" * 40)
    print(f"  {'Metric':<20} {'Simulation':>12} {'Analytical':>12}")
    for label, s_val, a_val in [
        ("Expected Loss",  EL_sim,    EL_ana),
        ("VaR 99%",        VaR99_sim,  VaR99_ana),
        ("VaR 99.9%",      VaR999_sim, VaR999_ana),
        ("ES 99%",         ES99_sim,   ES99_ana),
        ("ES 99.9%",       ES999_sim,  ES999_ana),
    ]:
        print(f"  {label:<20} {s_val:>12.6f} {a_val:>12.6f}")

    for method, EL, v99, v999, e99, e999 in [
        ("Simulation", EL_sim,  VaR99_sim,  VaR999_sim,  ES99_sim,  ES999_sim),
        ("Analytical", EL_ana,  VaR99_ana,  VaR999_ana,  ES99_ana,  ES999_ana),
    ]:
        g_results.append({
            "Portfolio": portfolio_name, "Method": method,
            "Expected Loss": EL, "VaR 99%": v99, "VaR 99.9%": v999,
            "ES 99%": e99, "ES 99.9%": e999,
        })

    g_parameters.append({
        "Portfolio": portfolio_name, "m": m, "Total EAD": total_EAD,
        "p": p, "threshold": default_threshold,
        "rho_y": rho_y_g, "rho": rho, "asset corr. rho^2": asset_correlation,
    })

    g_loss_data.append(pd.DataFrame({"Portfolio": portfolio_name, "Loss Rate": loss_rate}))
    g_var_data.append({"Portfolio": portfolio_name,
                        "VaR 99%": VaR99_sim, "VaR 99.9%": VaR999_sim})


gaussian_threshold_results_all    = pd.DataFrame(g_results)
gaussian_threshold_parameters_all = pd.DataFrame(g_parameters)
g_loss_plot_df = pd.concat(g_loss_data, ignore_index=True)
g_var_plot_df  = pd.DataFrame(g_var_data)

print("\n" + "=" * 60)
print("GAUSSIAN ONE-FACTOR -- RESULTS TABLE (ALL PORTFOLIOS)")
print("=" * 60)
print(gaussian_threshold_results_all.round(6).to_string(index=False))
print("\nParameters:")
print(gaussian_threshold_parameters_all.round(6).to_string(index=False))


# =============================================================================
# PLOTS -- GAUSSIAN ONE-FACTOR MODEL
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Gaussian One-Factor: Simulated Loss Distribution",
             fontsize=14, fontweight="bold")
for ax, pname in zip(axes, portfolios_g.keys()):
    data    = g_loss_plot_df[g_loss_plot_df["Portfolio"] == pname]
    var_row = g_var_plot_df[g_var_plot_df["Portfolio"] == pname].iloc[0]
    sns.histplot(data=data, x="Loss Rate", bins=40, kde=True, stat="density", ax=ax)
    ax.axvline(var_row["VaR 99%"],   linestyle="--", linewidth=1.5,
               label=f"VaR 99%: {var_row['VaR 99%']:.2%}")
    ax.axvline(var_row["VaR 99.9%"], linestyle=":",  linewidth=2,
               label=f"VaR 99.9%: {var_row['VaR 99.9%']:.2%}")
    ax.set_title(pname); ax.set_xlabel("Loss Rate"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

metrics_long_g = gaussian_threshold_results_all.melt(
    id_vars=["Portfolio", "Method"],
    value_vars=["Expected Loss", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"],
    var_name="Metric", value_name="Loss Rate"
)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# =============================================================================
# THIS PLOT IS USED: FIGURE 2
# =============================================================================
fig.suptitle("Figure 2: Gaussian One-Factor: Simulation vs Analytical",
             fontsize=14, fontweight="bold")
for ax, pname in zip(axes, portfolios_g.keys()):
    data = metrics_long_g[metrics_long_g["Portfolio"] == pname]
    sns.barplot(data=data, x="Metric", y="Loss Rate", hue="Method", ax=ax)
    ax.set_title(pname); ax.set_xlabel(""); ax.set_ylabel("Loss Rate")
    ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{PATH}/Figure_2_Gaussian_Simulation_vs_Analytical.png", dpi=200, bbox_inches="tight")
print("  -> Figure 2 saved to Figure_2_Gaussian_Simulation_vs_Analytical.png")
plt.show()


# =============================================================================
# PORTFOLIO 3: ONE-FACTOR vs TWO-FACTOR GAUSSIAN THRESHOLD
# =============================================================================
#
#   PRIMARY SOURCE
#   Factors.py
#     Lines 132-146 : estimate correlation between two index return series
#     Lines 203-214 : simulate two correlated factors via
#                     np.random.multivariate_normal
#     Lines 245-249 : assign each company to its industry factor with
#                     np.where
#     Lines 265-276 : X_i = rho * F_company + sqrt(1-rho^2) * epsilon
#                     defaults = X_i <= threshold

print("\n" + "=" * 60)
print("PORTFOLIO 3: ONE-FACTOR vs TWO-FACTOR GAUSSIAN THRESHOLD")
print("=" * 60)

# --- Step 1: Factor correlation from industry index returns [Factors.py 132-146] ---
ii_A = (ii[ii["industry_norm"] == "Industry A"]
        .dropna(subset=["index_return"])
        .set_index("date_parsed")["index_return"])

ii_B = (ii[ii["industry_norm"] == "Industry B"]
        .dropna(subset=["index_return"])
        .set_index("date_parsed")["index_return"])

# Inner join by date + drop NaN -- same approach as Factors.py lines 118-126
aligned_returns = pd.concat([ii_A.rename("A"), ii_B.rename("B")],
                             axis=1, join="inner").dropna()

# Pearson correlation between the two index return series [Factors.py line 147]
factor_corr = aligned_returns["A"].corr(aligned_returns["B"])

print(f"\n  corr(F_A, F_B) = {factor_corr:.4f}  "
      f"({len(aligned_returns)} overlapping days, Factors.py method)")

# --- Step 2: Portfolio 3 full data with industry labels ---
p3_full = m2_p3.merge(
    m2_ei[["entity_code", "base_pd_hint", "lgd", "industry_norm"]],
    on="entity_code", how="left"
)
total_EAD_p3           = p3_full["EAD"].sum()
p3_full["exposure_weight"] = p3_full["EAD"] / total_EAD_p3

# --- Step 3: Per-industry p and threshold ---

two_factor_params = {}

for industry in ["Industry A", "Industry B"]:
    sub    = p3_full[p3_full["industry_norm"] == industry].copy()
    p_ind = np.clip(
        np.sum(sub["EAD"] * sub["base_pd_hint"]) / sub["EAD"].sum(), 1e-6, 1 - 1e-6
    )
    two_factor_params[industry] = {
        "p": p_ind, "threshold": norm.ppf(p_ind), "n": len(sub)
    }
    print(f"  {industry}: n={len(sub)}, p={p_ind:.4f}, "
          f"threshold={norm.ppf(p_ind):.4f}")

# --- Step 4: Simulate two correlated factors [Factors.py lines 203-214] ---
corr_matrix_2f = np.array([[1.0, factor_corr],
                             [factor_corr, 1.0]])

market_factors = np.random.multivariate_normal(
    mean=[0, 0], cov=corr_matrix_2f, size=N
)
F_A_sim = market_factors[:, 0]   # Industry A factor
F_B_sim = market_factors[:, 1]   # Industry B factor

print(f"\n  Simulated corr(F_A, F_B) = "
      f"{np.corrcoef(F_A_sim, F_B_sim)[0,1]:.4f}  (target: {factor_corr:.4f})")

# --- Step 5: Uniform rho for all obligors [Factors.py line 178 convention] ---
# Factors.py uses a single rho for all firms.
# Here: portfolio-level rho from the one-factor model for Portfolio 3.
is_A = (p3_full["industry_norm"].values == "Industry A")

rho_p3_uniform = gaussian_threshold_parameters_all[
    gaussian_threshold_parameters_all["Portfolio"] == "Portfolio 3"
]["rho"].values[0]

rho_per_obligor = np.full(len(p3_full), rho_p3_uniform)
print(f"\n  Uniform rho for all obligors (from one-factor P3): {rho_p3_uniform:.4f}")

# Industry-specific thresholds [Credit_Metrics.py line 130]
threshold_per_obligor = np.where(is_A,
                                  two_factor_params["Industry A"]["threshold"],
                                  two_factor_params["Industry B"]["threshold"])

# Each obligor gets only its own industry factor [Factors.py lines 245-249]
F_company = np.where(is_A, F_A_sim[:, None], F_B_sim[:, None])

# --- Step 6: Latent variables, defaults, losses [Factors.py lines 265-276] ---
epsilon_2f = np.random.normal(0, 1, size=(N, len(p3_full)))

# X_i = rho * F_industry(i) + sqrt(1-rho^2) * epsilon_i
# [Factors.py line 270; Credit_Metrics.py line 169]
X_2f = rho_per_obligor * F_company + np.sqrt(1 - rho_per_obligor ** 2) * epsilon_2f

defaults_2f  = (X_2f <= threshold_per_obligor).astype(int)   # [Factors.py line 274]
exposure_p3  = p3_full["exposure_weight"].values
lgd_p3       = p3_full["lgd"].values
loss_rate_2f = np.sum(exposure_p3 * lgd_p3 * defaults_2f, axis=1)  # [CM.py line 175]

EL_2f    = loss_rate_2f.mean()
VaR99_2f  = np.quantile(loss_rate_2f, 0.99)
VaR999_2f = np.quantile(loss_rate_2f, 0.999)
ES99_2f   = loss_rate_2f[loss_rate_2f >= VaR99_2f].mean()
ES999_2f  = loss_rate_2f[loss_rate_2f >= VaR999_2f].mean()

p3_1f = gaussian_threshold_results_all[
    (gaussian_threshold_results_all["Portfolio"] == "Portfolio 3") &
    (gaussian_threshold_results_all["Method"]    == "Simulation")
].iloc[0]

print("\n" + "=" * 60)
print("Portfolio 3: One-Factor vs Two-Factor (Simulation)")
print("=" * 60)

comparison_p3 = pd.DataFrame({
    "Method":        ["One-Factor", "Two-Factor"],
    "Expected Loss": [round(p3_1f["Expected Loss"], 6), round(EL_2f,    6)],
    "VaR 99%":       [round(p3_1f["VaR 99%"],  6),     round(VaR99_2f,  6)],
    "VaR 99.9%":     [round(p3_1f["VaR 99.9%"], 6),    round(VaR999_2f, 6)],
    "ES 99%":        [round(p3_1f["ES 99%"],  6),       round(ES99_2f,   6)],
    "ES 99.9%":      [round(p3_1f["ES 99.9%"], 6),      round(ES999_2f,  6)],
})
print(comparison_p3.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# =============================================================================
# THIS PLOT IS USED: FIGURE 3
# =============================================================================
fig.suptitle("Figure 3: Portfolio 3: One-Factor vs Two-Factor Gaussian Threshold",
             fontsize=13, fontweight="bold")

p3_losses_1f = g_loss_plot_df[
    g_loss_plot_df["Portfolio"] == "Portfolio 3"]["Loss Rate"].values

ax1.hist(p3_losses_1f, bins=50, density=True, alpha=0.6, label="One-Factor")
ax1.hist(loss_rate_2f,  bins=50, density=True, alpha=0.6, label="Two-Factor")
ax1.axvline(VaR999_2f,           color="red",  linestyle="--",
            label=f"VaR 99.9% (2F) = {VaR999_2f:.4f}")
ax1.axvline(p3_1f["VaR 99.9%"], color="blue", linestyle="--",
            label=f"VaR 99.9% (1F) = {p3_1f['VaR 99.9%']:.4f}")
ax1.set_title("Loss Distribution"); ax1.set_xlabel("Loss Rate")
ax1.set_ylabel("Density"); ax1.legend(fontsize=8)

metrics_names = ["Expected Loss", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"]
vals_1f = [p3_1f["Expected Loss"], p3_1f["VaR 99%"], p3_1f["VaR 99.9%"],
           p3_1f["ES 99%"],        p3_1f["ES 99.9%"]]
vals_2f = [EL_2f, VaR99_2f, VaR999_2f, ES99_2f, ES999_2f]
x_pos = np.arange(len(metrics_names))
width = 0.35
ax2.bar(x_pos - width/2, vals_1f, width, label="One-Factor")
ax2.bar(x_pos + width/2, vals_2f, width, label="Two-Factor")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metrics_names, rotation=35, ha="right", fontsize=8)
ax2.set_title("Risk Metrics Comparison"); ax2.set_ylabel("Loss Rate")
ax2.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f"{PATH}/Figure_3_P3_OneF_vs_TwoF.png", dpi=200, bbox_inches="tight")
print("  -> Figure 3 saved to Figure_3_P3_OneF_vs_TwoF.png")
plt.show()


# =============================================================================
# FINAL COMPARISON: MODEL 1 (BETA-BINOMIAL) vs MODEL 2 (GAUSSIAN ONE-FACTOR)
# =============================================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON: BETA-BINOMIAL vs GAUSSIAN ONE-FACTOR")
print("=" * 60)

bb_comp           = beta_binomial_results_all.copy()
bb_comp["Model"]  = "Beta-Binomial"
g_comp            = gaussian_threshold_results_all.copy()
g_comp["Model"]   = "Gaussian One-Factor"

model_comparison = pd.concat([bb_comp, g_comp], ignore_index=True)
model_comparison = model_comparison[[
    "Model", "Portfolio", "Method",
    "Expected Loss", "VaR 99%", "VaR 99.9%", "ES 99%", "ES 99.9%"
]]
print(model_comparison.round(6).to_string(index=False))

analytical_comp = model_comparison[model_comparison["Method"] == "Analytical"].copy()
analytical_long = analytical_comp.melt(
    id_vars=["Model", "Portfolio"],
    value_vars=["Expected Loss", "VaR 99.9%", "ES 99.9%"],
    var_name="Metric", value_name="Loss Rate"
)

fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.suptitle("Model Comparison: Analytical Metrics (BB vs Gaussian)",
             fontsize=14, fontweight="bold")
for ax, metric in zip(axes, ["Expected Loss", "VaR 99.9%", "ES 99.9%"]):
    data = analytical_long[analytical_long["Metric"] == metric]
    sns.barplot(data=data, x="Portfolio", y="Loss Rate", hue="Model", ax=ax)
    ax.set_title(metric); ax.set_xlabel("Portfolio"); ax.set_ylabel("Loss Rate")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()