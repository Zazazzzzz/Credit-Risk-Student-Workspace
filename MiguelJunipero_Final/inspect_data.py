import os
import pandas as pd
# Data directory:
from MiguelJunipero_Final.main import DIR_DATA

def inspect_data():
# Check Portfolios and Entities data
    for filename in ["entities_info.csv", "portfolio_1.csv", "portfolio_2.csv", "portfolio_3.csv"]:
        path = os.path.join(DIR_DATA, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Get the number of rows for each unclean dataset
            print(f"\n{filename} (Rows: {len(df)})")
            # Get the number of missing values per column for each unclean dataset: entities_info, portfolio_1, portfolio_2, portfolio_3
            print(f"Missing values per column:")
            print(df.isna().sum())

            # Get all anomalies for the column "entity_code" - to help future standardisation
            if "entity_code" in df.columns:
                raw_codes = df["entity_code"].astype(str)
                broken_hyphen = raw_codes.str.match(r"^[AB][^-]").sum()
                capitalisation_and_spaces = (raw_codes != raw_codes.str.upper().str.strip()).sum()
                duplicates = raw_codes.str.upper().str.strip().duplicated().sum()

                print(f"Entity code anomalies:")
                print(f"      - Missing hyphens: {broken_hyphen}")
                print(f"      - Non-standardised capitalisation or trailing whitespaces: {capitalisation_and_spaces}")
                print(f"      - Duplicate records for same security: {duplicates}")
                
# Check stock_returns data
    stock_path = os.path.join(DIR_DATA, "stock_returns.csv")
    if os.path.exists(stock_path):
        stock_df = pd.read_csv(stock_path)
        # Get the number of rows for the unclean dataset "stock_returns"
        print(f"\nstock_returns.csv (Rows: {len(stock_df)})")

        # Get the number of missing values per column for the unclean dataset "stock_returns"
        print(f"Missing values per column:")
        print(stock_df.isna().sum())
        # Get all anomalies for the column "entity_code" (except for duplicates)
        raw_stock_codes = stock_df["entity_code"].astype(str)
        broken_hyphen_stock = raw_stock_codes.str.match(r"^[AB][^-]").sum()
        capitalisation_and_spaces_stock = (raw_stock_codes != raw_stock_codes.str.upper().str.strip()).sum()

        # Get the number of duplicated entries (rows): same date and same entity_code
        # Temporarily standardise entity_codes and dates to accurately check for duplicates
        std_codes = stock_df["entity_code"].astype(str).str.upper().str.strip()
        std_dates = pd.to_datetime(stock_df["date"], format="mixed")
        temp_df = pd.DataFrame({"date": std_dates, "entity_code": std_codes})
        duplicate_rows = temp_df.duplicated(subset=["date", "entity_code"]).sum()

        print(f"Entity code anomalies:")
        print(f"      - Missing hyphens: {broken_hyphen_stock}")
        print(f"      - Non-standardised capitalisation or trailing whitespaces: {capitalisation_and_spaces_stock}")
        print(f"      - Duplicate records (same date and same entity): {duplicate_rows}")

        # Date format samples to check for inconsistency
        print("Sample dates:")
        print(stock_df["date"].head(3).tolist())

# Check industry_index_returns data
    ind_path = os.path.join(DIR_DATA, "industry_index_returns.csv")
    if os.path.exists(ind_path):
        ind_df = pd.read_csv(ind_path)
        # Get the number of rows for the unclean dataset "industry_index_returns"
        print(f"\nindustry_index_returns.csv (Rows: {len(ind_df)})")
        # Get the number of missing values per column for the unclean dataset "industry_index_returns"
        print("Missing values in industry_index_returns.csv:")
        print(ind_df.isna().sum())
        # Get all unique formats for the column "industry"
        print("Unique industry labels found in industry_index_returns.csv:")
        print(ind_df["industry"].unique())
        # Date format samples to check for inconsistency
        print("Sample dates:")
        print(ind_df["date"].head(3).tolist())

# Check default_history_20y data
    def_path = os.path.join(DIR_DATA, "default_history_20y.csv")
    if os.path.exists(def_path):
        def_df = pd.read_csv(def_path)
        # Get the number of rows for the unclean dataset "default_history_20y"
        print(f"\ndefault_history_20y.csv (Rows: {len(def_df)})")
        # Get the number of missing values per column for the unclean dataset "default_history_20y"
        print("Missing values in default_history_20y.csv:")
        print(def_df.isna().sum())

    # Get all anomalies for the column "entity_code" (except for duplicates)
        if "entity_code" in def_df.columns:
            raw_codes_def = def_df["entity_code"].astype(str)
            broken_hyphen_def = raw_codes_def.str.match(r"^[AB][^-]").sum()
            capitalisation_and_spaces_def = (raw_codes_def != raw_codes_def.str.upper().str.strip()).sum()

            # Get the number of duplicated entries (rows): same date and same entity_code
            # Temporarily standardise entity_codes and dates to accurately check for duplicates
            std_codes_def = def_df["entity_code"].astype(str).str.upper().str.strip()
            std_dates_def = pd.to_datetime(def_df["date"], format="mixed")
            temp_df_def = pd.DataFrame({"date": std_dates_def, "entity_code": std_codes_def})
            duplicate_rows_def = temp_df_def.duplicated(subset=["date", "entity_code"]).sum()
            
            print(f"Entity code anomalies:")
            print(f"      - Missing hyphens: {broken_hyphen_def}")
            print(f"      - Non-standardised capitalisation or trailing whitespaces: {capitalisation_and_spaces_def}")
            print(f"      - Duplicate records for same security: {duplicate_rows_def}")

if __name__ == "__main__":
    inspect_data()