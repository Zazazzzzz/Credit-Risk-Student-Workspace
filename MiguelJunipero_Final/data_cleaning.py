import os
import pandas as pd

def clean_industries(text):
    """
    Standardises all types of string variations on the column industry into strict "INDUSTRY A" or "INDUSTRY B"
    :param text:
    :return:
    """
    if pd.isna(text):
        return "UNKNOWN"
    s = str(text).upper().replace(".", "").replace(" ", "").strip()
    if "INDA" in s or "INDUSTRYA" in s:
        return "INDUSTRY_A"
    if "INDB" in s or "INDUSTRYB" in s:
        return "INDUSTRY_B"
    return s


def clean_entity_codes(text):
    """
    Standardises entity codes into the format: "A-D1XPP4"
    :param text:
    :return:
    """
    if pd.isna(text):
        return "UNKNOWN"
    s = str(text).upper().strip()
    # If it does not have a hyphen as the second character, insert it
    if len(s) == 7 and s[0] in ["A", "B"] and s[1] != "-":
        s = f"{s[0]}-{s[1:]}"
    return s

def clean_portfolio(dir_data, file_name):
    """
    Loads, standardises, handles missing EAD, and aggregates each portfolio's file
    :param dir_data, file_name:
    :return:
    """
    path = os.path.join(dir_data, file_name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["entity_code"] = df["entity_code"].apply(clean_entity_codes)

    # Fill missing EADs with portfolio's median
    df["EAD"] = df["EAD"].fillna(df["EAD"].median())

    # Remove duplicates
    df = df.drop_duplicates(subset=["entity_code"], keep="first")
    return df

def clean_and_merge(dir_data):
    """

    :param dir_data:
    :return:
    """
    # Clean "entities_info" file
    entities_path = os.path.join(dir_data, "entities_info.csv")
    entities = pd.read_csv(entities_path)
    entities["entity_code"] = entities["entity_code"].apply(clean_entity_codes)
    entities["industry"] = entities["industry"].apply(clean_industries)

    # Drop true duplicates, if they exist
    entities = entities.drop_duplicates(subset=["entity_code"], keep="first")

    # Fill missing LGDs with the industry's median
    entities["lgd"] = entities.groupby("industry")["lgd"].transform(lambda x: x.fillna(x.median()))

    # Clean "default_history_20y" file
    defaults_path = os.path.join(dir_data, "default_history_20y.csv")
    defaults = pd.read_csv(defaults_path)
    defaults["entity_code"] = defaults["entity_code"].apply(clean_entity_codes)
    # Fix mixed date formats
    defaults["date"] = pd.to_datetime(defaults["date"], format="mixed")
    # Fill missing values in column "default_event" - having a missing value is to be assumed it is 0
    defaults["default_event"] = defaults["default_event"].fillna(0).astype(int)
    # Drop duplicates
    defaults = defaults.drop_duplicates(subset=["date", "entity_code"], keep="first")

    # Map industries to default rates
    ent_ind_map = entities.set_index("entity_code")["industry"].to_dict()
    defaults["industry"] = defaults["entity_code"].map(ent_ind_map).apply(clean_industries)

    # Map industries to defaults file
    ind_pd_rates = defaults.groupby("industry")["default_event"].mean().to_dict()
    global_pd = defaults["default_event"].mean()

    # Missing pd values
    entities["base_pd_hint"] = entities["base_pd_hint"].fillna(entities["industry"].map(ind_pd_rates).fillna(global_pd))

    # Clean "stock_returns" file
    stocks_path = os.path.join(dir_data, "stock_returns.csv")
    stocks = pd.read_csv(stocks_path)
    stocks["entity_code"] = stocks["entity_code"].apply(clean_entity_codes)
    stocks["date"] = pd.to_datetime(stocks["date"], format="mixed")
    # Drop duplicates
    stocks = stocks.drop_duplicates(subset=["date", "entity_code"], keep="first")
    # Ordering before applying the forward fill
    stocks = stocks.sort_values(["entity_code", "date"])
    # Forward fill for missing values - under the assumption that if a day is missing, the stock did not change value (e.g., closed exchange)
    stocks["stock_return"] = stocks.groupby("entity_code")["stock_return"].transform(lambda x: x.ffill().fillna(0))

    # Clean "industry_index_returns" file
    inds_path = os.path.join(dir_data, "industry_index_returns.csv")
    inds = pd.read_csv(inds_path)
    inds["industry"] = inds["industry"].apply(clean_industries)
    inds["date"] = pd.to_datetime(inds["date"], format="mixed")
    # Drop duplicates
    inds = inds.drop_duplicates(subset=["date", "industry"], keep="first")
    # Transform "industry_index_returns" file to have "INDUSTRY_A" and "INDUSTRY_B" as columns, and returns per row
    inds_new = inds.pivot(index="date", columns="industry", values="index_return").reset_index()
    inds_new = inds_new.sort_values("date").ffill().fillna(0)

    # Extract individual portfolios
    p1 = clean_portfolio(dir_data, "portfolio_1.csv")
    p2 = clean_portfolio(dir_data, "portfolio_2.csv")
    p3 = clean_portfolio(dir_data, "portfolio_3.csv")

    # Merge "entities_info" and portfolio files
    features = ["entity_code", "industry", "region", "lgd", "base_pd_hint"]
    cleaned_p1 = pd.merge(p1, entities[features], on="entity_code", how="left")
    cleaned_p2 = pd.merge(p2, entities[features], on="entity_code", how="left")
    cleaned_p3 = pd.merge(p3, entities[features], on="entity_code", how="left")

    # Safety parameter (Basel regulatory assumption of a standard 40% LGD)
    cleaned_p1["lgd"] = cleaned_p1["lgd"].fillna(0.40)
    cleaned_p2["lgd"] = cleaned_p2["lgd"].fillna(0.40)
    cleaned_p3["lgd"] = cleaned_p3["lgd"].fillna(0.40)

    return cleaned_p1, cleaned_p2, cleaned_p3, stocks, inds_new, defaults
