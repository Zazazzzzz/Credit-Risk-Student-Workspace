import pandas as pd
import numpy as np
from scipy.stats import norm

def estimate_risk_parameters(defaults, p1, p2):
    """
    
    """

    # Get global baseline pd from default history
    global_pd = float(defaults["default_event"].sum() / len(defaults))
    if global_pd == 0:
        global_pd = 0.02

    # Map binary states to annual timelines
    defaults["year"] = defaults["date"].dt.year
    annual_rates = defaults.groupby("year")["default_event"].agg(["count", "sum"])
    annual_rates["rate"] = annual_rates["sum"] / annual_rates["count"]

    # Variance across years
    annual_variance = annual_rates["rate"].var()

    # Calculate systemic asset correlation (Rho)
    rho = annual_variance / (global_pd * (1.0 - global_pd))

    # 1-year transition matrix
    transition_matrix = np.array([
        [1.0 - global_pd, global_pd],
        [0.0, 1.0]
    ])

    # Assure structural consistency
    portfolios_out = []
    for i, p_df in enumerate([p1, p2], start=1):
        p_clean = p_df.copy()

        p_clean["PD"] = p_clean["base_pd_hint"].astype(float)
        p_clean["EAD"] = p_clean["EAD"].astype(float)
        p_clean["lgd"] = p_clean["lgd"].astype(float)

        clipped_pds = np.clip(p_clean["PD"].values, 0.0001, 0.9999)
        p_clean["default_threshold"] = norm.ppf(clipped_pds)

        portfolios_out.append(p_clean)

    return portfolios_out[0], portfolios_out[1], global_pd, rho, transition_matrix
