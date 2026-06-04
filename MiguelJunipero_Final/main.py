import os
import pandas as pd
import numpy as np
from data_cleaning import clean_and_merge
from parameter_estimation import estimate_risk_parameters
from simulation import credit_simulation, risk_measures

# Data directory:
DIR_DATA = "../Final Assignment/Data"
# Results directory
DIR_RESULTS = "results"

def main():
# Verify that the raw data exists
    if not os.path.exists(DIR_DATA):
        print(f"Error: {DIR_DATA} is not a directory.")
        return
# Create results folder
    if not os.path.exists(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)
        print(f"Results directory {DIR_RESULTS} created.")
# Call clean and merge function
    try:
        cleaned_p1, cleaned_p2, cleaned_p3, stocks, inds_new, defaults = clean_and_merge(DIR_DATA)
    except Exception as e:
        print(f"Data Cleaning Error: {e}")
        return

# Call parameter estimation function
    try:
        param_p1, param_p2, global_pd, rho, transition_matrix = estimate_risk_parameters(defaults, cleaned_p1, cleaned_p2)
    except Exception as e:
        print(f"Parameter Estimation Error: {e}")

# Call simulation and compute risk measures
    losses_p1_A, losses_p1_B = credit_simulation(param_p1, rho, num_scenarios=10000)
    metrics_p1_A = risk_measures(losses_p1_A, alpha=0.99)
    metrics_p1_B = risk_measures(losses_p1_B, alpha=0.99)

    losses_p2_A, losses_p2_B = credit_simulation(param_p2, rho, num_scenarios=10000)
    metrics_p2_A = risk_measures(losses_p2_A, alpha=0.99)
    metrics_p2_B = risk_measures(losses_p2_B, alpha=0.99)

# Print results:
    print(f"PORTFOLIO 1:")
    print(f"  Model A (One-Factor F) ->  EL: {metrics_p1_A['EL']:11.2f} | 99% VaR: {metrics_p1_A['VaR']:11.2f} | 99% ES: {metrics_p1_A['ES']:11.2f}")
    print(f"  Model B (Independent)  ->  EL: {metrics_p1_B['EL']:11.2f} | 99% VaR: {metrics_p1_B['VaR']:11.2f} | 99% ES: {metrics_p1_B['ES']:11.2f}")
    print("--------------------------------------------------------------------------")
    print(" PORTFOLIO 2:")
    print(f"  Model A (One-Factor F) ->  EL: {metrics_p2_A['EL']:11.2f} | 99% VaR: {metrics_p2_A['VaR']:11.2f} | 99% ES: {metrics_p2_A['ES']:11.2f}")
    print(f"  Model B (Independent)  ->  EL: {metrics_p2_B['EL']:11.2f} | 99% VaR: {metrics_p2_B['VaR']:11.2f} | 99% ES: {metrics_p2_B['ES']:11.2f}")

# Save results
    np.savetxt(os.path.join(DIR_RESULTS, "losses_p1_model_A.txt"), losses_p1_A)
    np.savetxt(os.path.join(DIR_RESULTS, "losses_p1_model_B.txt"), losses_p1_B)
    np.savetxt(os.path.join(DIR_RESULTS, "losses_p2_model_A.txt"), losses_p2_A)
    np.savetxt(os.path.join(DIR_RESULTS, "losses_p2_model_B.txt"), losses_p2_B)

# Save cleaned data in the directory
    param_p1.to_csv(os.path.join(DIR_RESULTS, "cleaned_portfolio1.csv"), index=False)
    param_p2.to_csv(os.path.join(DIR_RESULTS, "cleaned_portfolio2.csv"), index=False)
    stocks.to_csv(os.path.join(DIR_RESULTS, "cleaned_stock_returns.csv"), index=False)
    defaults.to_csv(os.path.join(DIR_RESULTS, "cleaned_default_history.csv"), index=False)
    inds_new.to_csv(os.path.join(DIR_RESULTS, "cleaned_industry_returns.csv"), index=False)

if __name__ == "__main__":
    main()
