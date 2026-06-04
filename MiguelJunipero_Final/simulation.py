import numpy as np

def credit_simulation(portfolio_df, rho, num_scenarios=10000, random_seed=42):
    """

    """
    np.random.seed(random_seed)
    num_assets = len(portfolio_df)

    # Transform into vectors
    ead = portfolio_df["EAD"].values
    lgd = portfolio_df["lgd"].values
    pd_vec = portfolio_df["PD"].values
    thresholds = portfolio_df["default_threshold"].values

    # Model A: One-factor model simulation
    # Systematic factors:
    F = np.random.normal(0, 1 , size=(num_scenarios, 1))

    # Epsilon:
    epsilon = np.random.normal(0, 1, size=(num_scenarios, num_assets))

    # Rho
    rho_i = np.sqrt(rho)

    # Xi = rho_i * F + sqrt(1-rho_i^2) * epsilon
    Xi = (rho_i * F) + (np.sqrt(1.0 - rho) * epsilon)

    # Default triggered if Xi < threshold
    defaults_model_a = Xi <= thresholds
    losses_model_a = np.sum(defaults_model_a * (ead * lgd), axis=1)

    # Model B: Independent bernoulli simulation (rho = 0, no common factor F)
    # Draw independent uniform variables
    uniform_draws = np.random.uniform(0, 1, size=(num_scenarios, num_assets))
    defaults_model_b = uniform_draws <= pd_vec
    losses_model_b = np.sum(defaults_model_b * (ead * lgd), axis=1)

    return losses_model_a, losses_model_b

def risk_measures(losses, alpha=0.99):
    """

    """
    el = np.mean(losses)
    var = np.percentile(losses, alpha * 100)

    tail_losses = losses[losses >= var]
    es = np.mean(tail_losses) if len(tail_losses) > 0 else var

    return {"EL": el, "VaR": var, "ES": es}



