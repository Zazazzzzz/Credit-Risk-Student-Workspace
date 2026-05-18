# ---------------------------------------------------------------
# Script Name: From Default Correlation to Asset Correlation
# Author: Hongyi Shen
#
# Description:
# This script converts a target default correlation into the
# corresponding asset correlation under a homogeneous Gaussian
# one-factor threshold model.
#
# In the model, each obligor's latent credit variable is given by
#
#     X_i = rho F + sqrt(1 - rho^2) epsilon_i,
#
# where F is the common systematic factor and epsilon_i is the
# idiosyncratic risk component.
#
# The factor loading is rho, while the asset correlation between
# two obligors is rho^2. Default occurs when the latent variable
# falls below the default threshold Phi^{-1}(p).
#
# The script numerically solves for the factor loading rho that
# matches a given target default correlation, and then reports the
# corresponding asset correlation rho^2.
# ---------------------------------------------------------------


from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq

# --------------------------------------------------
# Inputs
# --------------------------------------------------
p = 0.01          # unconditional default probability
rho_y = 0.005     # target default correlation, e.g. 0.5%

# --------------------------------------------------
# Default threshold
# --------------------------------------------------
d = norm.ppf(p)

# --------------------------------------------------
# Function: default correlation implied by factor loading rho
# --------------------------------------------------
def implied_default_correlation(rho, p):
    """
    Gaussian one-factor model:

        X_i = rho F + sqrt(1-rho^2) epsilon_i

    The asset correlation is rho^2.

    For homogeneous obligors with default probability p,
    default occurs if X_i <= Phi^{-1}(p).

    This function returns Corr(Y_i, Y_j).
    """

    d = norm.ppf(p)

    asset_corr = rho ** 2

    cov_matrix = [
        [1.0, asset_corr],
        [asset_corr, 1.0]
    ]

    joint_default_prob = multivariate_normal.cdf(
        [d, d],
        mean=[0, 0],
        cov=cov_matrix
    )

    default_corr = (joint_default_prob - p ** 2) / (p * (1 - p))

    return default_corr

# --------------------------------------------------
# Solve for rho
# --------------------------------------------------
def objective(rho):
    return implied_default_correlation(rho, p) - rho_y

rho_solution = brentq(objective, 0.0, 0.999999)

asset_corr_solution = rho_solution ** 2

print(f"Target default correlation: {rho_y:.4%}")
print(f"Factor loading rho: {rho_solution:.4f}")
print(f"Asset correlation rho^2: {asset_corr_solution:.4%}")
print(f"Check implied default correlation: {implied_default_correlation(rho_solution, p):.4%}")