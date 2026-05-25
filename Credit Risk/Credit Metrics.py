# ============================================================
# Task: Credit Portfolio Risk Measurement
# Simulation-Based and Analytical VaR/ES under a One-Factor
# Gaussian Threshold Model
# ============================================================
#
# Objective
# ---------
# We analyze a credit portfolio consisting of 10 obligors.
# Each obligor has:
#
#   1. an exposure weight,
#   2. a loss given default (LGD),
#   3. an asset correlation parameter rho,
#   4. a current credit rating.
#
# The portfolio is analyzed under a one-factor Gaussian threshold model:
#
#   X_i = rho_i * F + sqrt(1 - rho_i^2) * epsilon_i
#
# where:
#
#   F         = common systematic risk factor,
#   epsilon_i = idiosyncratic risk factor of obligor i,
#   rho_i     = sensitivity of obligor i to the systematic factor.
#
# Obligor i defaults if:
#
#   X_i <= d_i
#
# where:
#
#   d_i = Phi^{-1}(p_i)
#
# and p_i is the one-year default probability implied by the current rating.
#
# The portfolio loss in one scenario is:
#
#   L = sum_i exposure_i * LGD_i * Y_i
#
# where Y_i is the default indicator.
#
# The analysis has four parts:
#
#   1. Simulate correlated defaults and the portfolio loss distribution.
#   2. Compute VaR and ES from the simulated loss distribution.
#   3. Compute the analytical conditional expected loss function l(F).
#   4. Compute analytical VaR and ES from l(F).
#
# Important distinction
# ---------------------
# The simulation method includes both:
#
#   - systematic risk through F,
#   - idiosyncratic risk through epsilon_i.
#
# The analytical method calculates:
#
#   l(F) = sum_i exposure_i * LGD_i * p_i(F)
#
# where p_i(F) is the conditional default probability given the systematic
# factor. Therefore, the analytical method smooths out idiosyncratic default
# randomness and focuses on the systematic component of portfolio risk.
#
# ============================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# 1. Input portfolio data
# ============================================================

portfolio = pd.DataFrame({
    "obligor": ["CO1", "CO2", "CO3", "CO4", "CO5", "CO6", "CO7", "CO8", "CO9", "CO10"],
    "exposure": [0.10, 0.50, 0.01, 0.02, 0.05, 0.03, 0.07, 0.09, 0.06, 0.07],
    "LGD": [0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
    "rho": [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7], # or  "rho": [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7], "rho": [0.30, 0.25, 0.25, 0.30, 0.27, 0.27, 0.28, 0.29, 0.27, 0.27]
    "rating": ["AAA", "A", "BBB", "BB", "BBB", "AA", "A", "A", "A", "A"]  # or "rating": ["AAA", "A", "BBB", "BB", "BBB", "AA", "A", "A", "A", "A"]
})


# ============================================================
# 2. Assign one-year default probabilities by rating
# ============================================================
#
# The following default probabilities are taken from the one-year
# rating transition matrix.
#a
# They are transition probabilities into the default state D.
# The values are written in decimal form, not percentage form.
#
# Example:
#   0.06% = 0.0006
#
# For AAA, the transition probability to default is 0.00% in the
# transition matrix. Numerically, we replace 0 by a very small value
# so that Phi^{-1}(p) is finite.
#
# This avoids -infinity when calculating default thresholds.

rating_pd = {
    "AAA": 0.000001,  # approximately zero, adjusted for numerical stability
    "AA": 0.000001,  # 0.00% in the matrix
    "A": 0.0001,     # 0.01%
    "BBB": 0.0006,   # 0.06%
    "BB": 0.0040,     # 0.40%
    "B": 0.0238
}

portfolio["pd"] = portfolio["rating"].map(rating_pd)


# ============================================================
# 3. Convert default probabilities into default thresholds
# ============================================================
#
# In the Gaussian threshold model:
#
#   p_i = P(X_i <= d_i)
#
# Since X_i is standard normally distributed:
#
#   d_i = Phi^{-1}(p_i)

portfolio["threshold"] = norm.ppf(portfolio["pd"])

print("Portfolio with default probabilities and thresholds:")
print(portfolio)


# ============================================================
# 4. Simulation of correlated defaults
# ============================================================
#
# In each simulation:
#
#   1. Draw one common factor F.
#   2. Draw one idiosyncratic shock epsilon_i for each obligor.
#   3. Compute X_i.
#   4. Determine whether each obligor defaults.
#   5. Compute total portfolio loss.
#
# The simulated loss is:
#
#   L = sum_i exposure_i * LGD_i * Y_i

np.random.seed(42)

n_sim = 10000
m = len(portfolio)

exposure = portfolio["exposure"].values
LGD = portfolio["LGD"].values
rho = portfolio["rho"].values
threshold = portfolio["threshold"].values

# Common systematic factor: one draw per simulation
F = np.random.normal(0, 1, size=n_sim)

# Idiosyncratic shocks: one draw for each obligor in each simulation
epsilon = np.random.normal(0, 1, size=(n_sim, m))

# Latent creditworthiness variables
X = rho * F[:, None] + np.sqrt(1 - rho**2) * epsilon

# Default indicators
defaults = (X <= threshold).astype(int)

# Portfolio loss in each simulation
loss_sim = np.sum(exposure * LGD * defaults, axis=1)


# ============================================================
# 5. Simulation-based VaR and ES
# ============================================================

def empirical_var(losses, alpha):
    """
    Empirical Value at Risk at confidence level alpha.
    """
    return np.quantile(losses, alpha)


def empirical_es(losses, alpha):
    """
    Empirical Expected Shortfall at confidence level alpha.

    This is calculated as the average loss conditional on the loss
    being greater than or equal to the empirical VaR.
    """
    var_alpha = empirical_var(losses, alpha)
    return losses[losses >= var_alpha].mean()


EL_sim = loss_sim.mean()
VaR_999_sim = empirical_var(loss_sim, 0.999)
VaR_99_sim = empirical_var(loss_sim, 0.99)
ES_999_sim = empirical_es(loss_sim, 0.999)
ES_99_sim = empirical_es(loss_sim, 0.99)

print("\nSimulation-based risk measures:")
print(f"Expected Loss: {EL_sim:.6f}")
print(f"VaR 99.9%:       {VaR_999_sim:.6f}")
print(f"VaR 99%:       {VaR_99_sim:.6f}")
print(f"ES 99.9%:        {ES_999_sim:.6f}")
print(f"ES 99%:        {ES_99_sim:.6f}")


# ============================================================
# 6. Plot simulated loss distribution
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(loss_sim, bins=30, density=True, edgecolor="black")

plt.axvline(EL_sim, linestyle="--", label=f"EL = {EL_sim:.4f}")
plt.axvline(VaR_999_sim, linestyle="--", label=f"VaR 99.9% = {VaR_999_sim:.4f}")
plt.axvline(VaR_99_sim, linestyle="--", label=f"VaR 99% = {VaR_99_sim:.4f}")

plt.xlabel("Portfolio loss")
plt.ylabel("Density")
plt.title("Simulated Portfolio Loss Distribution")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. Analytical conditional expected loss function
# ============================================================
#
# Conditional on the systematic factor F = f, the conditional default
# probability of obligor i is:
#
#   p_i(f) = Phi((d_i - rho_i * f) / sqrt(1 - rho_i^2))
#
# where:
#
#   d_i = Phi^{-1}(p_i)
#
# The analytical conditional expected portfolio loss is:
#
#   l(f) = sum_i exposure_i * LGD_i * p_i(f)
#
# This gives a smooth loss function driven only by the systematic factor.

def conditional_pd(f):
    """
    Conditional default probabilities for all obligors given factor value f.
    """
    return norm.cdf((threshold - rho * f) / np.sqrt(1 - rho**2))


def analytical_loss_function(f):
    """
    Analytical conditional expected portfolio loss l(f).
    """
    return np.sum(exposure * LGD * conditional_pd(f))


# ============================================================
# 8. Analytical VaR
# ============================================================
#
# Because the loss function l(F) is decreasing in F, high portfolio losses
# correspond to bad systematic states, i.e. low values of F.
#
# For confidence level alpha:
#
#   VaR_alpha = l(Phi^{-1}(1 - alpha))

def analytical_var(alpha):
    """
    Analytical VaR based on the conditional expected loss function.
    """
    f_alpha = norm.ppf(1 - alpha)
    return analytical_loss_function(f_alpha)


VaR_999_ana = analytical_var(0.999)
VaR_99_ana = analytical_var(0.99)


# ============================================================
# 9. Analytical ES
# ============================================================
#
# Analytical ES averages l(F) over the worst 1-alpha part of the
# systematic factor distribution:
#
#   ES_alpha = E[l(F) | F <= Phi^{-1}(1-alpha)]
#
# Numerically:
#
#   ES_alpha = 1/(1-alpha) * integral_0^{1-alpha} l(Phi^{-1}(u)) du
#
# We approximate the integral using a fine grid over u.

def analytical_es(alpha, n_grid=100_000):
    """
    Analytical Expected Shortfall calculated by numerical integration
    over the lower tail of the systematic factor.
    """
    u_grid = np.linspace(1e-10, 1 - alpha, n_grid)
    f_grid = norm.ppf(u_grid)
    loss_grid = np.array([analytical_loss_function(f) for f in f_grid])
    return loss_grid.mean()


ES_999_ana = analytical_es(0.999)
ES_99_ana = analytical_es(0.99)

# Analytical expected loss:
# Since the unconditional default probability is p_i, analytical EL is:
#
#   EL = sum_i exposure_i * LGD_i * p_i

EL_ana = np.sum(exposure * LGD * portfolio["pd"].values)


print("\nAnalytical risk measures:")
print(f"Expected Loss: {EL_ana:.6f}")
print(f"VaR 99.9%:       {VaR_999_ana:.6f}")
print(f"VaR 99%:       {VaR_99_ana:.6f}")
print(f"ES 99.9%:        {ES_999_ana:.6f}")
print(f"ES 99%:        {ES_99_ana:.6f}")


# ============================================================
# 10. Compare simulation-based and analytical results
# ============================================================

results = pd.DataFrame({
    "Method": ["Simulation", "Analytical"],
    "Expected Loss": [EL_sim, EL_ana],
    "VaR 99.9%": [VaR_999_sim, VaR_999_ana],
    "VaR 99%": [VaR_99_sim, VaR_99_ana],
    "ES 99.9%": [ES_999_sim, ES_999_ana],
    "ES 99%": [ES_99_sim, ES_99_ana]
})

print("\nComparison of risk measures:")
print(results)


# ============================================================
# 11. Plot analytical loss function l(F)
# ============================================================

f_values = np.linspace(-4, 4, 500)
l_values = np.array([analytical_loss_function(f) for f in f_values])

plt.figure(figsize=(10, 6))
plt.plot(f_values, l_values)

plt.axvline(norm.ppf(0.05), linestyle="--", label="F 5% quantile")
plt.axvline(norm.ppf(0.01), linestyle="--", label="F 1% quantile")

plt.xlabel("Systematic factor F")
plt.ylabel("Conditional expected portfolio loss l(F)")
plt.title("Analytical Conditional Expected Loss Function")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 12. Interpretation guide
# ============================================================
#
# After running the code, discuss:
#
# 1. Why the simulated loss distribution is discrete.
#    The portfolio has only 10 obligors, so losses occur in jumps:
#
#       exposure_i * LGD_i
#
# 2. Why the analytical loss function is smooth.
#    It uses conditional default probabilities p_i(F), not realized
#    default indicators Y_i.
#
# 3. Why simulated VaR and analytical VaR may differ.
#    Simulation includes idiosyncratic default randomness.
#    The analytical method averages over idiosyncratic risk conditional
#    on the systematic factor.
#
# 4. Which obligor is most important.
#    CO2 has the largest exposure, 0.50, so even though it is A-rated,
#    it can dominate portfolio risk.
#
# 5. Why BBB is riskier than AAA.
#    BBB has a larger transition probability to default than AAA.
#    Therefore, the BBB default threshold is less extreme than the
#    AAA default threshold.
#
# ============================================================


# ============================================================
# Results
#
# high rho, B
# Comparison of risk measures:
#        Method  Expected Loss  VaR 99.9%   VaR 99%  ES 99.9%    ES 99%
# 0  Simulation       0.004632    0.20000  0.200000  0.200192  0.200192
# 1  Analytical       0.004818    0.09022  0.047202  0.108629  0.065633
#
# low rho, B
# Comparison of risk measures:
#        Method  Expected Loss  VaR 99.9%   VaR 99%  ES 99.9%    ES 99%
# 0  Simulation       0.004451   0.200000  0.200000  0.200073  0.200073
# 1  Analytical       0.004818   0.021804  0.015162  0.024806  0.018038
#
# high rho, A
# Comparison of risk measures:
#        Method  Expected Loss  VaR 99.9%   VaR 99%  ES 99.9%    ES 99%
# 0  Simulation       0.000072   0.008012  0.000000   0.04120  0.000072
# 1  Analytical       0.000078   0.004787  0.001117   0.00871  0.002670
#
# low rho, A
# Comparison of risk measures:
#        Method  Expected Loss  VaR 99.9%   VaR 99%  ES 99.9%    ES 99%
# 0  Simulation       0.000071   0.008000  0.000000  0.014933  0.000071  
# 1  Analytical       0.000078   0.000835  0.000442  0.001063  0.000609
# ============================================================

# B rating versus A rating
# Role of rho: it mainly affects tail risk, not expected loss
# Why analytical VaR is much lower than simulated VaR in the B case
# A-rating cases: risk is much lower, but rho still matters in the tail
# 0.000071  # This is mechanically caused by the ES definition in the code, not necessarily the best empirical ES estimate.