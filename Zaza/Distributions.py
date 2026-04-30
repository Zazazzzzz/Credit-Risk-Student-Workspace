# ---------------------------------------------------------------
# Script Name: Distributions and Risk Modeling Foundations
# Author: Hongyi Shen
# Description:
# This script introduces key probability distributions used in
# credit risk modeling and demonstrates their properties through
# simulation and visualization.
#
# Covered topics:
# - Discrete distributions: Bernoulli, Binomial, Poisson
# - Continuous distributions: Normal, Beta, Gamma
# - Relationship between distributions (e.g. Binomial → Poisson)
# - Concept of PDF vs PMF
# - Standardization of normal variables
# - Modeling heterogeneity (Beta) and intensity (Gamma)
#
# The script combines theoretical formulas with simulated data
# to build intuition for how distributions are used in practice.
# ---------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import gamma

# Reproducibility
np.random.seed(42)

# Parameters
p = 0.05          # default probability
m = 50            # number of obligors
n_sim = 5000      # number of simulations

############################################
# --- Bernoulli: one obligor ---
bernoulli_samples = np.random.binomial(n=1, p=p, size=n_sim)
print(bernoulli_samples.mean())
print(bernoulli_samples.var())

############################################
# --- Binomial: number of defaults ---

# The Binomial distribution models the number of successes
# in a fixed number of trials.
#
# K ~ Binomial(m, p)
#
# - m: number of trials (fixed)
# - p: probability of success in each trial
#
# Key properties:
# - E[K] = m * p
# - Var(K) = m * p * (1 - p)
#
# Interpretation:
# - There is a clear trial structure
# - Each trial has the same probability p
#
# Example:
# - Number of defaults among m firms
# - Number of heads in m coin flips

binomial_samples = np.random.binomial(n=m, p=p, size=n_sim)
binomial_samples.mean() # 50*0.05
binomial_samples.var() # 50*0.05*0.95

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# Bernoulli distribution
axes[0].hist(bernoulli_samples, bins=[-0.5, 0.5, 1.5], density=True, rwidth=0.6)
axes[0].set_xticks([0, 1])
axes[0].set_title("Bernoulli Distribution (One Obligor)")
axes[0].set_xlabel("Outcome")
axes[0].set_ylabel("Relative Frequency")
axes[0].grid(axis="y", alpha=0.3)

# Binomial distribution
axes[1].hist(binomial_samples, bins=range(0, max(binomial_samples)+2), density=True)
axes[1].set_title("Binomial Distribution (Number of Defaults)")
axes[1].set_xlabel("Number of Defaults")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

###########################################
# --- Poisson Distribution: number of defaults ---

# The Poisson distribution models the number of events occurring
# in a fixed interval (time, space, etc.).

# K ~ Poisson(lambda)
# - lambda (λ): expected number of events in the interval
# Key properties:
# - E[K] = λ
# - Var(K) = λ

# Interpretation:
# - Events occur randomly and independently
# - There is no fixed number of trials
# - We only care about how many events occur in a given period

# Example:
# - Number of defaults in a large portfolio over 1 year
# - Number of calls arriving in a minute

# Important:
# - Time is continuous, but the count K is discrete (0,1,2,...)



# Key Differences: Poisson vs Binomial
#
# 1. Structure:
#    - Binomial: fixed number of trials
#    - Poisson: no explicit trials, events occur over time
#
# 2. Parameters:
#    - Binomial: (m, p)
#    - Poisson: (λ)
#
# 3. Upper bound:
#    - Binomial: K <= m
#    - Poisson: no upper bound
#
# 4. Use case:
#    - Binomial: count successes out of known trials
#    - Poisson: count random events in an interval

# Poisson can be seen as a limit of the Binomial:
#
# - m → large
# - p → small
# - m * p = λ (constant)
# Then:
#     Binomial(m, p) ≈ Poisson(λ)
#
# Interpretation:
# - Many opportunities (large m)
# - Each event is rare (small p)
# - Only the total expected count matters (λ)
#
# Example in credit risk:
# - Large portfolio, small default probabilities
# → total defaults can be approximated by a Poisson distribution

# Binomial = count successes from fixed trials
# Poisson  = count random events over time

# Poisson
# define the distribution
lam = 4

# draw samples
samples = np.random.poisson(lam=lam, size=1000)

# histogram (empirical)
k_vals = np.arange(0, max(samples)+1)

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=np.arange(-0.5, max(samples)+1.5, 1),
         density=True, alpha=0.5, label="Simulated data")

# theoretical Probability Mass Function (PMF, for discrete random variable)
pmf = poisson.pmf(k_vals, mu=lam)

plt.plot(k_vals, pmf, marker='o', linestyle='-', label="Theoretical PMF")
plt.xlabel("Number of Events (k)")
plt.ylabel("Probability")
plt.title(f"Poisson Distribution (λ = {lam})")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# Comparision with small m
# Parameters
m = 20
p = 0.3              # try also p = 0.3 later
lam = m * p

# support (possible values)
x = np.arange(0, m+1)

# PMFs
binom_pmf = binom.pmf(x, m, p)
poisson_pmf = poisson.pmf(x, lam)

# Plot
plt.figure(figsize=(8, 5))

plt.bar(x - 0.2, binom_pmf, width=0.4, label="Binomial", alpha=0.7)
plt.bar(x + 0.2, poisson_pmf, width=0.4, label="Poisson", alpha=0.7)

plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.title(f"Binomial vs Poisson (m={m}, p={p}, λ={lam})")
plt.legend()
plt.grid(alpha=0.3)

plt.show()



# Connection: Binomial → Poisson (Credit Risk Interpretation)
#
# Consider a large portfolio with m obligors.
# Each obligor defaults independently with probability p.
#
# Binomial view:
# - We have m trials (one for each obligor)
# - Each trial has probability p of default
# - Total defaults:
#     K ~ Binomial(m, p)
#
# Poisson view:
# - Instead of tracking individual firms, we focus on total defaults
# - Over a fixed time period (e.g., one year), defaults occur randomly
# - The expected number of defaults is:
#     λ = m * p
#
# - Then:
#     K ~ Poisson(λ)
#
# Key idea:
# - When m is large and p is small (rare events),
#   the Binomial distribution is well approximated by a Poisson distribution:
#
#     Binomial(m, p) ≈ Poisson(λ = m * p)
#
# Interpretation:
# - Binomial: structured view (fixed number of firms/trials)
# - Poisson: aggregate view (random number of default events over time)
#
# In credit risk:
# - Large portfolio + small default probabilities
# → Poisson is often a convenient approximation for total defaults



############################################
# --- Beta Distribution: random probabilities ---

# The Beta distribution is used to model random variables that lie between 0 and 1.
# This makes it especially useful for modeling probabilities, such as default probabilities (p_i)

# Definition:
# If X ~ Beta(alpha, beta), then X takes values in [0, 1].

# Key idea:
# - alpha controls how much weight is placed toward 1 (higher values)
# - beta  controls how much weight is placed toward 0 (lower values)

# Mean and variance:
# E[X] = alpha / (alpha + beta)
# Var(X) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

# Shape intuition:
# - alpha = beta = 1  -> Uniform distribution (all values equally likely)
# - alpha > beta      -> Skewed toward 1 (higher probabilities more likely)
# - alpha < beta      -> Skewed toward 0 (lower probabilities more likely)
# - alpha, beta large -> Concentrated (low variance, more certainty)
# - alpha, beta < 1   -> U-shaped (values near 0 and 1 more likely)

# Why useful in modeling:
# Instead of assuming a fixed probability p for all firms,
# we can model heterogeneity:
#
#     p_i ~ Beta(alpha, beta)
#
# This allows different firms to have different default probabilities.

# Connection to Bernoulli:
# If:
#     Y_i | p_i ~ Bernoulli(p_i)
# and:
#     p_i ~ Beta(alpha, beta)
#
# Then we introduce additional randomness (mixture model),
# which increases variability and can create dependence across observations.
# The Beta distribution models random probabilities on [0,1] with flexible shapes.

x = np.linspace(0, 1, 1000)

params = [
    (1, 1),
    (2, 5),
    (5, 2),
    (0.5, 0.5)
]

plt.figure(figsize=(8, 5))

for a, b in params:
    plt.plot(x, beta.pdf(x, a, b), label=f"alpha={a}, beta={b}")

plt.title("Beta Distribution Shapes")
plt.xlabel("x (probability)")
plt.ylabel("Density")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# Gamma Distribution
#
# The Gamma distribution models a positive continuous variable:
#     X > 0
#
# It is often used for quantities like:
# - rates / intensities (e.g. default intensity λ)
# - waiting times
# - magnitudes of positive variables
#
# Parameters
#
# X ~ Gamma(shape=k, scale=theta)
#
# - shape (k):
#     controls the form of the distribution
#     small k   -> highly right-skewed (mass near 0)
#     large k   -> more symmetric (bell-shaped)
#
# - scale (theta):
#     stretches the distribution horizontally
#     larger theta -> more spread / heavier tail
#
# - defined only on positive values
# - very flexible shape:
#     can be skewed or close to symmetric

# Connection to Poisson modeling
#
# Instead of assuming a fixed rate:
#     K | λ ~ Poisson(λ)
#
# we allow:
#     λ ~ Gamma(...)
#
# Interpretation:
# - λ varies across states of the world
# - Gamma captures this uncertainty

# The Gamma distribution models uncertainty in a positive rate or intensity
# and is a natural choice when extending Poisson models.


# Gamma
x = np.linspace(0, 20, 1000)

# Different choices of shape a and scale theta
params = [
    (1, 2),    # very right-skewed
    (2, 2),
    (5, 1),
    (9, 0.5)
]

plt.figure(figsize=(8, 5))

for a, theta in params:
    pdf = gamma.pdf(x, a=a, scale=theta)
    plt.plot(x, pdf, label=f"shape={a}, scale={theta}")

plt.title("Gamma Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.show()


############################################
# --- Normal: asset return ---
mu, sigma = 0, 1

# Step 1: simulate data (random draws)
samples = np.random.normal(mu, sigma, size=1000)

# Plot
plt.figure(figsize=(8, 5))

# Histogram (empirical)
plt.hist(samples, bins=30, density=True, alpha=0.5, label="Simulated Data")
plt.title("Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()

# Step 2: grid for theoretical PDF
x = np.linspace(-4, 4, 1000)
pdf = norm.pdf(x, mu, sigma)

# PDF: theoretical density of the normal distribution
# This gives the "height" of the curve at each x
# A PDF assigns density to values; probabilities are obtained as areas under the curve.

# A probability density function (PDF) describes how probability is distributed
# over the values of a continuous random variable.

# Formal definition:
# A function f(x) is a PDF of a random variable X if:
# 1. Non-negativity:
#    f(x) >= 0 for all x
#
# 2. Total probability equals 1:
#    integral from -infinity to infinity of f(x) dx = 1
#
# 3. Probabilities are given by areas:
#    For any interval [a, b]:
#    P(a <= X <= b) = integral from a to b of f(x) dx

# Key interpretation:
# - f(x) is NOT a probability
# - It is a density (height of the curve). The height of the curve indicates how concentrated probability is around that value.
# - Density × Width = Probability
# - Probability comes from the area under the curve
# - For continuous variables: P(X = x) = 0

# Example: Normal distribution
# If X ~ N(mu, sigma^2), then the PDF is:
# f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))

# Intuition:
# - The PDF can be seen as the smooth limit of a histogram
# - Histogram (data) approximates the PDF (theoretical distribution)

# Plot
plt.figure(figsize=(8, 5))

# Histogram (empirical)
plt.hist(samples, bins=30, density=True, alpha=0.5, label="Simulated Data")

# Theoretical PDF
plt.plot(x, pdf, label="Normal PDF")

plt.title("From Samples to Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()


