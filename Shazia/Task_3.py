# ---------------------------------------------------------------
# Script Name: Task 3 - Dependency Structure of Obligors
# Author: Shazia Ishaq
# Description: Understanding how two companies can default
#              together and measuring their relationship
#

# Company 1 (Y1) and Company 2 (Y2) can either default (1)
# or survive (0) during a given year.
# We are given the probabilities of all possible combinations.
# ---------------------------------------------------------------

import numpy as np
from scipy.stats import norm

# ── Step 1: The Joint Probability Table ───────────────────────
# These are the probabilities of all possible combinations
# of default and survival for the two companies.

# like a weather forecast:
# P(rain AND cold) = 0.05
# P(rain AND warm) = 0.03


p_00 = 0.85  # both companies survive (most common scenario)
p_10 = 0.07  # company 1 defaults, company 2 survives
p_01 = 0.03  # company 1 survives, company 2 defaults
p_11 = 0.05  # both companies default together (worst case!)

print("=" * 50)
print("   JOINT PROBABILITY TABLE")
print("=" * 50)
print(f"P(Y1=0, Y2=0) = {p_00}  both survive")
print(f"P(Y1=1, Y2=0) = {p_10}  only company 1 defaults")
print(f"P(Y1=0, Y2=1) = {p_01}  only company 2 defaults")
print(f"P(Y1=1, Y2=1) = {p_11}  both default together!")
print(f"\nSum of all probabilities = {p_00+p_10+p_01+p_11}")
print("(must equal 1.0 - all cases covered)")

# ── Step 2: Marginal Default Probabilities ────────────────────
# Marginal probability = probability that ONE company defaults
# regardless of what happens to the other company

# For company 1:
# p1 = P(Y1=1) = P(Y1=1 AND Y2=0) + P(Y1=1 AND Y2=1)
# In words: company 1 defaults either alone OR together with 2

p1 = p_10 + p_11  # probability company 1 defaults
p2 = p_01 + p_11  # probability company 2 defaults

print("\n" + "=" * 50)
print("   MARGINAL DEFAULT PROBABILITIES")
print("=" * 50)
print(f"Company 1 default probability:")
print(f"p1 = P(Y1=1,Y2=0) + P(Y1=1,Y2=1)")
print(f"p1 = {p_10} + {p_11} = {p1:.4f} ({p1*100:.1f}%)")
print(f"\nCompany 2 default probability:")
print(f"p2 = P(Y1=0,Y2=1) + P(Y1=1,Y2=1)")
print(f"p2 = {p_01} + {p_11} = {p2:.4f} ({p2*100:.1f}%)")

# ── Step 3: Default Correlation ───────────────────────────────
# Correlation measures HOW MUCH the two companies
# tend to default together.

# Range: -1 to +1
# +1 = they always default together
#  0 = completely independent
# -1 = when one defaults, the other never does
#
# Formula:
# rho = (P(both default) - p1*p2) / sqrt(p1*(1-p1)*p2*(1-p2))

# The numerator measures: does joint default happen MORE
# than we would expect if they were independent?
# If independent: P(both default) = p1 * p2

independent_joint = p1 * p2  # what we expect if independent
actual_joint = p_11           # what actually happens

numerator = actual_joint - independent_joint
denominator = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
default_corr = numerator / denominator

print("\n" + "=" * 50)
print("   DEFAULT CORRELATION")
print("=" * 50)
print(f"If companies were INDEPENDENT:")
print(f"P(both default) would be = p1 * p2 = {p1} * {p2} = {independent_joint:.4f}")
print(f"\nActual P(both default) = {actual_joint}")
print(f"Difference = {actual_joint} - {independent_joint:.4f} = {numerator:.4f}")
print(f"(positive means they default together MORE than expected)")
print(f"\nDenominator = sqrt(p1*(1-p1)*p2*(1-p2))")
print(f"           = sqrt({p1}*{1-p1}*{p2}*{1-p2})")
print(f"           = {denominator:.4f}")
print(f"\nDefault correlation = {numerator:.4f} / {denominator:.4f}")
print(f"Default correlation = {default_corr:.4f}")

# ── Step 4: Interpret the Result ──────────────────────────────
print("\n" + "=" * 50)
print("   INTERPRETATION")
print("=" * 50)
if default_corr > 0.3:
    print(f"Strong positive correlation ({default_corr:.4f})")
    print("The two companies tend to default together.")
    print("This is dangerous for a bank holding both loans!")
elif default_corr > 0:
    print(f"Weak positive correlation ({default_corr:.4f})")
    print("Some tendency to default together but not strong.")
else:
    print(f"Negative or zero correlation ({default_corr:.4f})")
    print("Companies tend NOT to default at the same time.")

print(f"\nIn credit risk terms:")
print(f"If we lend to BOTH companies, the chance of losing")
print(f"money on BOTH loans at the same time is {p_11*100:.0f}%")
print(f"This is {'higher' if p_11 > independent_joint else 'lower'} "
      f"than the {independent_joint*100:.2f}% we would expect")
print(f"if the companies were completely independent.")

# ── Step 5: Threshold Model and Copula ───────────────────────
# The threshold model connects continuous creditworthiness
# variables X1, X2 to binary default indicators Y1, Y2
#
# Each company has a creditworthiness variable Xi ~ N(0,1)
# Company i defaults when Xi falls below a threshold di
#
# The threshold di is chosen so that:
# P(Xi <= di) = pi  (matches the marginal default probability)
#
# Therefore: di = Phi^(-1)(pi)
# where Phi^(-1) is the inverse standard normal (norm.ppf)

print("\n" + "=" * 50)
print("   THRESHOLD MODEL AND COPULA")
print("=" * 50)

# Calculate default thresholds
d1 = norm.ppf(p1)
d2 = norm.ppf(p2)

print(f"Default threshold for company 1:")
print(f"d1 = Phi^(-1)(p1) = Phi^(-1)({p1}) = {d1:.4f}")
print(f"\nDefault threshold for company 2:")
print(f"d2 = Phi^(-1)(p2) = Phi^(-1)({p2}) = {d2:.4f}")
print(f"\nMeaning: Company 1 defaults when its")
print(f"creditworthiness X1 falls below {d1:.4f}")

# Gaussian copula C(p1, p2) under independence
# Under independence the joint probability is simply:
# C(p1, p2) = p1 * p2
# This is the independence (product) copula

C_independence = p1 * p2

print(f"\nGaussian Copula under INDEPENDENCE:")
print(f"C(p1, p2) = p1 * p2")
print(f"C({p1}, {p2}) = {p1} * {p2} = {C_independence:.4f}")

print(f"\nComparison:")
print(f"Independence copula C(p1,p2) = {C_independence:.4f}")
print(f"Actual joint default prob    = {p_11:.4f}")
print(f"Difference                   = {p_11 - C_independence:.4f}")
print(f"\nThe actual joint default probability ({p_11}) is")
print(f"{'HIGHER' if p_11 > C_independence else 'LOWER'} than")
print(f"the independence assumption ({C_independence:.4f}).")
print(f"This confirms positive default correlation between")
print(f"the two companies.")