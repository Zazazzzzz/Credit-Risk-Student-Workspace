# ---------------------------------------------------------------
# Script Name: Merton Model - Structural Default Model
# Author: Shazia Ishaq
# Description: This script simulates how a company asset value
#              moves over time and whether it crosses below the
#              debt level causing a default.
#
# WHAT IS MERTON MODEL?
# ---------------------
# Robert Merton (1974) said:
# A company defaults when its ASSETS fall below its DEBT.
#
# Think of a company like a person:
#   Assets = salary, savings, house value
#   Debt   = mortgage, loans, credit cards
#
# If assets > debt  the person can pay back loans (no default)
# If assets < debt  the person goes bankrupt (DEFAULT!)
#
# The asset value moves randomly every day like a stock price.
# We model this random movement using Geometric Brownian Motion.
#
# WHY GEOMETRIC BROWNIAN MOTION?
# ------------------------------
# Asset values cannot go negative (a company cannot be worth
# less than zero). The exponential function in GBM ensures
# the asset value always stays positive.
#
# THE FORMULA:
# V_t = V_0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )
#
# Where:
#   V_t   = asset value at time t
#   V_0   = starting asset value
#   mu    = expected annual growth rate (drift)
#   sigma = annual volatility (uncertainty)
#   W_t   = Brownian motion (random walk)
#   t     = time in years
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------

V0 = 100
# The company assets are worth 100 today (think euros or millions)
# This is where we start our simulation

mu_V = 0.05
# The assets grow by 5% per year on AVERAGE
# Called the DRIFT - general upward trend
# Like house prices tend to go up over time on average
# But with uncertainty around this trend

sigma_V = 0.2
# Annual VOLATILITY of 20%
# Measures how much the asset value bounces around
# High sigma = big random swings (risky company)
# Low sigma  = smooth predictable growth (stable company)
# Example: tech startups have high sigma, utilities have low sigma

T = 1.0
# We simulate ONE YEAR into the future
# After 1 year the company must repay its debt
# If assets < debt at any point -> DEFAULT

B = 90
# The company owes 90 in DEBT (the default threshold/barrier)
# If asset value V_t falls below B = 90 at any time -> DEFAULT
# The bank loses money!
# Distance to default = V0 - B = 100 - 90 = 10
# Only a 10% drop in asset value causes default in this example

N = 1000
# We divide the year into 1000 tiny time steps
# Each step = 1/1000 of a year = about 8.76 hours
# More steps = smoother path = more realistic simulation

dt = T / N
# Length of each time step = 0.001 years
# Used to scale the random shocks at each step

# Set random seed for reproducibility
# Every time you run the code you get the SAME result
# Important for checking and comparing results
np.random.seed(42)

# ---------------------------------------------------------------
# STEP 1: GENERATE BROWNIAN MOTION
# ---------------------------------------------------------------
# Brownian motion W_t models pure random movement
# Like a pollen grain bouncing randomly in water
# Each tiny step is equally likely to go up or down
#
# How we build it:
# 1. Generate N random shocks from Normal(0, sqrt(dt))
#    Each shock is tiny because dt = 0.001 is small
# 2. cumsum() adds shocks progressively to build the path
#
# Example with 5 steps:
# Shocks:      [+0.02, -0.03, +0.01, -0.02, +0.04]
# Cumulative:  [+0.02, -0.01, +0.00, -0.02, +0.02]
# This builds W_t step by step from t=0 to t=T

W = np.random.normal(0, np.sqrt(dt), N).cumsum()
# np.random.normal(mean, std, size)
# mean = 0       each step has no preferred direction
# std  = sqrt(dt) scales the randomness by time step size
# N    = 1000    generate 1000 random shocks
# .cumsum()      add them up to build the full path

# ---------------------------------------------------------------
# STEP 2: CREATE TIME GRID
# ---------------------------------------------------------------
# We need time points: 0, 0.001, 0.002, ..., 1.0
# These are the x-axis values for our plot
# np.linspace creates evenly spaced numbers
# np.linspace(start, stop, num_points)

time_grid = np.linspace(0, T, N)
# Creates array: [0.000, 0.001, 0.002, ..., 0.999]
# 1000 evenly spaced time points from 0 to 1 year

# ---------------------------------------------------------------
# STEP 3: SIMULATE ASSET VALUE PATH
# ---------------------------------------------------------------
# Using the exact GBM formula:
# V_t = V_0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )
#
# Why -0.5*sigma^2?
# This is the ITO CORRECTION from advanced stochastic calculus
# Without it the model would overestimate expected growth
# It adjusts for the curvature of the exponential function
# You do not need to derive it - just know it must be there!
#
# The formula has two parts:
# (mu - 0.5*sigma^2)*t = DETERMINISTIC part (predictable growth)
# sigma*W_t            = RANDOM part (unpredictable fluctuations)

VT = V0 * np.exp((mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W)
# V0 = 100 (starting value)
# exp(...) ensures value is always positive (cannot go below 0)
# time_grid has shape (1000,)
# W has shape (1000,)
# VT has shape (1000,) - asset value at each of 1000 time points

# ---------------------------------------------------------------
# STEP 4: ANALYZE DEFAULT
# ---------------------------------------------------------------
# Check if asset value ever fell below debt level B = 90
# any(VT <= B) returns True if ANY value in VT is below B
# It returns False if asset value stayed above B the whole year

default_occurred = any(VT <= B)
min_value = VT.min()
final_value = VT[-1]

print("=" * 55)
print("   MERTON MODEL SIMULATION RESULTS")
print("=" * 55)
print(f"Starting asset value:    {V0}")
print(f"Debt level (threshold):  {B}")
print(f"Expected growth (drift): {mu_V*100:.1f}% per year")
print(f"Asset volatility:        {sigma_V*100:.1f}% per year")
print(f"Time horizon:            {T} year")
print(f"Number of time steps:    {N}")
print("-" * 55)
print(f"Final asset value:       {final_value:.4f}")
print(f"Lowest asset value:      {min_value:.4f}")
print(f"Default threshold:       {B}")
print(f"Closest approach to B:   {min_value - B:.4f} above threshold")
print(f"DEFAULT OCCURRED:        {default_occurred}")
print("=" * 55)

if not default_occurred:
    print("The company survived the full year without defaulting.")
    print("The bank gets its money back in full.")
    print(f"Safety margin: asset value stayed {min_value - B:.2f} above debt.")
else:
    print("The company crossed the default threshold!")
    print("The bank loses part or all of its loan.")

# ---------------------------------------------------------------
# STEP 5: PLOT THE ASSET VALUE PATH
# ---------------------------------------------------------------
# The plot shows:
# Blue line  = asset value path over 1 year (our simulation)
# Red dashed = debt level B = 90 (the DANGER LINE)
# Gray dots  = starting value V0 = 100
# Red shaded = DANGER ZONE below the debt level

plt.figure(figsize=(12, 7))

# Plot the simulated asset value path
plt.plot(time_grid, VT,
         color='steelblue',
         linewidth=1.5,
         label="Asset Value Path $V_t$")

# Add horizontal line at debt level B
plt.axhline(y=B,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Default Threshold (Debt) B = {B}")

# Add horizontal line at starting value
plt.axhline(y=V0,
            color='gray',
            linestyle=':',
            linewidth=1,
            alpha=0.7,
            label=f"Starting Value V0 = {V0}")

# Shade the danger zone below B in light red
plt.fill_between(time_grid, 0, B,
                 alpha=0.08,
                 color='red',
                 label='Default Zone')

plt.title("Merton Structural Model: Simulated Asset Value Path\n"
          f"GBM parameters: mu={mu_V}, sigma={sigma_V}, T={T} year",
          fontsize=13,
          fontweight='bold')
plt.xlabel("Time (Years)", fontsize=12)
plt.ylabel("Asset Value", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# STEP 6: MONTE CARLO - SIMULATE MANY PATHS
# ---------------------------------------------------------------
# One path tells us one possible future.
# To estimate the PROBABILITY of default we need many paths!
# This is called Monte Carlo simulation.
#
# We simulate 5000 paths and count how many result in default.
# Default probability = number of default paths / total paths

num_simulations = 5000
default_count = 0

print(f"\nRunning Monte Carlo with {num_simulations} simulations...")

for i in range(num_simulations):
    W_mc = np.random.normal(0, np.sqrt(dt), N).cumsum()
    VT_mc = V0 * np.exp(
        (mu_V - 0.5 * sigma_V**2) * time_grid + sigma_V * W_mc
    )
    if any(VT_mc <= B):
        default_count += 1

pd_estimate = default_count / num_simulations

print(f"\nMonte Carlo Results:")
print(f"Default paths:           {default_count} out of {num_simulations}")
print(f"Estimated default prob:  {pd_estimate*100:.2f}%")
print(f"\nInterpretation:")
print(f"With V0={V0}, B={B}, mu={mu_V}, sigma={sigma_V}:")
print(f"There is approximately a {pd_estimate*100:.1f}% chance")
print(f"this company defaults within {T} year.")
if pd_estimate < 0.01:
    print("This is investment grade credit quality (low risk).")
elif pd_estimate < 0.05:
    print("This is sub-investment grade (moderate risk).")
else:
    print("This is high yield / speculative grade (high risk).")