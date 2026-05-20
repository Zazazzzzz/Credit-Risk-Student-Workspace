#Task 2
import numpy as np
import pandas as pd
from scipy.stats import norm
from pathlib import Path
import matplotlib.pyplot as plt

rating_to_prob={
"AAA" :0.0001,
"AA" : 0.0001,
"A" : 0.0001,
"BBB" : 0.04,
"BB" : 0.06,
"B": 0.11
}

# das hatte ich gemacht, war falsch!!!
#dAAA = norm.ppf(prob_AAA)
#dAA = norm.ppf(prob_AA)
#dA = norm.ppf(prob_A)
#dBBB = norm.ppf(prob_BBB)
#dBB = norm.ppf(prob_BB)
#dB = norm.ppf(prob_B)

#print(dBBB)


base_path = Path(__file__).resolve().parent.parent
portfolio_file = base_path / "Credit Risk" / "portfolio.xlsx"
portf_data = pd.read_excel(portfolio_file)
print(portf_data.head())

portf_data['prob_default'] = portf_data['rating'].map(rating_to_prob)

# Step 2: Convert each p into a default threshold d = Phi^{-1}(p)
portf_data['threshold'] = norm.ppf(portf_data['prob_default'])

print("\nAfter adding probability and threshold:")
print(portf_data)

# Step 3: Simulate correlated defaults
np.random.seed(42)

n_simulations = 100000
n_obligors = len(portf_data)

rhos = portf_data['rho'].values

#systematic factor F (commonfor all obligors)
F = np.random.normal(0, 1, n_simulations)

#idiosyncratic shocks for each obligor
epsilon = np.random.normal(0, 1, (n_simulations, n_obligors))

# Calculate latent asset values
Z = np.sqrt(rhos) * F.reshape(-1, 1) + np.sqrt(1 - rhos) * epsilon #Whyreshape?

# 1 if default, 0 otherwise
# Default when Z < threshold
thresholds = portf_data['threshold'].values.reshape(1, -1)
defaults = (Z < thresholds).astype(int)

# Default statistics per obligor
default_prob_simulated = defaults.mean(axis=0)
print("\nDefault probabilities (simulated vs input):")
for i, obligor in enumerate(portf_data['obligors']):
    print(f"{obligor}: input={portf_data['prob_default'].iloc[i]:.4f}, "
          f"simulated={default_prob_simulated[i]:.4f}")

# Step 4 portfolio loss
# L = sum_i exposure_i * LGD_i * Y_i
exposures = portf_data['exposure'].values

lgds = portf_data['LGD'].values

# Calculate loss per simulation
losses = defaults @ (exposures * lgds)   # dimensions !!! Matrices and scalars

#results display (really nic -> REMEMBER)!!
print("\n" + "="*60)
print("PORTFOLIO LOSS STATISTICS")
print("="*60)
print(f"Number of simulations: {n_simulations:,}")
print(f"Number of obligors: {n_obligors}")
print(f"\nLoss Statistics:")
print(f"  Mean loss: {losses.mean():.6f}")
print(f"  Median loss: {np.median(losses):.6f}")
print(f"  Standard deviation: {losses.std():.6f}")
print(f"  Min loss: {losses.min():.6f}")
print(f"  Max loss: {losses.max():.6f}")

# VaR
var_95 = np.percentile(losses, 95)
var_99 = np.percentile(losses, 99)
var_999 = np.percentile(losses, 99.9)

print(f"\nValue at Risk (VaR):")
print(f"  VaR 95%: {var_95:.6f}")
print(f"  VaR 99%: {var_99:.6f}")
print(f"  VaR 99.9%: {var_999:.6f}")

# Expected Shortfall
es_95 = losses[losses >= var_95].mean()
es_99 = losses[losses >= var_99].mean()

print(f"\nExpected Shortfall (CVaR):")
print(f"  ES 95%: {es_95:.6f}")
print(f"  ES 99%: {es_99:.6f}")

# Expected Loss and Unexpected Loss
EL = losses.mean()
UL = losses.std()

print(f"\nRisk Metrics:")
print(f"  Expected Loss (EL): {EL:.6f}")
print(f"  Unexpected Loss (UL): {UL:.6f}")


#Loss distrib stats
print(f"\nLoss Distribution Percentiles:")
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  {p}th percentile: {np.percentile(losses, p):.6f}")

# Number of defaults distribution
num_defaults = defaults.sum(axis=1)
print(f"\nDefault Statistics:")
print(f"  Average number of defaults: {num_defaults.mean():.2f}")
print(f"  Max number of defaults: {num_defaults.max()}")
print(f"  Probability of no default: {(num_defaults == 0).mean():.4f}")
print(f"  Probability of any default: {(num_defaults > 0).mean():.4f}")

#figure only AI
# Create a figure with multiple subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Loss Distribution", fontsize=16, fontweight='bold')

# Plot 1: Histogram of losses
ax1 = axes[0]
n_bins = 50
counts, bins, patches = ax1.hist(losses, bins=n_bins, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.axvline(losses.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {losses.mean():.4f}')
ax1.axvline(np.percentile(losses, 95), color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {np.percentile(losses, 95):.4f}')
ax1.axvline(np.percentile(losses, 99), color='darkred', linestyle='--', linewidth=2, label=f'VaR 99%: {np.percentile(losses, 99):.4f}')
ax1.set_xlabel('Portfolio Loss', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title(f'Loss Distribution (n={len(losses):,} simulations)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative distribution function (CDF)
ax2 = axes[1]
sorted_losses = np.sort(losses)
cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
ax2.plot(sorted_losses, cdf, color='steelblue', linewidth=2)
ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=0.99, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=np.percentile(losses, 95), color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axvline(x=np.percentile(losses, 99), color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Portfolio Loss', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Distribution Function', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

