'''
A bank lends 100,000EUR each to 10 firms

All firms promise to repay the full amount in one year (no interest, single payment).

Each Firm may default:
Firm 1: 1% chance of default
Firm 2: 2%
...
Firm 10: 10%

If a firm defaults, it only repays 50,000EUR
⇒ the bank loses 50,000EUR
'''

import numpy as np
import matplotlib.pyplot as plt
loan_amount = 100000
recovery_amount = 50000
loss_given_default = loan_amount - recovery_amount

'''
# On average:
# • How much loss should the bank expect?

# If a firm defaults
default_probs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
expected_losses = default_probs * loss_given_default
total_expected_loss = expected_losses.sum()

print(f"Expected loss for each firm: {expected_losses}")
print(f"Total expected loss: {total_expected_loss}")

Expected loss for each firm: [ 500. 1000. 1500. 2000. 2500. 3000. 3500. 4000. 4500. 5000.]
Total expected loss: 27500.0
'''



'''
In reality:
• Losses are uncertain
• If we repeat this situation 1,000 times (simulation), what does
the distribution of total loss look like?

Simulation steps:
1. simulate default indicators
2. compute losses for each obligor
3. aggregate portfolio losses
4. repeat for many scenarios
5. analyse the loss distribution
'''

n_firms = 10
n_scenarios = 1000

# Average default probabilities: 1%, 2%, ..., 10%
mean_default_probs = np.arange(1, 11) / 100

# Beta distribution concentration parameter
kappa = 100
# Higher kappa means default probabilities are less uncertain and vice versa
alpha = mean_default_probs * kappa
beta = (1 - mean_default_probs) * kappa

np.random.seed(42)

# Store total portfolio loss for each scenario
total_losses = np.zeros(n_scenarios)
# prepare an empty storage array for the total loss from each simulated scenario

for s in range(n_scenarios):

    # simulate default probabilities
    default_probs = np.random.beta(alpha, beta)

    # simulate default indicators: 1 default, 0 no default
    default_indicators = np.random.binomial(1, default_probs)

    # compute losses for each firm
    losses = default_indicators * loss_given_default

    total_losses[s] = losses.sum()

# Analyse the loss distribution
mean_loss = total_losses.mean()
median_loss = np.median(total_losses)
percentile_95 = np.percentile(total_losses, 95)
max_loss = total_losses.max()

print(f"Mean portfolio loss: {mean_loss:.2f} EUR")
print(f"Median portfolio loss: {median_loss:.2f} EUR")
print(f"95th percentile loss: {percentile_95:.2f} EUR")
print(f"Maximum loss: {max_loss:.2f} EUR")

# Plot loss distribution
# -----------------------------
plt.figure(figsize=(10, 5))
plt.hist(total_losses, bins=20, edgecolor="black")

plt.xlabel("Total portfolio loss")
plt.ylabel("Frequency")
plt.title("Distribution of Total Portfolio Losses")
plt.grid(True)
plt.show()


