import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

n_obligors = 100
p = 0.03
rho = 0.5
lgd = 0.4
w = 0.01
n_simulations = 2000
confidence_level = 0.999

np.random.seed(42)
threshold = stats.norm.ppf(p)

Z = np.random.normal(0, 1, (n_simulations, 1))
epsilon = np.random.normal(0, 1, (n_simulations, n_obligors))
X = np.sqrt(rho) * Z + np.sqrt(1 - rho) * epsilon

defaults = (X < threshold).astype(int)
portfolio_losses = np.sum(defaults, axis=1) * lgd * w

mean_loss = np.mean(portfolio_losses)
std_loss = np.std(portfolio_losses)
analytical_var = stats.norm.ppf(confidence_level, loc=mean_loss, scale=std_loss)
empirical_var = np.percentile(portfolio_losses, confidence_level * 100)

print(f"Analytical VaR: {analytical_var:.4f}")
print(f"Simulated VaR: {empirical_var:.4f}")

plt.figure(figsize=(8, 6))
kde = stats.gaussian_kde(portfolio_losses)
x_axis = np.linspace(-0.02, 0.20, 500)
y_axis = kde(x_axis)

plt.fill_between(x_axis, y_axis, color="#C49A9A", alpha=0.5)
plt.plot(x_axis, y_axis, color="#C49A9A", linewidth=1.5)
plt.axvline(x=0.125, color="gray", linestyle="--", label="VaR (99.9%) = 0.12")

plt.title("Simulated Loss Distribution", fontsize=14, pad=15)
plt.xlabel("Total Loss", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(-0.02, 0.20)
plt.ylim(0, 55)
plt.xticks([0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175])
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc="upper right")

plt.tight_layout()
plt.savefig("loss_distribution_plot.png", dpi=300)
plt.show()