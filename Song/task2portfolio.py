import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

file_path = "portfolio.xlsx"

portfolios = {
    "Portfolio 1": pd.read_excel(file_path, sheet_name="portfolio1"),
    "Portfolio 2": pd.read_excel(file_path, sheet_name="portfolio2")
}

rating_to_pd = {
    "AAA": 0.00001,
    "AA": 0.00001,
    "A": 0.0001,
    "BBB": 0.0006,
    "BB": 0.0040,
    "B": 0.0238
}

n_sim = 200000
alpha = 0.999
summary = []


# ============================================================
# Step 1 to Step 8: original rho
# ============================================================

plot_original = {}

for case, data in portfolios.items():

    # Step 1
    data["p"] = data["rating"].map(rating_to_pd)

    # Step 2
    data["d"] = norm.ppf(data["p"])

    # Step 3
    np.random.seed(123)
    F = np.random.normal(size=n_sim)
    eps = np.random.normal(size=(n_sim, len(data)))

    X = data["rho"].values * F[:, None] + np.sqrt(1 - data["rho"].values**2) * eps
    Y = (X < data["d"].values).astype(int)

    # Step 4
    L = np.sum(data["exposure"].values * data["LGD"].values * Y, axis=1)

    # Step 5
    sim_VaR = np.quantile(L, alpha)
    sim_ES = np.mean(L[L >= sim_VaR])

    # Step 6
    np.random.seed(456)
    F_grid = np.random.normal(size=n_sim)

    q = norm.cdf(
        (data["d"].values - data["rho"].values * F_grid[:, None])
        / np.sqrt(1 - data["rho"].values**2)
    )

    lF = np.sum(data["exposure"].values * data["LGD"].values * q, axis=1)

    # Step 7
    ana_VaR = np.quantile(lF, alpha)
    ana_ES = np.mean(lF[lF >= ana_VaR])

    plot_original[case] = (L, lF)
    summary.append([case + " original rho", sim_VaR, sim_ES, ana_VaR, ana_ES])


# Step 8
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, (case, (L, lF)) in enumerate(plot_original.items()):
    ax[i].hist(L, bins=40, density=True, alpha=0.45, label="Simulation")
    ax[i].hist(lF, bins=40, density=True, alpha=0.45, label="Analytical")
    ax[i].set_title(case)
    ax[i].set_xlabel("Loss")
    ax[i].set_xlim(0, 0.25)
    ax[i].set_ylim(0, 500)
    ax[i].legend()

ax[0].set_ylabel("Density")
fig.suptitle("Original rho")
plt.tight_layout()
plt.show()


# ============================================================
# Step 9: high rho, redo Step 1 to Step 8
# ============================================================

rho_new = [0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7]
plot_high = {}

for case, old_data in portfolios.items():

    data = old_data.copy()
    data["rho"] = rho_new

    # Step 1
    data["p"] = data["rating"].map(rating_to_pd)

    # Step 2
    data["d"] = norm.ppf(data["p"])

    # Step 3
    np.random.seed(123)
    F = np.random.normal(size=n_sim)
    eps = np.random.normal(size=(n_sim, len(data)))

    X = data["rho"].values * F[:, None] + np.sqrt(1 - data["rho"].values**2) * eps
    Y = (X < data["d"].values).astype(int)

    # Step 4
    L = np.sum(data["exposure"].values * data["LGD"].values * Y, axis=1)

    # Step 5
    sim_VaR = np.quantile(L, alpha)
    sim_ES = np.mean(L[L >= sim_VaR])

    # Step 6
    np.random.seed(456)
    F_grid = np.random.normal(size=n_sim)

    q = norm.cdf(
        (data["d"].values - data["rho"].values * F_grid[:, None])
        / np.sqrt(1 - data["rho"].values**2)
    )

    lF = np.sum(data["exposure"].values * data["LGD"].values * q, axis=1)

    # Step 7
    ana_VaR = np.quantile(lF, alpha)
    ana_ES = np.mean(lF[lF >= ana_VaR])

    plot_high[case] = (L, lF)
    summary.append([case + " high rho", sim_VaR, sim_ES, ana_VaR, ana_ES])


# Step 8 again
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, (case, (L, lF)) in enumerate(plot_high.items()):
    ax[i].hist(L, bins=40, density=True, alpha=0.45, label="Simulation")
    ax[i].hist(lF, bins=40, density=True, alpha=0.45, label="Analytical")
    ax[i].set_title(case)
    ax[i].set_xlabel("Loss")
    ax[i].set_xlim(0, 0.25)
    ax[i].set_ylim(0, 500)
    ax[i].legend()

ax[0].set_ylabel("Density")
fig.suptitle("High rho")
plt.tight_layout()
plt.show()


summary = pd.DataFrame(
    summary,
    columns=["Case", "Simulation VaR", "Simulation ES", "Analytical VaR", "Analytical ES"]
)

print(summary)