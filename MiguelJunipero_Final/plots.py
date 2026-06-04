import os
import numpy as np
import matplotlib.pyplot as plt
from MiguelJunipero_Final.main import DIR_RESULTS

def generate_portfolio_plots(dir_results):
    """
    """
    # Verify the results folder exists
    if not os.path.exists(dir_results):
        print(f"Error: The directory '{dir_results}' does not exist. Run main.py first.")
        return

    # Define portfolios
    portfolios = {
        "1": {"a_file": "losses_p1_model_A.txt", "b_file": "losses_p1_model_B.txt"},
        "2": {"a_file": "losses_p2_model_A.txt", "b_file": "losses_p2_model_B.txt"}
}

    for label, files in portfolios.items():
        path_a = os.path.join(dir_results, files["a_file"])
        path_b = os.path.join(dir_results, files["b_file"])

        # Safe check to make sure main.py actually exported the data first
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"Skipping Portfolio {label}: Loss text files not found.")
            continue

        losses_a = np.loadtxt(path_a)
        losses_b = np.loadtxt(path_b)

        # Compute risk measures for the visual marker overlays
        el_a = np.mean(losses_a)
        var_a = np.percentile(losses_a, 99)
        es_a = np.mean(losses_a[losses_a >= var_a])

        el_b = np.mean(losses_b)
        var_b = np.percentile(losses_b, 99)
        es_b = np.mean(losses_b[losses_b >= var_b])

        losses_a_nonzero = losses_a[losses_a > 0]
        losses_b_nonzero = losses_b[losses_b > 0]

        plt.figure(figsize=(12, 6))
        plt.hist(losses_a_nonzero / 1e6, bins=100, alpha=0.5, label="Model A (One-Factor Systemic)", color="crimson", density=True)
        plt.hist(losses_b_nonzero / 1e6, bins=100, alpha=0.5, label="Model B (Independent Bernoulli)", color="dodgerblue",
                 density=True)

        # Overlay the risk measure anchors
        plt.axvline(el_a / 1e6, color="darkred", linestyle="--", linewidth=1.5, label=f"Model A EL (${el_a / 1e6:.2f}M)")
        plt.axvline(var_a / 1e6, color="red", linestyle="-", linewidth=2, label=f"Model A 99% VaR (${var_a / 1e6:.2f}M)")
        plt.axvline(es_a / 1e6, color="purple", linestyle="-.", linewidth=2, label=f"Model A 99% ES (${es_a / 1e6:.2f}M)")

        plt.axvline(el_b / 1e6, color="navy", linestyle=":", linewidth=1.5, label=f"Model B EL (${el_b / 1e6:.2f}M)")
        plt.axvline(var_b / 1e6, color="teal", linestyle="-", linewidth=2, label=f"Model B 99% VaR (${var_b / 1e6:.2f}M)")
        plt.axvline(es_b / 1e6, color="royalblue", linestyle="-.", linewidth=2, label=f"Model B 99% ES (${es_b / 1e6:.2f}M)")

        plt.title(f"Portfolio {label} Loss Distribution Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Total Portfolio Loss ($)", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=10, loc="upper right", framealpha=0.95)

        plt.tight_layout()
        output_image_path = os.path.join(dir_results, f"portfolio_{label}_loss_distribution.png")
        plt.savefig(output_image_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    generate_portfolio_plots(dir_results=DIR_RESULTS)