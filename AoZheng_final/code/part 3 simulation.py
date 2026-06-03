import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./Data/"

def run_simulation():

    df = pd.read_csv(DATA_DIR + 'final_modeling_data.csv')

    df = df.dropna(subset=['EAD', 'lgd', 'default_threshold', 'beta'])

    n_assets = len(df)
    EAD = df['EAD'].values
    LGD = df['lgd'].values
    beta = df['beta'].values
    threshold = df['default_threshold'].values
    L_i  = EAD * LGD
    eta = np.sqrt(1 - beta ** 2)

    n_simulations = 10000
    np.random.seed(2)

    #F (10000 x 1)
    F = np.random.standard_normal((n_simulations, 1))

    #epsilon (10000 x 56)
    epsilon = np.random.standard_normal((n_simulations, n_assets))

    #X
    X = beta * F + eta * epsilon

    default_events = X < threshold

    portfolio_losses = np.sum(default_events * L_i, axis=1)

    EL = np.mean(portfolio_losses)

    confidence_level = 0.99

    #VaR
    VaR_99 = np.percentile(portfolio_losses, confidence_level * 100)

    #ES
    tail_losses = portfolio_losses[portfolio_losses > VaR_99]
    ES_99 = np.mean(tail_losses) if len(tail_losses) > 0 else VaR_99

    results_df = pd.DataFrame({
        'Risk_Metric': ['Total EAD', 'Expected Loss (EL)', '99% VaR', '99% ES'],
        'Value': [EAD.sum(), EL, VaR_99, ES_99]
    })
    # Table 2: Simulated Portfolio Risk Metrics
    csv_path = DATA_DIR + 'simulation_metrics.csv'
    results_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))

    n, bins, patches = plt.hist(portfolio_losses, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

    plt.axvline(EL, color='green', linestyle='dashed', linewidth=2, label=f'Expected Loss (EL)')
    plt.axvline(VaR_99, color='orange', linestyle='dashed', linewidth=2, label=f'99% VaR')
    plt.axvline(ES_99, color='red', linestyle='dashed', linewidth=2, label=f'99% ES')

    for i in range(len(bins) - 1):
        if bins[i] >= VaR_99:
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.6)

    plt.title('Portfolio Loss Distribution (10,000 Simulations)', fontsize=14)
    plt.xlabel('Portfolio Loss Amount', fontsize=12)
    plt.ylabel('Frequency (Number of Scenarios)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    img_path = DATA_DIR + 'loss_distribution.png'
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    # Figure 1: Portfolio Loss Distribution
    plt.show()


if __name__ == "__main__":
    run_simulation()