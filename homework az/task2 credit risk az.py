import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
PD_MAP = {
    'AAA': 1e-7, 'AA': 0.0002, 'A': 0.0005, 'BBB': 0.0016,
    'BB': 0.0063, 'B': 0.0326, 'CCC': 0.2668
}
def analyze_portfolio(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    p = df['rating'].map(PD_MAP).values
    rho = df['rho'].values
    d = norm.ppf(p)
    n_sims, n_assets = 100000, len(df)
    np.random.seed(42)
    M = np.random.normal(0, 1, (n_sims, 1))
    Z = np.random.normal(0, 1, (n_sims, n_assets))
    X = np.sqrt(rho) * M + np.sqrt(1 - rho) * Z
    losses = np.sum((X < d) * (df['exposure'] * df['LGD']).values, axis=1)
    alpha = 0.999
    sim_var = np.percentile(losses, alpha * 100)
    sim_es = losses[losses >= sim_var].mean()
    f_alpha = norm.ppf(1 - alpha)  # 0.1% 极差环境
    p_cond = norm.cdf((d - np.sqrt(rho) * f_alpha) / np.sqrt(1 - rho))
    ana_var = np.sum(df['exposure'] * df['LGD'] * p_cond)
    print(f"--- 结果: {sheet_name} ---")
    print(f"预期损失 (EL): {losses.mean():.6f}")
    print(f"模拟法 VaR:   {sim_var:.6f}")
    print(f"解析法 VaR:   {ana_var:.6f}")
    print(f"模拟法 ES:    {sim_es:.6f}\n")
file = 'portfolio.xlsx'
analyze_portfolio(file, 'portfolio1')
analyze_portfolio(file, 'portfolio2')