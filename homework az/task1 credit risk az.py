import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

file_asia = r'D:\pycharm\Credit-Risk-Student-Workspace1\Credit Risk\EM Asia monthly.xls'
file_europe = r'D:\pycharm\Credit-Risk-Student-Workspace1\Credit Risk\EM Europe and Middle East monthly.xls'


def get_log_returns(file_name, region_name):
    df = pd.read_excel(file_name, header=6)
    date_col = df.columns[0]
    price_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    df = df.sort_values(date_col)
    ret_col = f'Ret_{region_name}'
    df[ret_col] = np.log(df[price_col].astype(float) / df[price_col].astype(float).shift(1))
    return df[[date_col, ret_col]].rename(columns={date_col: 'Date'}).dropna()


try:
    asia_ret = get_log_returns(file_asia, 'Asia')
    europe_ret = get_log_returns(file_europe, 'Europe_ME')
    merged_df = pd.merge(asia_ret, europe_ret, on='Date')

    correlation = merged_df['Ret_Asia'].corr(merged_df['Ret_Europe_ME'])

    num_simulations = 10000
    num_companies = 10
    p_default = 0.05
    rho = 0.5
    threshold = norm.ppf(p_default)

    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    factors = np.random.multivariate_normal(mean, cov, num_simulations)
    F_asia = factors[:, 0]
    F_europe_me = factors[:, 1]

    defaults_matrix = np.zeros((num_simulations, num_companies))
    for i in range(num_companies):
        epsilon = np.random.normal(0, 1, num_simulations)
        f_reg = F_asia if i < 5 else F_europe_me
        x = rho * f_reg + np.sqrt(1 - rho ** 2) * epsilon
        defaults_matrix[:, i] = (x <= threshold)

    total_defaults = np.sum(defaults_matrix, axis=1)

    plt.hist(total_defaults, bins=np.arange(num_companies + 2) - 0.5, rwidth=0.8, density=True)
    plt.title(f'Default Distribution (Corr: {correlation:.2f})')
    plt.xticks(range(num_companies + 1))
    plt.show()

except Exception as e:
    print(f"Error: {e}")