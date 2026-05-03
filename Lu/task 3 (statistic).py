import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#task 1 数据清洗
df = pd.read_excel('EFM Asia daily.xlsx', header=6)
df.columns = ['Date', 'Price']
price_col = 'EFM ASIA Standard (Large+Mid Cap)'
date_col = 'Date'
#转换日期并处理末尾杂质clean data
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df = df.sort_values(by=date_col)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])
#只要保证 df.columns 里的名字和后面 df[price_col] 调用的名字一模一样
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])
df['Return'] = df['Price'].pct_change()#compute daily return
returns_df = df[[date_col, 'Return']].dropna()
print(returns_df.head(10))
#task 2
mean_return = returns_df['Return'].mean()
var_95_threshold = returns_df['Return'].quantile(0.05)
print(f"Mean Daily Return: {mean_return}")
print(f"95% VaR Threshold (5% Quantile): {var_95_threshold}")
plt.figure(figsize=(10, 6))#draw
plt.hist(returns_df['Return'], bins=50, color='blue', edgecolor='red', alpha=0.7)#bins 指分组数量
plt.axvline(mean_return, color='yellow', linestyle='--', label=f'Mean: {mean_return:.4f}')
plt.axvline(var_95_threshold, color='red', linestyle='--', label=f'95% VaR Limit: {var_95_threshold:.4f}')
plt.title('Daily Return Distribution and 95% VaR')
plt.xlabel('Daily Return')
plt.legend()
plt.show()
# Task 3:
mean_return = returns_df["Return"].mean()
quantile_5 = returns_df["Return"].quantile(0.05)
VaR_95 = -quantile_5
print("-" * 30)#分隔符divider
print(f"Mean Daily Return: {mean_return:.6f}")#:.6f保留六位小数的意思
print(f"5% Quantile: {quantile_5:.6f}")
print(f"95% Value at Risk (VaR): {VaR_95:.6f}")
print("-" * 30)
#task4
#the distribution is not perfectly symmetric.there are extreme values and a visible "fat tail" on the left side.
#The significant disparity between the Mean Daily Return (0.02%) and the 95% VaR (2.48%) highlights a severe risk-return asymmetry, suggesting that the asset is prone to fat-tail risks despite its stable average performance.
#We are 95% confident that the index will not lose more than 2.48% in a single day based on historical data.
#Extreme losses are statistically rare but significant in impact.





