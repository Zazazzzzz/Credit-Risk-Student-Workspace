# ---------------------------------------------------------------
# Script Name: Functions and Simple Data Manipulations - Tasks
# Author: Shazia
# Description: Section 2 Tasks
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# Task A: Net Present Value (NPV)
# --------------------------------------------------------------

def calculate_npv(cash_flows: list, discount_rate: float):
    npv = 0
    for t, cf in enumerate(cash_flows):
        npv = npv + cf / (1 + discount_rate) ** t
    print(f"Net Present Value (NPV): ${npv:.2f}")
    return npv

c = [100, 1000, 500, 300, 600, 800]
r = 0.01
v = calculate_npv(c, r)


# --------------------------------------------------------------
# Task B: Bank Account Balance
# --------------------------------------------------------------

tran = [
    {'amount': 150.0, 'type': 'expense'},
    {'amount': 200.0, 'type': 'income'},
    {'amount': 650.0, 'type': 'expense'},
    {'amount': 300.0, 'type': 'expense'},
    {'amount': 400.0, 'type': 'income'}
]
origin = 100
thre = 200
cap = 300


def balance(transactions: list, threshold: int or float,
            origin_balance: int or float, capacity: int or float):
    b = origin_balance
    for time, transaction in enumerate(transactions):
        if transaction['type'] == 'expense':
            b = b - transaction['amount']
            if transaction['amount'] > threshold:
                print(f'transaction {time+1} exceeded threshold {threshold}')
            if b < 0:
                print(f'transaction {time+1} created a negative balance')
                if b < capacity * (-1):
                    print(f'your account is canceled due to transaction {time+1}')
                    break
        else:
            b = b + transaction['amount']
            print(f'After the transaction {time+1}, the balance is ${b:.2f}')
    print(f'your balance now is {b}')
    return b

account_a = balance(tran, thre, origin, cap)
account_b = balance(threshold=thre, origin_balance=origin,
                    transactions=tran, capacity=cap)


# --------------------------------------------------------------
# Task A: Calculator with try/except
# --------------------------------------------------------------

def calculator():
    try:
        num1 = float(input('enter the first number: '))
        num2 = float(input('enter the second number: '))
        operation = input('enter the operation (+, -, *, /): ').strip()

        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            if num2 == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            result = num1 / num2
        else:
            raise ValueError("Invalid operation.")
        print(f"The result of {num1} {operation} {num2} is {result:.2f}.")

    except Exception as e:
        print(f'Error: {type(e).__name__}')

calculator()


# --------------------------------------------------------------
# prices.csv analysis
# --------------------------------------------------------------

df = pd.read_csv('prices.csv')

df["Daily_Return_percentage"] = df["Price"].pct_change()
df['Daily_Return'] = df['Price'] - df['Price'].shift(1)
df['30_Day_Log_Return'] = np.log(df['Price'] / df['Price'].shift(30))
df['Daily_Return_Volatility'] = df['Daily_Return_percentage'].rolling(window=30).std()
df['30_Day_Log_Return_Volatility'] = df['30_Day_Log_Return'].rolling(window=30).std()

df.set_index('Date', inplace=True)

xticks = df.index[::len(df)//4]

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Price"], label="Stock Price", color="blue")
plt.title("Stock Price Over Time")
plt.xticks(xticks)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', color='grey', linewidth=0.5)
plt.savefig('show.png')
plt.show()
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Daily_Return_percentage'], label='Daily Return', alpha=0.7)
plt.plot(df.index, df['30_Day_Log_Return'], label='30-Day Log Return', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.title("Stock Returns Over Time")
plt.xticks(xticks)
plt.xlabel("Date")
plt.ylabel("Returns")
plt.grid(True, axis='y', linestyle='--', color='grey', linewidth=0.5)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Daily_Return_Volatility'], label='Daily Return Volatility', alpha=0.7)
plt.plot(df.index, df['30_Day_Log_Return_Volatility'], label='30-Day Log Return Volatility', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.title("Volatility of Stock Returns Over Time")
plt.xticks(xticks)
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True, axis='y', linestyle='--', color='grey', linewidth=0.5)
plt.show()


# --------------------------------------------------------------
# NumPy arrays and operations
# --------------------------------------------------------------

array_1d = np.array([10, 20, 30, 40, 50])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

print("1D Array Slicing:")
print("Original:", array_1d)
print("First 3 elements:", array_1d[:3])
print("Every second element:", array_1d[::2])

print("2D Array Slicing:")
print("Original:\n", array_2d)
print("First row:", array_2d[0])
print("First column:", array_2d[:, 0])
print("Element at (1,2):", array_2d[1, 2])

print("Add 10 to every element:\n", array_2d + 10)
other_1d = np.array([1, 2, 3, 4, 5])
print("Addition:", array_1d + other_1d)
print("Multiplication:", array_1d * other_1d)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_product = A @ B
print("Matrix Product (A @ B):\n", matrix_product)


# --------------------------------------------------------------
# Correlated stock simulation
# --------------------------------------------------------------

np.random.seed(42)

num_days = 260
mean = [0, 0]
volatility = 1
correlation = 0.7

cov_matrix = np.array([[volatility**2, correlation * volatility**2],
                       [correlation * volatility**2, volatility**2]])

correlated_returns = np.random.multivariate_normal(mean, cov_matrix, size=num_days)

df_sim = pd.DataFrame(correlated_returns, columns=['Stock_A', 'Stock_B'])
df_sim.index = pd.bdate_range(start="2023-01-01", periods=num_days)

print("Simulated Stock Returns:")
print(df_sim.head())

plt.figure(figsize=(12, 6))
plt.plot(df_sim.index, df_sim['Stock_A'], label="Stock A", color="blue", alpha=0.8)
plt.plot(df_sim.index, df_sim['Stock_B'], label="Stock B", color="red", alpha=0.8)
plt.title("Simulated Stock Returns (Correlation = 0.7)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True, linestyle='--', color='grey', linewidth=0.5)
plt.savefig('show.png')
plt.show()


# --------------------------------------------------------------
# real_GDP_growth.xlsx analysis
# --------------------------------------------------------------

gdp = pd.read_excel('real_GDP_growth.xlsx')
gdp = gdp.rename(columns={gdp.columns[0]: 'country'})
gdp = gdp.replace('no data', pd.NA)
gdp = pd.melt(gdp, id_vars='country', var_name='year', value_name='gdp_growth')
gdp['year'] = pd.to_numeric(gdp['year'], errors='coerce')
gdp['gdp_growth'] = pd.to_numeric(gdp['gdp_growth'], errors='coerce')
gdp = gdp.dropna(subset=['year', 'gdp_growth'])
gdp = gdp.sort_values(['country', 'year']).reset_index(drop=True)
gdp = gdp[gdp['country'] != 'West Bank and Gaza']
gdp['avg_growth'] = gdp.groupby('country')['gdp_growth'].transform('mean')

two_countries = gdp[gdp['country'].isin(['Germany', 'France'])]
plot_data = two_countries.pivot(index='year', columns='country', values='gdp_growth')

plt.figure(figsize=(12, 6))
plt.plot(plot_data.index, plot_data['Germany'], label='Germany', color='black')
plt.plot(plot_data.index, plot_data['France'], label='France', color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
plt.title('Real GDP Growth: Germany vs France')
plt.xlabel('Year')
plt.ylabel('GDP Growth (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()