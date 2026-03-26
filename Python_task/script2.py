# ---------------------------------------------------------------
# Script Name: Functions and Simple Data Manipulations
# Author: Hongyi Shen
# Description: Section 2
# ----------------------------------------------------------------


### function
# def function_name(parameters):
#     # Code block
#     return result  # Optional

# without parameters
def schedule():
    print('8:00 - 10:00, Credit Risk')
# call the function
schedule()

# with parameters
def schedule_input(time: str, course: str):
    print(f'{time}, {course}')
schedule_input(time='8:00 - 10:00', course='Credit Risk')

# return: to send back a result from the function
# If no return statement is used, the function will return None by default.
def schedule_input_return(time: str, course: str):
    result = f'{time}, {course}'
    return result
s = schedule_input_return('8:00 - 10:00', 'Credit Risk')

### Task A: to calculate the Net Present Value (NPV) of a series of cash flows


def calculate_npv(cash_flows: list, discount_rate: float):
    npv = 0
    for t, cash_flows in enumerate(cash_flows):
        npv = npv + cash_flows/(1 + discount_rate) ** t
    print(f"Net Present Value (NPV): ${npv:.2f}")
    return npv

c = [100, 1000, 500, 300, 600, 800]
r = 0.01
v = calculate_npv(c, r)

### Task B: calculate the balance of a bank account and raise a concern when the expense is above the threshold
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

def balance(transactions: list, threshold: int or float, origin_balance: int or float, capacity: int or float):
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

# different ways to pass the arguments
account_a = balance(tran, thre, origin, cap)
account_b = balance(threshold=thre, origin_balance=origin, transactions=tran, capacity=cap)

### try, except, else, finally
# They allow you to handle errors gracefully, preventing your program from crashing when it encounters unexpected situations.
# try Block: Contains the code that might cause an exception.
# except Block: Contains the code that handles the exception if one occurs in the try block.
# else Block: Executes if no exceptions are raised in the try block
# finally Block: Executes no matter what, whether an exception was raised or not. It is often used for cleanup actions.

result = 10 / 0
try:
    result = 10 / 1
except Exception as e:
    print(f"Error: {type(e).__name__}.")
else:
    print(f"Result: {result}")
finally:
    print("Execution completed.")

### Task A:  to create a simple calculator that performs basic arithmetic operations: addition, subtraction, multiplication, and division.
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
            result = num1/num2
        else:
            raise ValueError("Invalid operation.")
        print(f"The result of {num1} {operation} {num2} is {result:.2f}.")

    except Exception as e:
        print(f'Error: {type(e).__name__}')

calculator()

### pandas is a powerful library in Python for data manipulation and analysis.
# It provides data structures and functions needed to efficiently handle and analyze data.
# https://pandas.pydata.org/docs/user_guide/index.html
# The primary data structures in Pandas are:
# Series: A one-dimensional labeled array.
# DataFrame: A two-dimensional labeled data structure with columns of potentially different types.

### NumPy is the fundamental package for scientific computing in Python.
# https://numpy.org/doc/stable/user/index.html
# It is a Python library that provides a multidimensional array object
# various derived objects (such as masked arrays and matrices)
# and an assortment of routines for fast operations on arrays
# including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
# At the core of the NumPy package, is the ndarray object.
# This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.

### Matplotlib is a powerful Python library used for creating static, animated, and interactive visualizations.
# https://matplotlib.org/stable/users/index
# Plot types: line plot, bar chart, scatter plot, histogram, box plot, etc.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a series
data = [10, 20, 30, 40, 50]
index = range(1,6)
series = pd.Series(data)
series_index = pd.Series(data, index)

# create a dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
index = range(1, 4)
df = pd.DataFrame(data)
df = pd.DataFrame(data, index)
# Accessing an entire column (returns a Series)
ages = df['Age']  # Get all values in the 'Age' column
# Accessing multiple columns (returns a DataFrame)
name_and_city = df[['Name', 'City']]  # Get 'Name' and 'City' columns
# Accessing a single row by index label (using loc)
row_1 = df.loc[1]  # Get the row where index == 1 (Alice's data)
# Accessing a single row by index position (using iloc)
first_row = df.iloc[0]  # Get the first row (Alice's data)

# Accessing a specific value using loc: row label and column name
Alice_age = df.loc[1, 'Age']  # Get Alice's age

# Accessing a specific value using iloc: position of row index and column index
Alice_age = df.iloc[0,1]
charlie_city = df.iloc[2, 2]  # Get Charlie's city

# Accessing a subset of rows and columns
subset = df.loc[1, ['Name', 'Age']]  # Get 'Name' and 'Age' for first row

# Conditional access: get all rows where Age > 25
older_than_25 = df[df['Age'] > 25] # sum(df['Age'] > 25)
Charlie_info = df[df['Name'] == 'Charlie']

# Create a second DataFrame
data2 = {
    'Name': ['Alice', 'Bob', 'David'],
    'Salary': [70000, 80000, 90000],
    'Department': ['HR', 'Engineering', 'Marketing']
}
df2 = pd.DataFrame(data2)

# Merge: Join df and df2 on 'Name'
merged_df = pd.merge(df, df2, on='Name', how='inner')
merged_df = pd.merge(df, df2, on='Name', how='left') # right
merged_df = pd.merge(df, df2, on='Name', how='outer')

# Join: Combine the dataframe by index
# Set 'Name' as index for both DataFrames to join on that
df_indexed = df.set_index('Name')
df2_indexed = df2.set_index('Name')

# Join the two DataFrames on the index ('Name')
joined_df = df_indexed.join(df2_indexed, how='inner')
joined_df.reset_index(inplace=True)

# Concat: Stack df and df2 vertically (must have same columns, so align them first)
# First align the columns by selecting common ones or renaming
df_common = df[['Name']]  # simulate some common structure
df2_common = df2[['Name']]
concat_df = pd.concat([df_common, df2_common], axis=0)
df.reset_index(drop=True, inplace=True)
concat_df_horizontal = pd.concat([df, df2], axis=1)

# Melt: Unpivot df to long format (e.g., make Age and City into variable/value pairs)
melted_df = pd.melt(df,
                    id_vars=['Name'],         # column to keep as identifier
                    value_vars=['Age'],       # column(s) to unpivot
                    var_name='Attribute',     # name of the new column indicating variable names
                    value_name='year old')    # name of the new column holding values

# Transpose
trans_df = df.T


# read the dataframe
df = pd.read_csv('prices.csv')
# Calculate daily returns
df["Daily_Return_percentage"] = df["Price"].pct_change()  # (p_t-p_{t-1})/p_{t-1}
df['Daily_Return'] = df['Price'] - df['Price'].shift(1)

# Calculate the 30-day log return
df['30_Day_Log_Return'] = np.log(df['Price'] / df['Price'].shift(30))
# df['Price']: pd.Series
# df['Price'].shift(30): pd.Series
# df['Price'] / df['Price'].shift(30): also pd.Series, division can be used element wise
# np.log(df['Price'] / df['Price'].shift(30)) : a series
# np.log() a numpy function, can be applied to pd.Series, it works element-wise

# Compute volatility (Rolling standard deviation)
# Rolling, here it looks like the current day and the 29 day before, 30 rows in total
df['Daily_Return_Volatility'] = df['Daily_Return_percentage'].rolling(window=30).std()
# df['Daily_Return_percentage']: a series
# df['Daily_Return'].rolling(window=30): a rollowing object
# .std(): a method of a rolling object
df['30_Day_Log_Return_Volatility'] = df['30_Day_Log_Return'].rolling(window=30).std()

df.set_index('Date', inplace=True) # Set the date as the index
# Plot the stock price
# Create a new figure with a size of 12 inches by 6 inches, become the current figure in the background
# the command that follows will be added to this figure by default
plt.figure(figsize=(12, 6))
# Plot the stock price with the date on the x-axis and price on the y-axis
plt.plot(df.index, df["Price"], label="Stock Price", color="blue")
# Add a title to the plot
plt.title("Stock Price Over Time")
# Take 4 evenly spaced dates from index
xticks = df.index[::len(df)//4]
# start:stop:step
# Since start and stop are left blank, it defaults to slicing from the beginning to the end of the index.
# step = len(df)//4 means picking every N-th element, where N is roughly one-fourth the length of the DataFrame.
plt.xticks(xticks)
# Label the x-axis
plt.xlabel("Date")
# Label the y-axis
plt.ylabel("Price")
# Display the legend to identify the plotted line
plt.legend()
# Add a grid with dashed grey lines to improve readability
plt.grid(True, linestyle='--', color='grey', linewidth=0.5)
# Save the figure to a file before showing it
plt.savefig('show.png')  # Save the plot as a PNG image named 'show.png'
# Display the plot
plt.show()
plt.close()

# Plot the three return series
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Daily_Return_percentage'], label='Daily Return', alpha=0.7)
plt.plot(df.index, df['30_Day_Log_Return'], label='30-Day Log Return', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.title("Stock Returns Over Time")
plt.xticks(xticks) # the ticks on x-axis
plt.xlabel("Date")
plt.ylabel("Returns")
plt.grid(True, axis='y', linestyle='--', color='grey', linewidth=0.5)
plt.show()


# Plot the volatility of returns
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


# numpy
# Create a 1D and 2D array
array_1d = np.array([10, 20, 30, 40, 50])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# SLICING
print("1D Array Slicing:")
print("Original:", array_1d)
print("First 3 elements:", array_1d[:3])       # [10 20 30]
print("Every second element:", array_1d[::2])  # [10 30 50] ,from:to:step


print("2D Array Slicing:")
print("Original:\n", array_2d)
print("First row:", array_2d[0])               # [1 2 3]
print("First column:", array_2d[:, 0])         # [1 4]
print("Element at (1,2):", array_2d[1, 2])      # 6


# ARITHMETIC OPERATIONS
# NumPy performs element-wise calculations for arithmetic operations on arrays of the same shape.
print("Add 10 to every element:\n", array_2d + 10)
other_1d = np.array([1, 2, 3, 4, 5])
print("Arithmetic with 1D arrays:")
print("Original:", array_1d)
print("Other:", other_1d)
print("Addition:", array_1d + other_1d)         # [11 22 33 44 55]
print("Multiplication:", array_1d * other_1d)   # [10 40 90 160 250]

other_2d = np.array([[10, 10, 10],
                     [1, 1, 1]])
print("Arithmetic with 2D arrays:")
print("Original:\n", array_2d)
print("Other:\n", other_2d)
print("Multiplication:\n", array_2d * other_2d)

# Define two 2D arrays (matrices)
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Matrix Multiplication
matrix_product = A @ B
print("Matrix Product (A @ B):\n", matrix_product)
matrix_product2 = np.matmul(A, B)
print("\nMatrix Product (np.matmul()):\n", matrix_product2)
# A * B



### np.random: a module in NumPy for generating random numbers and simulating statistical distributions

np.random.seed(42)  # Set the seed for reproducibility (makes results repeatable)
np.random.uniform(0, 1, size=(2, 2))  # Generate a 2x2 array of random numbers from a uniform distribution between 0 and 1
np.random.uniform(0, 1, size=10)  # Generate a 1D array with 10 random numbers from a uniform distribution between 0 and 1
np.random.normal(0, 1, size=10)  # Generate 10 random numbers from a normal (Gaussian) distribution with mean 0 and standard deviation 1
# Simulate dependent random variables with a multivariate normal distribution
# np.zeros(2): mean vector [0, 0]; cov_matrix: 2x2 covariance matrix; size=10: generate 10 samples
volatility = 1  # Volatility
correlation = 0.5  # Correlation between the two samples
cov_matrix = np.array([[volatility**2, correlation * volatility**2],
                       [correlation * volatility**2, volatility**2]]) # cov(X, Y) = rho * vol_X * vol_Y
np.random.multivariate_normal(np.zeros(2), cov_matrix, size=10)
np.random.beta(1, 1, size=10)  # Generate 10 samples from a Beta distribution with shape parameters α=1, β=1
np.random.gamma(1, 1, size=10)  # Generate 10 samples from a Gamma distribution with shape=1 and scale=1
np.random.poisson(5, size=10)
np.random.standard_t(5, size=10)

#### To simulate two stock returns with a mean of 0, volatility of 1, and a correlation of 0.7
# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_days = 260  # Number of trading days in a year
mean = [0, 0]   # Mean returns for both stocks
volatility = 1  # Volatility (standard deviation) for both stocks
correlation = 0.7  # Correlation between the two stocks

# Covariance matrix
# cov(X, Y) = rho * vol_X * vol_Y
cov_matrix = np.array([[volatility**2, correlation * volatility**2],
                       [correlation * volatility**2, volatility**2]])

# Generate correlated returns using multivariate normal distribution
correlated_returns = np.random.multivariate_normal(mean, cov_matrix, size=num_days)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(correlated_returns, columns=['Stock_A', 'Stock_B'])

# Add a date index
df.index = pd.bdate_range(start="2023-01-01", periods=num_days)

# Display the first few rows
print("Simulated Stock Returns:")
print(df.head())

# Plot the simulated returns
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Stock_A'], label="Stock A", color="blue", alpha=0.8)
plt.plot(df.index, df['Stock_B'], label="Stock B", color="red", alpha=0.8)
plt.title("Simulated Stock Returns (Correlation = 0.7)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.grid(True, linestyle='--', color='grey', linewidth=0.5)
plt.savefig('show.png')

