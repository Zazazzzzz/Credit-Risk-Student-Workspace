# ---------------------------------------------------------------
# Script Name: GDP analyse
# Author: Hongyi Shen
# Description: Section 2_Assignment
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load data
df_RGDPG = pd.read_excel(os.path.join('python basics', 'real_GDP_growth.xlsx'))

# 2. Clean: keep only rows with countries and valid years
# pd.set_option('future.no_silent_downcasting', True) # opt into the future behavior
df_RGDPG = df_RGDPG.iloc[1:197, :]
df_RGDPG = df_RGDPG.replace('no data', np.nan)
# df_RGDPG = df_RGDPG.replace('no data', np.nan).infer_objects(copy=False) #  retain the old behavior

# 3. Rename column
df_RGDPG.rename(columns={'Real GDP growth (Annual percent change)': 'country'}, inplace=True)

# 4. Reshape from wide to long format
df_RGDPG_long = pd.melt(df_RGDPG,
                        id_vars='country', # column to keep as identifier
                        # value_vars=[],       # column(s) to unpivot
                        var_name='year',   # name of the new column indicating variable names
                        value_name='Real GDP growth (Annual percent change)') # name of the new column holding values

# 5. Sort and remove unwanted countries
df_RGDPG_long.sort_values(by=['country', 'year'], inplace=True)
df_RGDPG_long = df_RGDPG_long[df_RGDPG_long['country'] != 'West Bank and Gaza']

# 6. Convert year and value to proper types
df_RGDPG_long['year'] = pd.to_numeric(df_RGDPG_long['year'], errors='coerce')
# pd.to_numeric(...): This is a pandas function that tries to convert values to numbers.
# errors='': It tells pandas what to do if it encounters a value that cannot be converted to a number
# 'coerce': Forces any invalid values to become NaN (missing values).
df_RGDPG_long['Real GDP growth (Annual percent change)'] = pd.to_numeric(
    df_RGDPG_long['Real GDP growth (Annual percent change)'], errors='coerce'
)

# 7. Calculate average GDP growth for each country using transform
df_RGDPG_long['avg_growth_country'] = df_RGDPG_long.groupby('country')['Real GDP growth (Annual percent change)'].transform('mean')
# df_RGDPG_long.groupby('country')
# This tells pandas: “Group the DataFrame by the country column.”
# Each group now contains all the rows for a specific country
# transform('mean'): computes the mean for each group (country), returns a Series with the same length as the original DataFrame.
# It assigns to each row the mean GDP growth of that row’s country.

# 8. Select two countries
countries = ['Germany', 'United States']
df_selected = df_RGDPG_long[df_RGDPG_long['country'].isin(countries)]
# df_RGDPG_long['country']
# This selects the country column of the DataFrame — i.e., all country names in the dataset.
# .isin(countries)
# This checks for each row: "Is this country in the list (or set, or Series) called countries?"
# The result is a Boolean Series — True for rows where the country is in the list, False otherwise.
# df_RGDPG_long[ ... ]
# This is how pandas filters rows: by passing a Boolean Series.

# [1,2,3].isin([1,2,3,4])
# pd.Series([1,2,3]).isin([1,2,3,4])

# 9. Calculate log change and volatility
# First sort again to ensure time order
df_selected.sort_values(by=['country', 'year'], inplace=True)

# Define GDP growth as percentage (e.g., 5% growth means a multiplier of 1.05)
df_selected['log_growth'] = np.log(1 + df_selected['Real GDP growth (Annual percent change)'] / 100)
# GDP_t = GDP_(t-1) * (1 + growth_rate)
# ⇒ ln(GDP_t) - ln(GDP_(t-1)) = ln(1 + growth_rate)
# df_selected['Real GDP growth (Annual percent change)'] is the annual percent change in GDP (e.g., 2.5%)
# Dividing by 100 turns it into a proportion (e.g., 0.025)
# Adding 1 shifts the base to apply the log return formula correctly

# 10. Calculate rolling volatility (optional)
# You can also calculate volatility over the entire period
volatility = df_selected.groupby('country')['log_growth'].std().reset_index(name='volatility')
mean = df_selected.groupby('country')['log_growth'].mean().reset_index(name='mean')
# groupby('country')['log_growth'].std()
# This gives you a Pandas Series where: The index is the country.
# The values are the standard deviations of log_growth for each country.
# .reset_index(name='volatility')
# Turns the Series into a DataFrame.
# Moves the index (which is 'country') back into a regular column.
# Names the column of values (the standard deviations) as 'volatility'.

# 11. Stack countries into one DataFrame, then pivot to wide for comparison, year as the index, countries as the column
# - Set 'year' as the index (rows will represent each year)
# - Columns will be the different countries
# - Values will be the 'log_growth' for each country in each year
df_wide = df_selected.pivot(index='country', columns='year', values='log_growth') # to get the previous structure
df_wide = df_selected.pivot(index='year', columns='country', values='log_growth')
print(df_wide.head())

# 12. Plot the trend of these two countries
# Create a figure and axes
plt.figure(figsize=(10, 6))  # define the figure size
# This function creates a new figure where you can plot things.
# You can think of it as a blank canvas
# where all your subsequent plotting commands (like plt.plot(), plt.scatter(), etc.) will draw on.

plt.ion()  # Enable interactive mode

# plot each country's data
plt.plot(df_wide.index, df_wide['Germany'], label='Germany', marker='o')
plt.pause(1)
plt.plot(df_wide.index, df_wide['United States'], label='United States', marker='s')
plt.pause(1)

# customize plot
plt.title('Real GDP Growth (%): Germany vs United States')
plt.pause(1)
plt.xlabel('Year')
plt.pause(1)
plt.ylabel('Real GDP Growth (%)')
plt.pause(1)
plt.legend()
plt.pause(1)
plt.grid(True, axis='y', linestyle='--')
plt.pause(1)
plt.tight_layout()
plt.pause(1)

# show the plot
plt.show()
plt.close()

# to plot
plt.figure(figsize=(10, 6))
plt.plot(df_wide.index, df_wide['Germany'], label='Germany', marker='o')
plt.plot(df_wide.index, df_wide['United States'], label='United States', marker='s')
plt.title('Real GDP Growth (%): Germany vs United States')
plt.xlabel('Year')
plt.ylabel('Real GDP Growth (%)')
plt.legend()
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()

# show the plot
plt.show()

