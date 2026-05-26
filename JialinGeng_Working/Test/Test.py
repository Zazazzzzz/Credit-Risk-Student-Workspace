#attempt = 1
#num = input(f'Attempt {attempt}: Enter your guess: ')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load data
df_RGDPG = pd.read_excel(os.path.join(
    '/Users',
    'jialingeng',
    'Desktop',
    'Seminar-of-Credit-Risk',
    'Credit-Risk-Student-Workspace',
    'JialinGeng_Working',
    'Python_task',
    'real_GDP_growth.xlsx'
))
# os.path.join(...): 把括号里的一个个文件夹名称和文件名组合成一个完整的路径：/Users/jialingeng/Desktop/.../real_GDP_growth.xlsx。
# 不区分Mac/Linux 和 Wins

# 2. Clean: keep only rows with countries and valid years
# pd.set_option('future.no_silent_downcasting', True) # opt into the future behavior
df_RGDPG = df_RGDPG.iloc[1:197, :]   # .iloc[1:197, :]: 切片操作。意思是切取第 1 行到第 196 行（Python 左闭右开，不包含 197），: 代表保留所有列。这通常是为了剔除 Excel 中最上方的表头、说明文字或最底部的空白注脚。
df_RGDPG = df_RGDPG.replace('no data', np.nan)   # .replace('no data', np.nan): 把表格里所有字符串 'no data'（缺失值标记）替换成 Python 能识别的正式空值 np.nan（Not a Number）。这样做方便后续直接忽略或填充空值。
# df_RGDPG = df_RGDPG.replace('no data', np.nan).infer_objects(copy=False) #  retain the old behavior

# 3. Rename column
df_RGDPG.rename(columns={'Real GDP growth (Annual percent change)': 'country'}, inplace=True)

# 4. Reshape from wide to long format
df_RGDPG_long = pd.melt(df_RGDPG,
                        id_vars='country', # column to keep as identifier
                        # value_vars=[],       # column(s) to unpivot    （留空默认转换除 id_vars 外的所有列）
                        var_name='year',   # name of the new column indicating variable names
                        value_name='Real GDP growth (Annual percent change)') # name of the new column holding values

print(f' Reshape 变化 \n {df_RGDPG_long.head()}')

# 5. Sort and remove unwanted countries
df_RGDPG_long.sort_values(by=['country', 'year'], inplace=True)   # .sort_values(by=['country', 'year'], ...)：将数据先按照国家字母顺序（A-Z）排序，同一个国家内再按照年份先后顺序（从小到大）排序，确保时间序列的连续性。
df_RGDPG_long = df_RGDPG_long[df_RGDPG_long['country'] != 'West Bank and Gaza']
# df_RGDPG_long['country'] != 'West Bank and Gaza'：筛选出国家名称不等于 'West Bank and Gaza' 的所有行。
# 外面嵌套的 df_RGDPG_long[...]：应用这个条件，过滤掉该地区的数据。

print(f' Sort and remove unwanted countries \n {df_RGDPG_long.head()}')

# 6. Convert year and value to proper types
df_RGDPG_long['year'] = pd.to_numeric(df_RGDPG_long['year'], errors='coerce')
# pd.to_numeric(...): This is a pandas function that tries to convert values to numbers.
# errors='': It tells pandas what to do if it encounters a value that cannot be converted to a number
# 'coerce': Forces any invalid values to become NaN (missing values).
df_RGDPG_long['Real GDP growth (Annual percent change)'] = pd.to_numeric(
    df_RGDPG_long['Real GDP growth (Annual percent change)'], errors='coerce'
)

print(f' Convert year and value to proper types \n {df_RGDPG_long.head()}')

# 7. Calculate average GDP growth for each country using transform
df_RGDPG_long['avg_growth_country'] = df_RGDPG_long.groupby('country')['Real GDP growth (Annual percent change)'].transform('mean')
# df_RGDPG_long.groupby('country')
# This tells pandas: “Group the DataFrame by the country column.”
# Each group now contains all the rows for a specific country
# transform('mean'): computes the mean for each group (country), returns a Series with the same length as the original DataFrame.
# It assigns to each row the mean GDP growth of that row’s country.

print(f' Calculate average GDP growth for each country using transform \n {df_RGDPG_long.head()}')

# 8. Select two countries
countries = ['Germany', 'United States']
df_selected = df_RGDPG_long[df_RGDPG_long['country'].isin(countries)]

print(f' Select two countries \n {df_selected.head()}')

# 9. Calculate log change and volatility
# First sort again to ensure time order
df_selected.sort_values(by=['country', 'year'], inplace=True)
df_selected['log_growth'] = np.log(1 + df_selected['Real GDP growth (Annual percent change)'] / 100)


# 10. Calculate rolling volatility (optional)
# You can also calculate volatility over the entire period
volatility = df_selected.groupby('country')['log_growth'].std().reset_index(name='volatility')
mean = df_selected.groupby('country')['log_growth'].mean().reset_index(name='mean')

print(f' Calculate log change and volatility \n {df_selected.head()}')

print(volatility, mean)

df_wide = df_selected.pivot(index='country', columns='year', values='log_growth') # 恢复之前的结构
df_wide = df_selected.pivot(index='year', columns='country', values='log_growth')

print(df_wide.head())

