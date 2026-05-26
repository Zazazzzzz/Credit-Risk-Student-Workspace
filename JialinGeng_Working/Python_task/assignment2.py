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

# .rename(columns={...})：将原本名为 'Real GDP growth (Annual percent change)' 的列重命名为更简短的 'country'。
# inplace=True：表示直接修改原本的 df_RGDPG 数据框，而不需要重新赋值给一个新变量。

# 4. Reshape from wide to long format
df_RGDPG_long = pd.melt(df_RGDPG,
                        id_vars='country', # column to keep as identifier
                        # value_vars=[],       # column(s) to unpivot    （留空默认转换除 id_vars 外的所有列）
                        # 含义： “需要被转换的列”。
                        # 解释： 代码里把这一行注释掉了（没启用）。正如你代码注释中所写：如果不写这个参数，Pandas 会默认把除了 id_vars（即 country）以外的所有列，全部拿来转换。
                        var_name='year',   # name of the new column indicating variable names   含义： “原来那些‘列名’变成新的一列后，那一列叫什么名字？”
                        value_name='Real GDP growth (Annual percent change)') # name of the new column holding values   含义： “原来那些格子里的‘具体数值’变成新的一列后，那一列叫什么名字？”

# 5. Sort and remove unwanted countries
df_RGDPG_long.sort_values(by=['country', 'year'], inplace=True)   # .sort_values(by=['country', 'year'], ...)：将数据先按照国家字母顺序（A-Z）排序，同一个国家内再按照年份先后顺序（从小到大）排序，确保时间序列的连续性。
df_RGDPG_long = df_RGDPG_long[df_RGDPG_long['country'] != 'West Bank and Gaza']
# df_RGDPG_long['country'] != 'West Bank and Gaza'：筛选出国家名称不等于 'West Bank and Gaza' 的所有行。
# 外面嵌套的 df_RGDPG_long[...]：应用这个条件，过滤掉该地区的数据。


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
# transform 的特殊之处：如果一个国家在数据集中有 10 年的数据（也就是 10 行），.mean() 只会返回 1 个平均值（数据行数变少了）。而 transform 会把算出来的这 1 个平均值，复制 10 次，填满这 10 行。
# 效果：它保持了原始数据框的行数没有任何改变，非常方便直接拼接到原表上。


# 8. Select two countries
countries = ['Germany', 'United States']
df_selected = df_RGDPG_long[df_RGDPG_long['country'].isin(countries)]
# df_RGDPG_long['country']
# This selects the country column of the DataFrame — i.e., all country names in the dataset.
# .isin(countries)
# This checks for each row: "Is this country in the list (or set, or Series) called countries?"
# The result is a Boolean Series — True for rows where the country is in the list, False otherwise.
# df_RGDPG_long[ ... ]
# This is how pandas filters rows: by passing a Boolean Series.  这是 pandas 过滤行的方法：通过传入一个布尔型 Series 来筛选数据。

# [1,2,3].isin([1,2,3,4])   会直接报错（报错信息为 AttributeError），因为普通的 Python 列表根本没有 .isin() 这个方法。
# pd.Series([1,2,3]).isin([1,2,3,4])   是正确的 Pandas 语法，它会返回一个布尔值序列（Series）。


# 9. Calculate log change and volatility
# First sort again to ensure time order
df_selected.sort_values(by=['country', 'year'], inplace=True)

# Define GDP growth as percentage (e.g., 5% growth means a multiplier of 1.05)
df_selected['log_growth'] = np.log(1 + df_selected['Real GDP growth (Annual percent change)'] / 100)
# GDP_t = GDP_(t-1) * (1 + growth_rate)
# ⇒ ln(GDP_t) - ln(GDP_(t-1)) = ln(1 + growth_rate)
# df_selected['Real GDP growth (Annual percent change)'] is the annual percent change in GDP (e.g., 2.5%)
# Dividing by 100 turns it into a proportion (比例) (e.g., 0.025)
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

# 此时volatility和mean都是DataFrame 表格类型数据

# 11. Stack countries into one DataFrame, then pivot to wide for comparison, year as the index, countries as the column
# - Set 'year' as the index (rows will represent each year)
# - Columns will be the different countries
# - Values will be the 'log_growth' for each country in each year
df_wide = df_selected.pivot(index='country', columns='year', values='log_growth') # to get the previous structure
df_wide = df_selected.pivot(index='year', columns='country', values='log_growth')
print(df_wide.head())

# 这段代码展示了如何使用 Pandas 的 pivot() 函数在宽数据（Wide Format）的两种不同形态之间进行转换。
# 通俗来说，这两行代码的核心区别在于：谁当“行索引”（Index），谁当“列名”（Columns）。



# 12. Plot the trend of these two countries
# Create a figure and axes
plt.figure(figsize=(10, 6))  # define the figure size
# This function creates a new figure where you can plot things.
# You can think of it as a blank canvas
# where all your subsequent plotting commands (like plt.plot(), plt.scatter(), etc.) will draw on.

plt.ion()  # Enable interactive mode
           # 解释：ion 代表 Interactive mode ON（开启交互模式）。在默认的阻塞模式下，代码必须执行到 plt.show() 才会弹出图片。开启交互模式后，任何绘图命令都会实时更新并显示在屏幕上。

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

#plt.ioff()  # Turn off the interactive mode

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


