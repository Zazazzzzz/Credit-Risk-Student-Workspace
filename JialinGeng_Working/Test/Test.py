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
print(df_RGDPG_long.head())