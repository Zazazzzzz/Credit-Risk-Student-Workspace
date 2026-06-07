# 4. 金融学核心考点：计算预估绝对损失额，以便后续执行“敞口加权平均”
df_merged['EAD_x_LGD'] = df_merged['EAD'] * df_merged['lgd']
df_merged['EAD_x_basePD'] = df_merged['EAD'] * df_merged['base_pd_hint']

# 5. 纵向聚合去重 (Groupby Obligor)：消除由于多笔建账日期产生的相同债务人重复行
df_final = df_merged.groupby('entity_code').agg(
    EAD=('EAD', 'sum'),  # EAD 绝对加总求和
    EAD_x_LGD=('EAD_x_LGD', 'sum'),  # 分子加总
    EAD_x_basePD=('EAD_x_basePD', 'sum'),  # 分子加总
    industry=('industry', 'first')  # 行业属性相同，取第一个即可
).reset_index()

# 6. 还原符合巴塞尔协议监管逻辑的“敞口加权平均指标”
df_final['LGD'] = df_final['EAD_x_LGD'] / df_final['EAD']
df_final['base_pd_hint'] = df_final['EAD_x_basePD'] / df_final['EAD']

# 清理辅助列
df_final.drop(columns=['EAD_x_LGD', 'EAD_x_basePD'], inplace=True)