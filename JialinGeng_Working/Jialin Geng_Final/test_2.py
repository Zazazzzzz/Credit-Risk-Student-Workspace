import os
from datetime import date

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 全局配置 (Essential Requirement)
DIR_DATA = "./data"  # 存放原始CSV的目录
DIR_RESULTS = "./results"  # 存放清洗后数据、图片、表格的目录

# os.makedirs(DIR_RESULTS, exist_ok=True)

# =============================================================================
# PART 1: DATA CLEANING (数据清洗与整合)
# =============================================================================

def standardize_string_series(series):
    """
    标准化字符串序列：转大写、去除两端空格（解决作业要求的标准化实体代码和行业标签）
    """
    return series.astype(str).str.upper().str.strip()


def clean_date_series(series):
    """
    精准识别 YYYY-MM-DD、DD/MM/YYYY 以及 DD-MM-YYYY 的混合日期列
    """

    def parse_single_date(date_val):
        # 1. 确保是字符串格式，并去除前后多余的空格
        date_str = str(date_val).strip()

        # 如果原本就是空值或异常文本，直接返回空日期
        if date_str in ['nan', 'None', 'NAT', '']:
            return pd.NaT

        # 2. 核心第一步：统一将 '/' 替换为 '-'
        date_str = date_str.replace('/', '-')

        try:
            # 3. 核心第二步：根据横杠拆分出 [前, 中, 后] 三部分
            parts = date_str.split('-')

            # 如果拆分出来的第一部分长度是 4，说明“年在前”（如 2023-01-02）
            if len(parts[0]) == 4:
                return pd.to_datetime(date_str, format='%Y-%m-%d')

            # 如果拆分出来的最后一部分长度是 4，说明“年在后”（如 03-01-2023 或 04-01-2023）
            elif len(parts[-1]) == 4:
                return pd.to_datetime(date_str, format='%d-%m-%Y')

            # 4. 兜底：如果拆分部分不符合，再尝试让 pandas 自动解析
            return pd.to_datetime(date_str, errors='coerce')
        except Exception:
            # 如果遇到无法解析的脏数据，返回空值（NaT）
            return pd.NaT

    # 💡 核心修复：使用 .apply() 将上面的单行判断逻辑，应用到这一整列的每一个单元格上
    parsed_series = series.apply(parse_single_date)

    # 5. 按照题目要求，统一转换为 'YYYY-MM-DD' 这种时间格式的字符串
    return parsed_series


#def clean_date_series(series):
    #"""
    #精准识别 YYYY-MM-DD 和 DD/MM/YYYY 的混合日期
    #"""
    # 1. 尝试用常规的 YYYY-MM-DD 格式解析（无法解析的会变成 NaT）
    #parsed_dates = pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')

    # 2. 找出那些解析失败的行，用 DD/MM/YYYY 格式再次解析并填补
    #is_na = parsed_dates.isna()
    #if is_na.any():
        # 显式指定 format='%d/%m/%Y'，确保 03/01/2023 被精准识别为 1月3日
        #slashed_dates = pd.to_datetime(series[is_na], format='%d/%m/%Y', errors='coerce')
        #parsed_dates.fillna(slashed_dates, inplace=True)

    #return parsed_dates


#
# def clean_date_series(series):
    #"""
    #确保日期格式一致为 YYYY-MM-DD
    #"""
    #return pd.to_datetime(series, errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')

    # format='mixed': 开启 Pandas 的自动混合格式解析功能
    # 作用：将标准日期对象重新格式化为特定格式的字符串。
    # 格式化占位符：
    # %Y：四位数的年份（例如：2026）
    # %m：两位数的月份，不足两位补零（01 至 12）
    # %d：两位数的日期，不足两位补零（01 至 31）
    # 结果：无论原始数据是 "2026/6/6"、"2026.06.06" 还是 "June 6, 2026"，经过这一步后，都会统一变成字符串 2026-06-06。



# 通用清洗方法
def clean_portfolio_data(file_name):
    # 读取数据：
    file_path = os.path.join(DIR_DATA, file_name)
    if not os.path.exists(file_path):
        print(f"[警告] 文件未找到: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 核心清洗步骤：
    # 1. 将明显的文本缺失（如 'NAN', 'NONE'）真正转换为 np.nan
    df = df.replace(["NAN", "NONE", r'^\s*$'], np.nan, regex=True)

    # 2. 统一日期格式
    if 'date' in df.columns:
        df['date'] = clean_date_series(df['date'])

    if 'booking_date' in df.columns:
        df['booking_date'] = clean_date_series(df['booking_date'])

    # 3. 统一大写、去除两端空格
    if 'entity_code' in df.columns:
        df['entity_code'] = standardize_string_series(df['entity_code'])

        # df['entity_code'] = df['entity_code'].astype(str).str.upper().str.strip()

    # 3. 修复可能缺失的横杠（如 BWMAEKB 修复为 B-WMAEKB
        df['entity_code'] = df['entity_code'].apply(lambda x: f"{x[0]}-{x[1:]}" if len(x) == 7 and x[1] != '-' else x)

    # 4.统一内部格式
    if 'industry' in df.columns:
        industry_map = {
            'INDUSTRY A': 'Industry A', 'IND. A': 'Industry A', 'INDUSTRYA': 'Industry A',
            'INDUSTRY B': 'Industry B', 'IND. B': 'Industry B', 'INDUSTRYB': 'Industry B', 'INDUSTRY b': 'Industry B'
        }
        df['industry'] = df['industry'].astype(str).str.upper().str.strip().map(industry_map)

    # 5. 构造新文件名（例如原名 data.csv -> 清洗后名为 cleaned_data.csv）
    output_file_name = "cleaned_" + file_name
    output_path = os.path.join(DIR_RESULTS, output_file_name)

    # 6. 输出为新的 CSV 文件
    # index=False 表示不把 pandas 的行索引（0, 1, 2...）写进 CSV 文件里
    # encoding='utf-8-sig' 可以防止中文或特殊字符在 Excel 中打开时出现乱码
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"数据清洗完毕，已成功保存至: {output_path}")

    # 7. 将清洗后的 DataFrame 返回，方便后续代码直接使用
    return df



def clean_entities_info():
    # 读取数据：
    file_path = os.path.join(DIR_DATA, "entities_info.csv")
    if not os.path.exists(file_path):
        print("[警告] 文件未找到: entities_info.csv")
        return None

    df = pd.read_csv(os.path.join(DIR_DATA, "entities_info.csv"))

    # 核心清洗步骤：
    # 1. 将明显的文本缺失（如 'NAN', 'NONE'）真正转换为 np.nan
    df = df.replace(["NAN", "NONE", r'^\s*$'], np.nan, regex=True)

    # 2. 统一大写、去除两端空格
    df['entity_code'] = standardize_string_series(df['entity_code'])

    # df['entity_code'] = df['entity_code'].astype(str).str.upper().str.strip()

    # 3. 统一行业标签：
    industry_map = {
        'INDUSTRY A': 'Industry A', 'IND. A': 'Industry A', 'INDUSTRYA': 'Industry A',
        'INDUSTRY B': 'Industry B', 'IND. B': 'Industry B', 'INDUSTRYB': 'Industry B', 'INDUSTRY b': 'Industry B'
    }
    df['industry'] = df['industry'].astype(str).str.upper().str.strip().replace(industry_map)     # 最后.replace(industry_map)也可替换为.map(industry_map)。注意两者区别。

    region_map = {
        'EUROPE': 'Europe', 'ASIA': 'Asia', 'SOUTHAMERICA': 'South America', 'NORTHAMERICA': 'North America'
    }
    df['region'] = df['region'].astype(str).str.upper().str.replace(" ", "", regex=False).replace(region_map)

    size_bucket_map = {
        'LARGE': 'Large', 'MID': 'Mid', 'SMALL': 'Small'
    }
    df['size_bucket'] = df['region'].astype(str).str.upper().str.replace(" ", "", regex=False).replace(size_bucket_map)

    # 4. 核心步骤：使用【行业 + 地区 + 规模】三维组合的平均值填充 PD (base_pd_hint)
    # transform(lambda x: x.fillna(x.mean())) 会动态计算当前所在小组的平均值并填充
    df["base_pd_hint"] = df.groupby(
        ["industry", "region", "size_bucket"]
    )["base_pd_hint"].transform(lambda x: x.fillna(x.mean()))
    # .transform() 是 Pandas 中非常强大的一个转换方法。它的核心特点是：保持数据的原有结构（行数）不变。
    # lambda 叫做匿名函数（也就是没有名字的临时函数）。
    # .fillna() 填充缺失值

    # 5. 核心步骤：使用【行业 + 地区 + 规模】三维组合的平均值填充 LGD(lgd)
    # transform(lambda x: x.fillna(x.mean())) 会动态计算当前所在小组的平均值并填充
    df["lgd"] = df.groupby(
        ["industry", "region", "size_bucket"]
    )["lgd"].transform(lambda x: x.fillna(x.mean()))


    # 处理LGD缺失值：用行业平均值或整体中位数进行合理填充.
    # median_lgd = df['lgd'].median()
    # df['lgd'] = df['lgd'].fillna(median_lgd)

    # 6. 构造新文件名
    output_file_name = "cleaned_" + 'entities_info.csv'
    output_path = os.path.join(DIR_RESULTS, output_file_name)

    # 7. 输出为新的 CSV 文件
    # index=False 表示不把 pandas 的行索引（0, 1, 2...）写进 CSV 文件里
    # encoding='utf-8-sig' 可以防止中文或特殊字符在 Excel 中打开时出现乱码
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"数据清洗完毕，已成功保存至: {output_path}")

    return df

# 后续编写核心合并逻辑，将Portfolio 1, 2 与 entities_info 以及收益率特征拼成 modeling_table.csv 存入 results/


# 生成初步整合数据表格，仅统一格式
cleaned_default_history_20y = clean_portfolio_data('default_history_20y.csv')
cleaned_industry_index_returns = clean_portfolio_data('industry_index_returns.csv')
cleaned_stock_returns = clean_portfolio_data('stock_returns.csv')
cleaned_portfolio_1 = clean_portfolio_data('portfolio_1.csv')
cleaned_portfolio_2 = clean_portfolio_data('portfolio_2.csv')
cleaned_portfolio_3 = clean_portfolio_data('portfolio_3.csv')
cleaned_entities_info = clean_entities_info()




def merge_portfolio(portfolio_id, entities_info):
    # 读取数据：
    file_path = os.path.join(DIR_RESULTS, portfolio_id)
    if not os.path.exists(file_path):
        print(f"[警告] 文件未找到: {file_path}")
        return None

    df_portfolio = pd.read_csv(file_path)

    file_path = os.path.join(DIR_RESULTS, entities_info)
    if not os.path.exists(file_path):
        print(f"[警告] 文件未找到: {file_path}")
        return None

    df_entities_info = pd.read_csv(file_path)

    # 合并表格
    # 横向左连接 (Left Join) 引入行业、LGD 和基础 PD 提示
    df_merged = pd.merge(df_portfolio, df_entities_info[['entity_code', 'industry', 'lgd', 'base_pd_hint']],
                         on='entity_code', how='left')

    df_merged.rename(columns={'base_pd_hint': 'PD', 'lgd': 'LGD'}, inplace=True)


    # 构造新文件名
    output_name = f"{portfolio_id}_merged.csv"
    output_path = os.path.join(DIR_RESULTS, 'merged_tables', output_name)

    # 输出为新的 CSV 文件
    df_merged.to_csv(output_path, index=False)

    print(f"[成功] 组合 {portfolio_id} & {entities_info}, 组合大表{output_name}已生成并保存至: {output_path}")


# 合并portfolio 与 entitles_info
portfolio_merged_1 = merge_portfolio('cleaned_portfolio_1.csv', 'cleaned_entities_info.csv')
portfolio_merged_2 = merge_portfolio('cleaned_portfolio_2.csv', 'cleaned_entities_info.csv')
portfolio_merged_3 = merge_portfolio('cleaned_portfolio_3.csv', 'cleaned_entities_info.csv')


def merge_stock_return_and_industry_index_returns(stock_returns, industry_index_returens):
    file_path = os.path.join(DIR_RESULTS, stock_returns)
    if not os.path.exists(file_path):
        print(f"[警告] 文件未找到: {file_path}")
        return None

    df_stock_returns = pd.read_csv(file_path)

    file_path = os.path.join(DIR_RESULTS, industry_index_returens)
    if not os.path.exists(file_path):
        print(f"[警告] 文件未找到: {file_path}")
        return None

    df_industry_index_returns = pd.read_csv(file_path)

    # 合并表格
    # 横向左连接 (Left Join) 引入行业和基industry_index 提示
    df_merged = pd.merge(df_stock_returns, df_industry_index_returns[['date', 'industry', 'index_return']],
                         on='date', how='left')



    # 构造新文件名
    output_name = "simulation.csv"
    output_path = os.path.join(DIR_RESULTS, 'merged_tables', output_name)

    # 输出为新的 CSV 文件
    df_merged.to_csv(output_path, index=False)

    print(f"[成功] 组合 {stock_returns} & {industry_index_returens}, 组合大表{output_name}已生成并保存至: {output_path}")


simulation = merge_stock_return_and_industry_index_returns('cleaned_stock_returns.csv', 'cleaned_industry_index_returns.csv')




# =============================================================================
# PART 2: PARAMETER ESTIMATION (参数估计：资产相关性 rho)
# =============================================================================


# portfolio_1
# ==========================================
# 1. 加载数据文件
# ==========================================
print("Portfolio_1的结果报告")
print("正在读取数据文件...")
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables', 'cleaned_portfolio_1.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_industry_index_returns.csv'))

# ==========================================
# 2. 参数估计：计算敏感度因子 (Factor Loading \rho_i)
# ==========================================
print(r"正在利用股票与行业指数历史收益率估计因子载荷 \rho_i...")

# 2.1 确定每个 entity_code 所属的行业 (根据 portfolio 中的 industry 列映射到股票收益率)
entity_to_industry = df_portfolio.set_index('entity_code')['industry'].to_dict()
df_stock_returns['industry'] = df_stock_returns['entity_code'].map(entity_to_industry)

# 2.2 合并个股收益率与行业指数收益率
df_merged_returns = pd.merge(
    df_stock_returns,
    df_industry_returns,
    on=['date', 'industry'],
    how='inner'
)

# 2.3 计算个股收益率与所属行业指数的相关系数（即 \rho_i）
rho_series = df_merged_returns.groupby('entity_code').apply(
    lambda x: x['stock_return'].corr(x['index_return']),
    include_groups=False
)
df_rho = rho_series.reset_index(name='rho')

# 2.4 将计算出的 \rho 合并到主资产组合表中
df_portfolio = pd.merge(df_portfolio, df_rho, on='entity_code', how='inner')

# 处理个别因历史数据不足导致相关性为 NaN 的样本，采用平均值填充
df_portfolio['rho'] = df_portfolio['rho'].fillna(df_portfolio['rho'].mean())

# ==========================================
# 3. 计算违约阈值 (Threshold)
# ==========================================
# 根据 Lecture 公式，违约阈值 d_i = \Phi^{-1}(PD_i)
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])

# 剔除资产负荷/敞口数据缺失的无效行
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD'])

m = len(df_portfolio)
print(f"数据处理与参数估计完成。有效建模债务人数量: {m}")

# ==========================================
# 4. 蒙特卡洛模拟组合损失 (Monte Carlo Simulation)
# ==========================================
print("正在进行蒙特卡洛模拟损失分布...")
N = 50000  # 模拟 50,000 种不同的宏观经济情景
np.random.seed(42)  # 设置随机种子以确保结果可复现

# 转换为 NumPy 数组以极大提升并行向量化计算效率
ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
rho_vec = df_portfolio['rho'].values
thresh_vec = df_portfolio['threshold'].values

# 步骤 A: 模拟系统性风险因子 F (标准正态分布，代表宏观大环境)
F = np.random.normal(0, 1, N)

# 步骤 B: 模拟个异性风险成分 \epsilon (大小为 N 情景 x m 债务人的矩阵)
epsilon = np.random.normal(0, 1, (N, m))

# 步骤 C: 根据单因子模型计算每个情景下每个主体的潜在信用变量 X
# 利用 np.newaxis 将 F 转换为矩阵，以便与各个主体的 rho 进行广播相乘
X = rho_vec * F[:, np.newaxis] + np.sqrt(1 - rho_vec**2) * epsilon

# 步骤 D: 判定是否违约 (若潜在变量 X <= 违约阈值，则记为 1，否则为 0)
default_matrix = (X <= thresh_vec).astype(int)

# 步骤 E: 计算组合经济损失
# 每家债务人在各情景下的损失 = 违约(0或1) * 风险敞口(EAD) * 违约损失率(LGD)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)  # 沿行方向加总，得到 N 个宏观情景下的资产组合总损失
portfolio_losses_millions = portfolio_losses / 1_000_000  #  在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

# ==========================================
# 5. 信用风险指标量化与研究分析
# ==========================================
print("\n" + "="*40)
print("     基于单因子模型的资产组合风险研究结果")
print("="*40)

# 1. 预期损失 (Expected Loss, EL)
EL = np.mean(portfolio_losses)
EL_m = EL / 1_000_000
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

# 2. 在险价值 (Value at Risk, VaR) - 设定置信水平为 99%
conf_level = 0.99
VaR_99 = np.percentile(portfolio_losses, conf_level * 100)
VaR_99_m = VaR_99 / 1_000_000
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

# 3. 预期不足 (Expected Shortfall, ES) - 衡量超过 VaR 的尾部平均损失
ES_99 = portfolio_losses[portfolio_losses >= VaR_99].mean()
ES_99_m = ES_99 / 1_000_000
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

# 4. 经济资本 (Economic Capital, EC) - 银行为抵御非预期损失需要配置的资本
EC = VaR_99 - EL
EC_m = EC / 1_000_000
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("="*40)

# ==========================================
# 6. 可视化损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='steelblue', alpha=0.4, label='Portfolio Loss')
sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='KDE Smooth Line')

plt.axvline(EL_m, color='darkblue', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title(' Simulated Loss Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions $ of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 7. 解析法 (Analytical Approach) 计算 VaR
# ==========================================
def calculate_vasicek_var(df, conf_level=0.99):
    """
    使用 Vasicek 单因子解析公式计算指定置信水平下的组合 VaR
    """
    # 计算总敞口，用于计算权重 w_i
    total_ead = df_portfolio['EAD'].sum()
    df_portfolio['weight'] = df_portfolio['EAD'] / total_ead

    # 映射公式中的各项参数
    pi = df_portfolio['PD'].values
    rho = df_portfolio['rho'].values
    lgd = df_portfolio['LGD'].values
    wi = df_portfolio['weight'].values

    # 核心数学映射：
    # 宏观因子 F 遵循标准正态分布。在 99% 的坏情景下，F 的分位数是 1% 分位数
    # 为了让公式里的符号和常见形式一致，我们直接计算标准正态分布在 (1 - conf_level) 处的逆函数
    # 99% 置信度对应 1% 极端糟糕的宏观环境：stats.norm.ppf(0.01) \approx -2.326
    # 对应你公式中分子的： \Phi^{-1}(p_i) - \rho_i * F
    f_bad = stats.norm.ppf(1 - conf_level)

    # 步骤 1：计算每个资产在 99% 坏情景下的【条件违约概率 p_i(F)】
    # 公式： \Phi( (\Phi^{-1}(p_i) - \rho_i * F) / \sqrt{1 - \rho_i^2} )
    numerator = stats.norm.ppf(pi) - rho * f_bad
    denominator = np.sqrt(1 - rho ** 2)
    conditional_pi = stats.norm.cdf(numerator / denominator)

    # 步骤 2：计算每个资产在极端情景下的损失贡献，并求和得到总条件损失率 L(F)
    # 公式： \sum (w_i * LGD_i * p_i(F))
    portfolio_loss_rate_var = np.sum(wi * lgd * conditional_pi)

    # 步骤 3：将损失率转换回绝对金额 ($)
    var_amount = portfolio_loss_rate_var * total_ead
    var_amount_m = var_amount / 1_000_000

    return var_amount_m, portfolio_loss_rate_var


# 执行计算
conf_level = 0.99
var_dollar, var_rate = calculate_vasicek_var(df_portfolio, conf_level)

# ==========================================
# 8. 打印解析法计算结果
# ==========================================
print("="*40)
print(f"   Vasicek 解析法计算结果 ({conf_level*100:.0f}% 置信度)")
print("="*40)
print(f"资产组合总敞口 (Total EAD) : ${df_portfolio['EAD'].sum() / 1_000_000 :,.2f}M")
print(f"组合条件损失率 (Loss Rate)  : {var_rate * 100:.2f}%")
print(f"在险价值 (Value at Risk, VaR): ${var_dollar:,.2f}M")
print("="*40)


# portfolio_2
# ==========================================
# 1. 加载数据文件
# ==========================================
print("Portfolio_2的结果报告")
print("正在读取数据文件...")
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables', 'cleaned_portfolio_2.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_industry_index_returns.csv'))

# ==========================================
# 2. 参数估计：计算敏感度因子 (Factor Loading \rho_i)
# ==========================================
print(r"正在利用股票与行业指数历史收益率估计因子载荷 \rho_i...")

# 2.1 确定每个 entity_code 所属的行业 (根据 portfolio 中的 industry 列映射到股票收益率)
entity_to_industry = df_portfolio.set_index('entity_code')['industry'].to_dict()
df_stock_returns['industry'] = df_stock_returns['entity_code'].map(entity_to_industry)

# 2.2 合并个股收益率与行业指数收益率
df_merged_returns = pd.merge(
    df_stock_returns,
    df_industry_returns,
    on=['date', 'industry'],
    how='inner'
)

# 2.3 计算个股收益率与所属行业指数的相关系数（即 \rho_i）
rho_series = df_merged_returns.groupby('entity_code').apply(
    lambda x: x['stock_return'].corr(x['index_return']),
    include_groups=False
)
df_rho = rho_series.reset_index(name='rho')

# 2.4 将计算出的 \rho 合并到主资产组合表中
df_portfolio = pd.merge(df_portfolio, df_rho, on='entity_code', how='inner')

# 处理个别因历史数据不足导致相关性为 NaN 的样本，采用平均值填充
df_portfolio['rho'] = df_portfolio['rho'].fillna(df_portfolio['rho'].mean())

# ==========================================
# 3. 计算违约阈值 (Threshold)
# ==========================================
# 根据 Lecture 公式，违约阈值 d_i = \Phi^{-1}(PD_i)
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])

# 剔除资产负荷/敞口数据缺失的无效行
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD'])

m = len(df_portfolio)
print(f"数据处理与参数估计完成。有效建模债务人数量: {m}")

# ==========================================
# 4. 蒙特卡洛模拟组合损失 (Monte Carlo Simulation)
# ==========================================
print("正在进行蒙特卡洛模拟损失分布...")
N = 50000  # 模拟 50,000 种不同的宏观经济情景
np.random.seed(42)  # 设置随机种子以确保结果可复现

# 转换为 NumPy 数组以极大提升并行向量化计算效率
ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
rho_vec = df_portfolio['rho'].values
thresh_vec = df_portfolio['threshold'].values

# 步骤 A: 模拟系统性风险因子 F (标准正态分布，代表宏观大环境)
F = np.random.normal(0, 1, N)

# 步骤 B: 模拟个异性风险成分 \epsilon (大小为 N 情景 x m 债务人的矩阵)
epsilon = np.random.normal(0, 1, (N, m))

# 步骤 C: 根据单因子模型计算每个情景下每个主体的潜在信用变量 X
# 利用 np.newaxis 将 F 转换为矩阵，以便与各个主体的 rho 进行广播相乘
X = rho_vec * F[:, np.newaxis] + np.sqrt(1 - rho_vec**2) * epsilon

# 步骤 D: 判定是否违约 (若潜在变量 X <= 违约阈值，则记为 1，否则为 0)
default_matrix = (X <= thresh_vec).astype(int)

# 步骤 E: 计算组合经济损失
# 每家债务人在各情景下的损失 = 违约(0或1) * 风险敞口(EAD) * 违约损失率(LGD)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)  # 沿行方向加总，得到 N 个宏观情景下的资产组合总损失
portfolio_losses_millions = portfolio_losses / 1_000_000  #  在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

# ==========================================
# 5. 信用风险指标量化与研究分析
# ==========================================
print("\n" + "="*40)
print("     基于单因子模型的资产组合风险研究结果")
print("="*40)

# 1. 预期损失 (Expected Loss, EL)
EL = np.mean(portfolio_losses)
EL_m = EL / 1_000_000
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

# 2. 在险价值 (Value at Risk, VaR) - 设定置信水平为 99%
conf_level = 0.99
VaR_99 = np.percentile(portfolio_losses, conf_level * 100)
VaR_99_m = VaR_99 / 1_000_000
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

# 3. 预期不足 (Expected Shortfall, ES) - 衡量超过 VaR 的尾部平均损失
ES_99 = portfolio_losses[portfolio_losses >= VaR_99].mean()
ES_99_m = ES_99 / 1_000_000
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

# 4. 经济资本 (Economic Capital, EC) - 银行为抵御非预期损失需要配置的资本
EC = VaR_99 - EL
EC_m = EC / 1_000_000
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("="*40)

# ==========================================
# 6. 可视化损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='steelblue', alpha=0.4, label='Portfolio Loss')

sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='KDE Smooth Line')

plt.axvline(EL_m, color='darkblue', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title(' Simulated Loss Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions $ of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 7. 解析法 (Analytical Approach) 计算 VaR
# ==========================================
def calculate_vasicek_var(df, conf_level=0.99):
    """
    使用 Vasicek 单因子解析公式计算指定置信水平下的组合 VaR
    """
    # 计算总敞口，用于计算权重 w_i
    total_ead = df_portfolio['EAD'].sum()
    df_portfolio['weight'] = df_portfolio['EAD'] / total_ead

    # 映射公式中的各项参数
    pi = df_portfolio['PD'].values
    rho = df_portfolio['rho'].values
    lgd = df_portfolio['LGD'].values
    wi = df_portfolio['weight'].values

    # 核心数学映射：
    # 宏观因子 F 遵循标准正态分布。在 99% 的坏情景下，F 的分位数是 1% 分位数
    # 为了让公式里的符号和常见形式一致，我们直接计算标准正态分布在 (1 - conf_level) 处的逆函数
    # 99% 置信度对应 1% 极端糟糕的宏观环境：stats.norm.ppf(0.01) \approx -2.326
    # 对应你公式中分子的： \Phi^{-1}(p_i) - \rho_i * F
    f_bad = stats.norm.ppf(1 - conf_level)

    # 步骤 1：计算每个资产在 99% 坏情景下的【条件违约概率 p_i(F)】
    # 公式： \Phi( (\Phi^{-1}(p_i) - \rho_i * F) / \sqrt{1 - \rho_i^2} )
    numerator = stats.norm.ppf(pi) - rho * f_bad
    denominator = np.sqrt(1 - rho ** 2)
    conditional_pi = stats.norm.cdf(numerator / denominator)

    # 步骤 2：计算每个资产在极端情景下的损失贡献，并求和得到总条件损失率 L(F)
    # 公式： \sum (w_i * LGD_i * p_i(F))
    portfolio_loss_rate_var = np.sum(wi * lgd * conditional_pi)


    # 步骤 3：将损失率转换回绝对金额 ($)
    var_amount = portfolio_loss_rate_var * total_ead
    var_amount_m = var_amount / 1_000_000

    return var_amount_m, portfolio_loss_rate_var


# 执行计算
conf_level = 0.99
var_dollar, var_rate = calculate_vasicek_var(df_portfolio, conf_level)

# ==========================================
# 8. 打印解析法计算结果
# ==========================================
print("="*40)
print(f"   Vasicek 解析法计算结果 ({conf_level*100:.0f}% 置信度)")
print("="*40)
print(f"资产组合总敞口 (Total EAD) : ${df_portfolio['EAD'].sum() / 1_000_000 :,.2f}M")
print(f"组合条件损失率 (Loss Rate)  : {var_rate * 100:.2f}%")
print(f"在险价值 (Value at Risk, VaR): ${var_dollar:,.2f}M")
print("="*40)







# 单因子潜在变量模型 (One-Factor Model)  portfolio 1 and portfolio 2


