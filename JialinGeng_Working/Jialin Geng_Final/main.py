import os
from datetime import date

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import PercentFormatter

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
                return pd.to_datetime(date_str, format='%m-%d-%Y')

            # 4. 兜底：如果拆分部分不符合，再尝试让 pandas 自动解析
            return pd.to_datetime(date_str, errors='coerce')
        except Exception:
            # 如果遇到无法解析的脏数据，返回空值（NaT）
            return pd.NaT

    # 💡 核心修复：使用 .apply() 将上面的单行判断逻辑，应用到这一整列的每一个单元格上
    parsed_series = series.apply(parse_single_date)

    # 5. 按照题目要求，统一转换为 'YYYY-MM-DD' 这种时间格式的字符串
    return parsed_series


# def clean_date_series(series):
# """
# 精准识别 YYYY-MM-DD 和 DD/MM/YYYY 的混合日期
# """
# 1. 尝试用常规的 YYYY-MM-DD 格式解析（无法解析的会变成 NaT）
# parsed_dates = pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')

# 2. 找出那些解析失败的行，用 DD/MM/YYYY 格式再次解析并填补
# is_na = parsed_dates.isna()
# if is_na.any():
# 显式指定 format='%d/%m/%Y'，确保 03/01/2023 被精准识别为 1月3日
# slashed_dates = pd.to_datetime(series[is_na], format='%d/%m/%Y', errors='coerce')
# parsed_dates.fillna(slashed_dates, inplace=True)

# return parsed_dates


#
# def clean_date_series(series):
# """
# 确保日期格式一致为 YYYY-MM-DD
# """
# return pd.to_datetime(series, errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')

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
    df['industry'] = df['industry'].astype(str).str.upper().str.strip().replace(
        industry_map)  # 最后.replace(industry_map)也可替换为.map(industry_map)。注意两者区别。

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

    print(
        f"[成功] 组合 {stock_returns} & {industry_index_returens}, 组合大表{output_name}已生成并保存至: {output_path}")


simulation = merge_stock_return_and_industry_index_returns('cleaned_stock_returns.csv',
                                                           'cleaned_industry_index_returns.csv')



# =============================================================================
# PART 2: PARAMETER ESTIMATION (参数估计：资产相关性 rho)
# =============================================================================


## Factor_Model

## 仅有industry一个系统因素 ##

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
X = rho_vec * F[:, np.newaxis] + np.sqrt(1 - rho_vec ** 2) * epsilon

# 步骤 D: 判定是否违约 (若潜在变量 X <= 违约阈值，则记为 1，否则为 0)
default_matrix = (X <= thresh_vec).astype(int)

# 步骤 E: 计算组合经济损失
# 每家债务人在各情景下的损失 = 违约(0或1) * 风险敞口(EAD) * 违约损失率(LGD)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)  # 沿行方向加总，得到 N 个宏观情景下的资产组合总损失
portfolio_losses_millions = portfolio_losses / 1_000_000  # 在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

losses_p1 = portfolio_losses_millions
# ==========================================
# 5. 信用风险指标量化与研究分析
# ==========================================
print("\n" + "=" * 40)
print("     基于单因子模型的资产组合风险研究结果")
print("=" * 40)

# 1. 预期损失 (Expected Loss, EL)
EL = np.mean(portfolio_losses)
EL_m = EL / 1_000_000
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

# 2. 在险价值 (Value at Risk, VaR) - 设定置信水平为 99%
conf_level = 0.99
VaR_99 = np.percentile(portfolio_losses, conf_level * 100)
VaR_99_m = VaR_99 / 1_000_000
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

single_p1_VaR_99_m = VaR_99_m

# 3. 预期不足 (Expected Shortfall, ES) - 衡量超过 VaR 的尾部平均损失
ES_99 = portfolio_losses[portfolio_losses >= VaR_99].mean()
ES_99_m = ES_99 / 1_000_000
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

# 4. 经济资本 (Economic Capital, EC) - 银行为抵御非预期损失需要配置的资本
EC = VaR_99 - EL
EC_m = EC / 1_000_000
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("=" * 40)

# ==========================================
# 6. 可视化损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='steelblue', alpha=0.4, label='Portfolio Loss')

sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='One-Factor KDE Curve')

plt.axvline(EL_m, color='darkblue', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title(' One-Industry Portfolio Simulated Loss Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, color='gainsboro', alpha=0.5)  # 使用高级的亮灰色背景网格线
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

analytical_p1_var = var_dollar

# ==========================================
# 8. 打印解析法计算结果
# ==========================================
print("=" * 40)
print(f"   Vasicek 解析法计算结果 ({conf_level * 100:.0f}% 置信度)")
print("=" * 40)
print(f"资产组合总敞口 (Total EAD) : ${df_portfolio['EAD'].sum() / 1_000_000 :,.2f}M")
print(f"组合条件损失率 (Loss Rate)  : {var_rate * 100:.2f}%")
print(f"在险价值 (Value at Risk, VaR): ${var_dollar:,.2f}M")
print("=" * 40)

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
X = rho_vec * F[:, np.newaxis] + np.sqrt(1 - rho_vec ** 2) * epsilon

# 步骤 D: 判定是否违约 (若潜在变量 X <= 违约阈值，则记为 1，否则为 0)
default_matrix = (X <= thresh_vec).astype(int)

# 步骤 E: 计算组合经济损失
# 每家债务人在各情景下的损失 = 违约(0或1) * 风险敞口(EAD) * 违约损失率(LGD)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)  # 沿行方向加总，得到 N 个宏观情景下的资产组合总损失
portfolio_losses_millions = portfolio_losses / 1_000_000  # 在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

losses_p2 = portfolio_losses_millions

# ==========================================
# 5. 信用风险指标量化与研究分析
# ==========================================
print("\n" + "=" * 40)
print("     基于单因子模型的资产组合风险研究结果")
print("=" * 40)

# 1. 预期损失 (Expected Loss, EL)
EL = np.mean(portfolio_losses)
EL_m = EL / 1_000_000
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

# 2. 在险价值 (Value at Risk, VaR) - 设定置信水平为 99%
conf_level = 0.99
VaR_99 = np.percentile(portfolio_losses, conf_level * 100)
VaR_99_m = VaR_99 / 1_000_000
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

single_p2_VaR_99_m = VaR_99_m

# 3. 预期不足 (Expected Shortfall, ES) - 衡量超过 VaR 的尾部平均损失
ES_99 = portfolio_losses[portfolio_losses >= VaR_99].mean()
ES_99_m = ES_99 / 1_000_000
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

# 4. 经济资本 (Economic Capital, EC) - 银行为抵御非预期损失需要配置的资本
EC = VaR_99 - EL
EC_m = EC / 1_000_000
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("=" * 40)

# ==========================================
# 6. 可视化损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='steelblue', alpha=0.4, label='Portfolio Loss')

sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='One-Factor KDE Curve')

plt.axvline(EL_m, color='darkblue', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title(' One-Industry Portfolio Simulated Loss Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, color='gainsboro', alpha=0.5)  # 使用高级的亮灰色背景网格线
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

analytical_p2_var = var_dollar

# ==========================================
# 8. 打印解析法计算结果
# ==========================================
print("=" * 40)
print(f"   Vasicek 解析法计算结果 ({conf_level * 100:.0f}% 置信度)")
print("=" * 40)
print(f"资产组合总敞口 (Total EAD) : ${df_portfolio['EAD'].sum() / 1_000_000 :,.2f}M")
print(f"组合条件损失率 (Loss Rate)  : {var_rate * 100:.2f}%")
print(f"在险价值 (Value at Risk, VaR): ${var_dollar:,.2f}M")
print("=" * 40)





# =============================================================================
# PART 2: MULTI-FACTOR PARAMETER ESTIMATION (多因子参数估计)
# =============================================================================


# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载最新多行业数据集...")
# 读取你上传的三个最新清洗后的文件
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables','cleaned_portfolio_3.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS,'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_industry_index_returns.csv'))

# 安全清洗：删除核心风控指标（EAD, LGD, PD, industry）包含空值的行
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD', 'PD', 'industry'])

# ==========================================
# 2. 参数估计：计算行业内敏感度 \rho_i 与行业间相关性
# ==========================================
print("正在利用历史数据估计行业间相关性与各自的因子载荷 (Asset Correlation)...")

# 步骤 A：把行业收益率数据转化为宽表（计算 A 和 B 行业的历史相关系数）
df_ind_pivot = df_industry_returns.pivot_table(index='date', columns='industry', values='index_return').dropna()
rho_AB = df_ind_pivot.corr().loc['Industry A', 'Industry B']
print(f">> 经历史数据测算，Industry A 与 Industry B 的相关系数为: {rho_AB:.4f}")

# 步骤 B：精确匹配每个企业个股与对应行业的历史收益率，计算敏感度 \rho_i
rho_list = []
for idx, row in df_portfolio.iterrows():
    entity = row['entity_code']
    ind = row['industry']

    # 获取该个股的历史收益率
    single_stock = df_stock_returns[df_stock_returns['entity_code'] == entity][['date', 'stock_return']]
    # 获取它对应的行业历史收益率
    comp_ind = df_ind_pivot[[ind]].reset_index()

    # 按照日期合并
    merged = pd.merge(single_stock, comp_ind, on='date').dropna()

    # 如果历史样本充足（大于10天），计算相关系数；否则给定默认行业相关性
    if len(merged) > 10:
        rho_i = merged['stock_return'].corr(merged[ind])
        # 边界防呆保护，防止极个别相关性计算超出 [-1, 1] 范围
        rho_i = max(min(rho_i, 0.99), -0.99)
    else:
        rho_i = 0.20
    rho_list.append(rho_i)

df_portfolio['rho'] = rho_list

# ==========================================
# 3. 设定违约阈值
# ==========================================
# 利用各自企业自带的真实 PD，通过标准正态分布的逆累积分布函数计算违约阈值
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])
print(f"参数估计完成。组合内包含总债务人数量: {len(df_portfolio)}")

# ==========================================
# 4. 多因子蒙特卡洛模拟组合损失分布
# ==========================================
print("正在开始多行业联合蒙特卡洛模拟...")

m = len(df_portfolio)  # 债务人总数
N = 50000  # 升级为 5 万次模拟情景，结果更稳健
np.random.seed(42)  # 设置随机种子以确保结果可复现

# 转换为 NumPy 数组以提高计算效率
ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
rho_vec = df_portfolio['rho'].values
thresh_vec = df_portfolio['threshold'].values
industry_vec = df_portfolio['industry'].values

# 【核心升级步骤 A】：基于行业相关性矩阵，抽取“自带相关性”的未来行业因子对 (F_A, F_B)
# 构造协方差矩阵（由于均值为0方差为1，协方差矩阵直接等于相关系数矩阵）
cov_matrix = [[1.0, rho_AB],
              [rho_AB, 1.0]]

# 联合随机抽取：生成形状为 (50000, 2) 的系统风险矩阵
# 第一列是未来的 F_A，第二列是未来的 F_B
F_joint = np.random.multivariate_normal([0, 0], cov_matrix, size=N)
F_dict = {'Industry A': F_joint[:, 0], 'Industry B': F_joint[:, 1]}

# 将对应行业的未来因子对号入座分发给每个企业，生成映射矩阵 (N, m)
F_matrix = np.zeros((N, m))
for j, ind in enumerate(industry_vec):
    F_matrix[:, j] = F_dict[ind]

# 【核心升级步骤 B】：抽取特有风险成分 \epsilon (N x m)
epsilon = np.random.normal(0, 1, (N, m))

# 【核心升级步骤 C】：多因子潜在信用变量 X 计算公式：
# X_i = \rho_i * F_行业 + \sqrt{1 - \rho_i^2} * \epsilon_i
X = rho_vec * F_matrix + np.sqrt(1 - rho_vec ** 2) * epsilon

# 【核心升级步骤 D】：判定是否违约
default_matrix = (X <= thresh_vec).astype(int)

# 【核心升级步骤 E】：计算每个情景下的资产组合总损失金额（单位：元）
# 每个债务人的损失 = 违约指示 * EAD * LGD
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)  # 沿行加总，得到 N 个情景的损失数组

# 为了画图和阅读美观，将整体损失金额转换为“百万元 (Millions)”单位
portfolio_losses_millions = portfolio_losses / 1_000_000

losses_p3 = portfolio_losses_millions

# ==========================================
# 5. 风险度量与结果分析
# ==========================================
print("\n===== 跨行业多因子资产组合信用风险研究结果 =====")

# 1. 预期损失 (Expected Loss)
EL_m = np.mean(portfolio_losses_millions)
print(f"预期损失 (EL): {EL_m:.2f} M")

# 2. 99% 置信水平下的在险价值 (VaR 99%)
VaR_99_m = np.percentile(portfolio_losses_millions, 99)
print(f"99% 在险价值 (VaR 99%): {VaR_99_m:.2f} M")

single_p3_VaR_99_m = VaR_99_m

# 3. 99% 置信水平下的预期不足 (Expected Shortfall / Tail VaR)
ES_99_m = portfolio_losses_millions[portfolio_losses_millions >= VaR_99_m].mean()
print(f"99% 预期不足 (ES 99%): {ES_99_m:.2f} M")

# 4. 经济资本 (Economic Capital) = 非预期损失 (Unexpected Loss)
EC_m = VaR_99_m - EL_m
print(f"应配置的经济资本 (EC): {EC_m:.2f} M")

# ==========================================
# 6. 可视化：绘制组合损失分布图（直方图 + KDE 红色连贯曲线完美融合）
# ==========================================
plt.figure(figsize=(11, 5.5))

# 安全清洗：过滤掉计算中可能产生的任何 NaN 极值，确保渲染不报错
clean_losses = portfolio_losses_millions[~np.isnan(portfolio_losses_millions)]

# 步骤一：使用 seaborn 绘制半透明的钢蓝色直方图
sns.histplot(clean_losses, bins=150, stat='density',
             color='steelblue', alpha=0.4, label='Simulated Loss Hist')

# 步骤二：叠加一条霸道的、绝对独立的royalblue色 KDE 平滑曲线（规避底层冲突）
sns.kdeplot(clean_losses, color='royalblue', linewidth=1, label='Multi-Factor KDE Curve')

# 步骤三：标出升级后的四大风控指标线（单位：百万元）
plt.axvline(EL_m, color='navy', linestyle='--', linewidth=1.5, label=f'EL: {EL_m:.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1.5, label=f'99% VaR: {VaR_99_m:.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1.5, label=f'99% ES: {ES_99_m:.2f}M')

# 通过 99.5% 分位数自动截短无人的横坐标，让左边的主峰和曲线纤毫毕现
# plt.xlim(0, np.percentile(clean_losses, 99.5))

# 消除讨厌的科学计数法，将纵轴优雅地转化为百分比密度 (%)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=4))

plt.title('Multi-Industry Portfolio Simulated Loss Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)


plt.legend(loc='upper right', fontsize=10)
plt.grid(True, color='gainsboro', alpha=0.5)  # 使用高级的亮灰色背景网格线
plt.tight_layout()
plt.show()

'''
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

    return var_amount, portfolio_loss_rate_var


# 执行计算
conf_level = 0.99
var_dollar, var_rate = calculate_vasicek_var(df_portfolio, conf_level)

# ==========================================
# 8. 打印解析法计算结果
# ==========================================
print("="*40)
print(f"   Vasicek 解析法计算结果 ({conf_level*100:.0f}% 置信度)")
print("="*40)
print(f"资产组合总敞口 (Total EAD) : ${df_portfolio['EAD'].sum():,.2f}")
print(f"组合条件损失率 (Loss Rate)  : {var_rate * 100:.2f}%")
print(f"在险价值 (Value at Risk, VaR): ${var_dollar:,.2f}")
print("="*40)
'''

# ==============================================
# 将portfolio1,2,3合并一起研究
# ==============================================

# 1. 组装数据字典 (请确保 losses_p1, losses_p2, losses_p3 已经在前文中被赋值)
portfolio_data = {
    'Portfolio 1 (Single-Ind)': {'data': losses_p1, 'color': 'royalblue', 'line': '-'},
    'Portfolio 2 (Single-Ind)': {'data': losses_p2, 'color': 'emerald', 'line': '--'},  # 可以替换为你喜欢的颜色
    'Portfolio 3 (Multi-Ind)': {'data': losses_p3, 'color': 'crimson', 'line': '-.'}
}

plt.figure(figsize=(12, 7))

# 2. 循环绘制每个组合的 KDE 曲线和核心指标
colors = ['#4F7CAC', '#2C9E4B', '#D62828']  # 优雅的学术风配色：钢蓝、内敛绿、朱红
styles = ['-', '--', '-.']

for i, (name, p_info) in enumerate(portfolio_data.items()):
    losses = p_info['data']
    color = colors[i]
    style = styles[i]

    # 计算核心指标
    el = np.mean(losses)
    var_99 = np.percentile(losses, 99)

    # 绘制平滑密度曲线 (去掉不透明的直方图，避免画面太满，改用 shade/fill)
    sns.kdeplot(losses, color=color, linestyle=style, linewidth=1, label=name, fill=True, alpha=0.05)

    # 标出各自的 EL 和 99% VaR
    plt.axvline(el, color=color, linestyle=':', linewidth=0.6, alpha=0.8)
    plt.axvline(var_99, color=color, linestyle=style, linewidth=0.6, alpha=0.9)

    # 在图形上方动态添加文本标注，避免线堆在一起分不清
    plt.text(var_99, plt.gca().get_ylim()[1] * (0.85 - i * 0.08), f"{name} 99% VaR: ${var_99:.2f}M",
             color=color, fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 3. 美化图表
plt.title('Comparison of Simulated Loss Distributions (Portfolio 1, 2 & 3)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)

plt.grid(True, color='gainsboro', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.show()



## 有多个系统因素  ##

# 1. 全局配置
DIR_RESULTS = "./results"  # 请根据你本地清洗后的实际路径调整

print("Portfolio_1 的多因子模型分析报告")
print("正在读取数据文件...")

# 读取 4 个核心 CSV 文件
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables', 'cleaned_portfolio_1.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS,'cleaned_industry_index_returns.csv'))
df_entities_info = pd.read_csv(os.path.join(DIR_RESULTS,'cleaned_entities_info.csv'))

# ==========================================
# 2. 参数估计：多元线性回归提取多因子载荷 (Beta) 与特异性权重 (Eta)
# ==========================================
print("正在对各行业系统性因子进行对齐并建立多因子矩阵...")

# 2.1 转换数据结构为时间序列矩阵
stock_pivot = df_stock_returns.pivot_table(
    index='date',
    columns='entity_code',
    values='stock_return',
    aggfunc='mean'
)
# 行业指数收益率也建议改用 pivot_table 规避可能重复的数据，并前向/后向填充缺失值，确保时序连续
industry_pivot = df_industry_returns.pivot_table(
    index='date',
    columns='industry',
    values='index_return',
    aggfunc='mean'
).ffill().bfill()

# 确保日期完全对齐
common_dates = stock_pivot.index.intersection(industry_pivot.index)
stock_pivot = stock_pivot.loc[common_dates]
industry_pivot = industry_pivot.loc[common_dates]

# 所有的行业因子名称
factor_names = list(industry_pivot.columns)
num_factors = len(factor_names)

# 2.2 估计每一只股票在全市场多因子上的暴露度
print("正在利用多元线性回归估计多因子敏感度系数...")
multi_factor_params = []

for code in df_portfolio['entity_code'].unique():
    if code in stock_pivot.columns:
        # 获取当前股票的收益率序列，并剔除其自身的 NaN（如停牌日）
        y_raw = stock_pivot[code].dropna()

        # 严格对齐：确保 X 和 y 使用完全相同的非空日期交集
        common_idx = y_raw.index.intersection(industry_pivot.index)

        if len(common_idx) > 30:  # 保证样本充足
            y = y_raw.loc[common_idx]
            X = industry_pivot.loc[common_idx, factor_names]

            # 【双重保险】再次剔除任何可能潜在的 NaN 行（sklearn 的严格要求）
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(y_clean) > 30:
                # 拟合多元线性回归
                reg = LinearRegression().fit(X_clean, y_clean)
                r_squared = reg.score(X_clean, y_clean)
                betas = reg.coef_

                param_dict = {
                    'entity_code': code,
                    'R_Squared': r_squared,
                    'eta': np.sqrt(max(0, 1 - r_squared))
                }
                for idx, factor in enumerate(factor_names):
                    param_dict[f'beta_{factor}'] = betas[idx]

                multi_factor_params.append(param_dict)

df_params = pd.DataFrame(multi_factor_params)

# 2.3 将因子参数以及 entities_info 中的丰富特征(如地域、规模)合并到主资产组合表中
df_portfolio = pd.merge(df_portfolio, df_params, on='entity_code', how='inner')
df_portfolio = pd.merge(df_portfolio, df_entities_info[['entity_code', 'region', 'size_bucket']], on='entity_code',
                        how='left')

# 填补可能缺失的多元回归参数
df_portfolio['R_Squared'] = df_portfolio['R_Squared'].fillna(df_portfolio['R_Squared'].mean())
df_portfolio['eta'] = df_portfolio['eta'].fillna(np.sqrt(1 - df_portfolio['R_Squared']))
for factor in factor_names:
    df_portfolio[f'beta_{factor}'] = df_portfolio[f'beta_{factor}'].fillna(0.0)

# ==========================================
# 3. 计算违约阈值 (Threshold)
# ==========================================
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD'])

m = len(df_portfolio)
print(f"多因子参数估计完成。用于多因子建模的债务人数量: {m}")

# ==========================================
# 4. 多因子蒙特卡洛模拟损失分布
# ==========================================
print("\n正在进行多因子框架下的蒙特卡洛模拟...")
N = 50000
np.random.seed(42)

ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
eta_vec = df_portfolio['eta'].values
thresh_vec = df_portfolio['threshold'].values

# 提取各资产对所有因子的 Beta 矩阵 (形状: m x num_factors)
beta_matrix = df_portfolio[[f'beta_{factor}' for factor in factor_names]].values

# 步骤 A: 模拟多个行业系统性风险因子矩阵 F
# 考虑到现实中各行业指数存在相关性，我们从历史数据中提取因子之间的协方差矩阵
factor_covariance = industry_pivot[factor_names].cov().values
factor_means = industry_pivot[factor_names].mean().values

# 从多元正态分布中抽取 N 个情景的系统性风险因子 (形状: N x num_factors)
F_multi = np.random.multivariate_normal(factor_means, factor_covariance, size=N)

# 步骤 B: 模拟特异性风险成分 epsilon (形状: N x m)
epsilon = np.random.normal(0, 1, (N, m))

# 步骤 C: 根据多因子公式计算每个情景下每个主体的潜在信用变量 X
# 1. 计算原始系统性部分 (N x m)
systematic_part_raw = np.dot(F_multi, beta_matrix.T)

# 2. 计算每个资产在历史/模拟状态下的系统性成分方差 (取每只股票在 N 个情景下的方差)
# 这是为了确保系统性方差 + 特异性方差 = 1
sys_variance = np.var(systematic_part_raw, axis=0)

# 3. 对系统性部分进行标准化，使其方差恰好等于多元回归的 R^2
# 这样便能完美契合： 总体方差 = R^2 + (1 - R^2) = 1
systematic_part = np.zeros_like(systematic_part_raw)
for i in range(m):
    if sys_variance[i] > 0:
        # 归一化并乘以 sqrt(R_squared)
        systematic_part[:, i] = (systematic_part_raw[:, i] - np.mean(systematic_part_raw[:, i])) / np.sqrt(sys_variance[i]) * np.sqrt(df_portfolio['R_Squared'].values[i])
    else:
        systematic_part[:, i] = 0.0

# 4. 叠加上特异性部分，形成标准的 N(0, 1) 潜在变量 X
X = systematic_part + eta_vec * epsilon

# 步骤 D & E: 判定违约并计算组合总损失
default_matrix = (X <= thresh_vec).astype(int)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)
portfolio_losses_millions = portfolio_losses / 1_000_000

losses_p4 = portfolio_losses_millions

# ==========================================
# 5. 多因子风险指标量化与报告
# ==========================================
print("\n" + "=" * 40)
print("     基于多因子模型的资产组合风险研究结果")
print("=" * 40)

EL_m = np.mean(portfolio_losses_millions)
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

conf_level = 0.99
VaR_99_m = np.percentile(portfolio_losses_millions, conf_level * 100)
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

multi_p1_VaR_99_m = VaR_99_m

ES_99_m = portfolio_losses_millions[portfolio_losses_millions >= VaR_99_m].mean()
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

EC_m = VaR_99_m - EL_m
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("=" * 40)

# ==========================================
# 6. 新增：结合 Entities Info 的多因子尾部风险归因
# ==========================================
print("\n正在利用实体基础信息(Entities Info)进行多因子违约归因分析...")

# 计算极端情景（损失大于 99% VaR 的情景）下各区域/规模的平均违约贡献率
bad_scenarios_idx = np.where(portfolio_losses_millions >= VaR_99_m)[0]
tail_defaults = default_matrix[bad_scenarios_idx, :]  # 筛选尾部情景下的违约矩阵

# 各标的在尾部极端情况下的平均违约概率
df_portfolio['tail_default_rate'] = tail_defaults.mean(axis=0)

print("\n[极端尾部情景下不同地域(Region)的平均违约敏感度]:")
print(df_portfolio.groupby('region')['tail_default_rate'].mean().to_string())

print("\n[极端尾部情景下不同规模(Size Bucket)的平均违约敏感度]:")
print(df_portfolio.groupby('size_bucket')['tail_default_rate'].mean().to_string())

# ==========================================
# 7. 可视化多因子模拟损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='teal', alpha=0.4, label='Multi-Factor Loss Distribution')

sns.kdeplot(portfolio_losses_millions, color='darkcyan', linewidth=1, label='Multi-Factor KDE')

plt.axvline(EL_m, color='darkgreen', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title('Multi-Factor Portfolio Simulated Loss Distribution (with Entity Info)', fontsize=13, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, color='gainsboro', alpha=0.5)
plt.tight_layout()
plt.show()




# portfolio_2

print("Portfolio_2 的多因子模型分析报告")
print("正在读取数据文件...")

# 读取 4 个核心 CSV 文件
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables', 'cleaned_portfolio_2.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS,'cleaned_industry_index_returns.csv'))
df_entities_info = pd.read_csv(os.path.join(DIR_RESULTS,'cleaned_entities_info.csv'))

# ==========================================
# 2. 参数估计：多元线性回归提取多因子载荷 (Beta) 与特异性权重 (Eta)
# ==========================================
print("正在对各行业系统性因子进行对齐并建立多因子矩阵...")

# 2.1 转换数据结构为时间序列矩阵
stock_pivot = df_stock_returns.pivot_table(
    index='date',
    columns='entity_code',
    values='stock_return',
    aggfunc='mean'
)
# 行业指数收益率也建议改用 pivot_table 规避可能重复的数据，并前向/后向填充缺失值，确保时序连续
industry_pivot = df_industry_returns.pivot_table(
    index='date',
    columns='industry',
    values='index_return',
    aggfunc='mean'
).ffill().bfill()

# 确保日期完全对齐
common_dates = stock_pivot.index.intersection(industry_pivot.index)
stock_pivot = stock_pivot.loc[common_dates]
industry_pivot = industry_pivot.loc[common_dates]

# 所有的行业因子名称
factor_names = list(industry_pivot.columns)
num_factors = len(factor_names)

# 2.2 估计每一只股票在全市场多因子上的暴露度
print("正在利用多元线性回归估计多因子敏感度系数...")
multi_factor_params = []

for code in df_portfolio['entity_code'].unique():
    if code in stock_pivot.columns:
        # 获取当前股票的收益率序列，并剔除其自身的 NaN（如停牌日）
        y_raw = stock_pivot[code].dropna()

        # 严格对齐：确保 X 和 y 使用完全相同的非空日期交集
        common_idx = y_raw.index.intersection(industry_pivot.index)

        if len(common_idx) > 30:  # 保证样本充足
            y = y_raw.loc[common_idx]
            X = industry_pivot.loc[common_idx, factor_names]

            # 【双重保险】再次剔除任何可能潜在的 NaN 行（sklearn 的严格要求）
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(y_clean) > 30:
                # 拟合多元线性回归
                reg = LinearRegression().fit(X_clean, y_clean)
                r_squared = reg.score(X_clean, y_clean)
                betas = reg.coef_

                param_dict = {
                    'entity_code': code,
                    'R_Squared': r_squared,
                    'eta': np.sqrt(max(0, 1 - r_squared))
                }
                for idx, factor in enumerate(factor_names):
                    param_dict[f'beta_{factor}'] = betas[idx]

                multi_factor_params.append(param_dict)

df_params = pd.DataFrame(multi_factor_params)

# 2.3 将因子参数以及 entities_info 中的丰富特征(如地域、规模)合并到主资产组合表中
df_portfolio = pd.merge(df_portfolio, df_params, on='entity_code', how='inner')
df_portfolio = pd.merge(df_portfolio, df_entities_info[['entity_code', 'region', 'size_bucket']], on='entity_code',
                        how='left')

# 填补可能缺失的多元回归参数
df_portfolio['R_Squared'] = df_portfolio['R_Squared'].fillna(df_portfolio['R_Squared'].mean())
df_portfolio['eta'] = df_portfolio['eta'].fillna(np.sqrt(1 - df_portfolio['R_Squared']))
for factor in factor_names:
    df_portfolio[f'beta_{factor}'] = df_portfolio[f'beta_{factor}'].fillna(0.0)

# ==========================================
# 3. 计算违约阈值 (Threshold)
# ==========================================
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD'])

m = len(df_portfolio)
print(f"多因子参数估计完成。用于多因子建模的债务人数量: {m}")

# ==========================================
# 4. 多因子蒙特卡洛模拟损失分布
# ==========================================
print("\n正在进行多因子框架下的蒙特卡洛模拟...")
N = 50000
np.random.seed(42)

ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
eta_vec = df_portfolio['eta'].values
thresh_vec = df_portfolio['threshold'].values

# 提取各资产对所有因子的 Beta 矩阵 (形状: m x num_factors)
beta_matrix = df_portfolio[[f'beta_{factor}' for factor in factor_names]].values

# 步骤 A: 模拟多个行业系统性风险因子矩阵 F
# 考虑到现实中各行业指数存在相关性，我们从历史数据中提取因子之间的协方差矩阵
factor_covariance = industry_pivot[factor_names].cov().values
factor_means = industry_pivot[factor_names].mean().values

# 从多元正态分布中抽取 N 个情景的系统性风险因子 (形状: N x num_factors)
F_multi = np.random.multivariate_normal(factor_means, factor_covariance, size=N)

# 步骤 B: 模拟特异性风险成分 epsilon (形状: N x m)
epsilon = np.random.normal(0, 1, (N, m))

# 步骤 C: 根据多因子公式计算每个情景下每个主体的潜在信用变量 X
# 1. 计算原始系统性部分 (N x m)
systematic_part_raw = np.dot(F_multi, beta_matrix.T)

# 2. 计算每个资产在历史/模拟状态下的系统性成分方差 (取每只股票在 N 个情景下的方差)
# 这是为了确保系统性方差 + 特异性方差 = 1
sys_variance = np.var(systematic_part_raw, axis=0)

# 3. 对系统性部分进行标准化，使其方差恰好等于多元回归的 R^2
# 这样便能完美契合： 总体方差 = R^2 + (1 - R^2) = 1
systematic_part = np.zeros_like(systematic_part_raw)
for i in range(m):
    if sys_variance[i] > 0:
        # 归一化并乘以 sqrt(R_squared)
        systematic_part[:, i] = (systematic_part_raw[:, i] - np.mean(systematic_part_raw[:, i])) / np.sqrt(sys_variance[i]) * np.sqrt(df_portfolio['R_Squared'].values[i])
    else:
        systematic_part[:, i] = 0.0

# 4. 叠加上特异性部分，形成标准的 N(0, 1) 潜在变量 X
X = systematic_part + eta_vec * epsilon

# 步骤 D & E: 判定违约并计算组合总损失
default_matrix = (X <= thresh_vec).astype(int)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)
portfolio_losses_millions = portfolio_losses / 1_000_000

losses_p5 = portfolio_losses_millions



# ==========================================
# 5. 多因子风险指标量化与报告
# ==========================================
print("\n" + "=" * 40)
print("     基于多因子模型的资产组合风险研究结果")
print("=" * 40)

EL_m = np.mean(portfolio_losses_millions)
print(f"预期损失 (Expected Loss, EL)      : ${EL_m:,.2f}M")

conf_level = 0.99
VaR_99_m = np.percentile(portfolio_losses_millions, conf_level * 100)
print(f"99% 在险价值 (Value at Risk, VaR) : ${VaR_99_m:,.2f}M")

multi_p2_VaR_99_m = VaR_99_m

ES_99_m = portfolio_losses_millions[portfolio_losses_millions >= VaR_99_m].mean()
print(f"99% 预期不足 (Expected Shortfall, ES): ${ES_99_m:,.2f}M")

EC_m = VaR_99_m - EL_m
print(f"应配置的经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("=" * 40)

# ==========================================
# 6. 新增：结合 Entities Info 的多因子尾部风险归因
# ==========================================
print("\n正在利用实体基础信息(Entities Info)进行多因子违约归因分析...")

# 计算极端情景（损失大于 99% VaR 的情景）下各区域/规模的平均违约贡献率
bad_scenarios_idx = np.where(portfolio_losses_millions >= VaR_99_m)[0]
tail_defaults = default_matrix[bad_scenarios_idx, :]  # 筛选尾部情景下的违约矩阵

# 各标的在尾部极端情况下的平均违约概率
df_portfolio['tail_default_rate'] = tail_defaults.mean(axis=0)

print("\n[极端尾部情景下不同地域(Region)的平均违约敏感度]:")
print(df_portfolio.groupby('region')['tail_default_rate'].mean().to_string())

print("\n[极端尾部情景下不同规模(Size Bucket)的平均违约敏感度]:")
print(df_portfolio.groupby('size_bucket')['tail_default_rate'].mean().to_string())

# ==========================================
# 7. 可视化多因子模拟损失分布
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='teal', alpha=0.4, label='Multi-Factor Loss Distribution')

sns.kdeplot(portfolio_losses_millions, color='darkcyan', linewidth=1, label='Multi-Factor KDE')

plt.axvline(EL_m, color='darkgreen', linestyle='--', linewidth=1, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title('Multi-Factor Portfolio Simulated Loss Distribution (with Entity Info)', fontsize=13, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, color='gainsboro', alpha=0.5)
plt.tight_layout()
plt.show()



## portfolio_3

# 1. 全局配置 (Essential Requirement)
DIR_RESULTS = "./results"  # 请根据您本地的实际目录进行微调

# ==========================================
# 1. 加载数据文件 (已更新为 portfolio_3 并引入 entities_info)
# ==========================================
print("Portfolio_3 的多因子精细化风险模型研究报告")
print("正在读取数据文件...")

# 将 portfolio_1 替换为 portfolio_3
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables','cleaned_portfolio_3.csv_merged.csv'))
df_stock_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_stock_returns.csv'))
df_industry_returns = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_industry_index_returns.csv'))
df_entities_info = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_entities_info.csv'))

# ==========================================
# 2. 参数估计：多元线性回归提取多因子载荷 (Beta) 与特异性权重 (Eta)
# ==========================================
print("正在对各行业系统性因子进行对齐并建立多因子矩阵...")

# 2.1 转换数据结构为时间序列矩阵
stock_pivot = df_stock_returns.pivot_table(
    index='date',
    columns='entity_code',
    values='stock_return',
    aggfunc='mean'
)

# 使用 ffill/bfill 确保宏观行业时序因子的连续性与稳健性
industry_pivot = df_industry_returns.pivot_table(
    index='date',
    columns='industry',
    values='index_return',
    aggfunc='mean'
).ffill().bfill()

# 确保日期完全对齐
common_dates = stock_pivot.index.intersection(industry_pivot.index)
stock_pivot = stock_pivot.loc[common_dates]
industry_pivot = industry_pivot.loc[common_dates]

# 提取市场中所有的行业系统性风险因子
factor_names = list(industry_pivot.columns)
num_factors = len(factor_names)

# 2.2 估计每只个股在全市场多因子上的暴露度
print("正在利用多元线性回归估计多因子敏感度系数...")
multi_factor_params = []

for code in df_portfolio['entity_code'].unique():
    if code in stock_pivot.columns:
        y_raw = stock_pivot[code].dropna()
        common_idx = y_raw.index.intersection(industry_pivot.index)

        if len(common_idx) > 30:  # 保证充足的时序样本
            y = y_raw.loc[common_idx]
            X = industry_pivot.loc[common_idx, factor_names]

            # 剔除潜在的 NaN 值，确保 sklearn 矩阵合规
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            if len(y_clean) > 30:
                reg = LinearRegression().fit(X_clean, y_clean)
                r_squared = reg.score(X_clean, y_clean)
                betas = reg.coef_

                # 完美契合讲义公式: 特异性风险权重 eta = sqrt(1 - R^2)
                param_dict = {
                    'entity_code': code,
                    'R_Squared': r_squared,
                    'eta': np.sqrt(max(0, 1 - r_squared))
                }
                for idx, factor in enumerate(factor_names):
                    param_dict[f'beta_{factor}'] = betas[idx]

                multi_factor_params.append(param_dict)

df_params = pd.DataFrame(multi_factor_params)

# 2.3 将多因子回归参数、entities_info(地域、规模等元数据)融合至 Portfolio_3 主表
df_portfolio = pd.merge(df_portfolio, df_params, on='entity_code', how='inner')
df_portfolio = pd.merge(df_portfolio, df_entities_info[['entity_code', 'region', 'size_bucket']], on='entity_code', how='left')

# 对偶发性时序中断导致的缺失值进行行业/总体均值稳健填充
df_portfolio['R_Squared'] = df_portfolio['R_Squared'].fillna(df_portfolio['R_Squared'].mean())
df_portfolio['eta'] = df_portfolio['eta'].fillna(np.sqrt(1 - df_portfolio['R_Squared']))
for factor in factor_names:
    df_portfolio[f'beta_{factor}'] = df_portfolio[f'beta_{factor}'].fillna(0.0)

# ==========================================
# 3. 计算违约阈值 (Threshold)
# ==========================================
# 剔除资产基础敞口数据缺失的无效主体
df_portfolio = df_portfolio.dropna(subset=['EAD', 'LGD', 'PD'])

# 根据讲义 CreditMetrics 映射公式: d_i = \Phi^{-1}(PD_i)
df_portfolio['threshold'] = stats.norm.ppf(df_portfolio['PD'])

m = len(df_portfolio)
print(f"多因子参数估计完成。Portfolio_3 用于多因子建模的债务人数量: {m}")

# ==========================================
# 4. 多因子蒙特卡洛模拟损失分布 (结合标准化方差校验)
# ==========================================
print("\n正在进行精细化多因子框架下的蒙特卡洛模拟...")
N = 50000
np.random.seed(42)

ead_vec = df_portfolio['EAD'].values
lgd_vec = df_portfolio['LGD'].values
eta_vec = df_portfolio['eta'].values
thresh_vec = df_portfolio['threshold'].values

# 提取 Beta 敏感度载荷矩阵 (形状: m x num_factors)
beta_matrix = df_portfolio[[f'beta_{factor}' for factor in factor_names]].values

# 步骤 A: 捕捉历史行业因子之间的关联性，生成多元正态联合宏观情景 F
factor_covariance = industry_pivot[factor_names].cov().values
factor_means = industry_pivot[factor_names].mean().values
F_multi = np.random.multivariate_normal(factor_means, factor_covariance, size=N)

# 步骤 B: 模拟特异性风险矩阵 epsilon (形状: N x m)
epsilon = np.random.normal(0, 1, (N, m))

# 步骤 C: 计算潜在资产信用变量 X = Systematic + Idiosyncratic
# 1. 矩阵乘法计算原始系统性部分载荷 (N x m)
systematic_part_raw = np.dot(F_multi, beta_matrix.T)

# 2. 捕获每只资产在 50,000 次模拟路径下的系统性成分原始方差
sys_variance = np.var(systematic_part_raw, axis=0)

# 3. 执行严格的方差重校准标准化，迫使系统性方差恰好等于回归模型的 R^2
systematic_part = np.zeros_like(systematic_part_raw)
for i in range(m):
    if sys_variance[i] > 0:
        # 归一化并配比多元 R_squared 权重
        systematic_part[:, i] = ((systematic_part_raw[:, i] - np.mean(systematic_part_raw[:, i])) /
                                 np.sqrt(sys_variance[i]) * np.sqrt(df_portfolio['R_Squared'].values[i]))
    else:
        systematic_part[:, i] = 0.0

# 4. 融合特异性部分，确保总方差严格恒等于 1
X = systematic_part + eta_vec * epsilon

# 步骤 D & E: 判定联合违约群聚，加总导出资产组合损失
default_matrix = (X <= thresh_vec).astype(int)
loss_matrix = default_matrix * (ead_vec * lgd_vec)
portfolio_losses = loss_matrix.sum(axis=1)
portfolio_losses_millions = portfolio_losses / 1_000_000

losses_p6 = portfolio_losses_millions



# ==========================================
# 5. 多因子风险指标量化与度量
# ==========================================
print("\n" + "=" * 50)
print("     基于多因子精细化模型的 Portfolio_3 风险量化研究结果")
print("=" * 50)

EL_m = np.mean(portfolio_losses_millions)
print(f"预期损失 (Expected Loss, EL)         : ${EL_m:,.2f}M")

conf_level = 0.99
VaR_99_m = np.percentile(portfolio_losses_millions, conf_level * 100)
print(f"99% 在险价值 (Value at Risk, VaR)    : ${VaR_99_m:,.2f}M")

multi_p3_VaR_99_m = VaR_99_m

# 评估极端超阈值状态下的平均尾部损失 (Expected Shortfall)
ES_99_m = portfolio_losses_millions[portfolio_losses_millions >= VaR_99_m].mean()
print(f"99% 预期不足 (Expected Shortfall, ES)  : ${ES_99_m:,.2f}M")

EC_m = VaR_99_m - EL_m
print(f"应配置的非预期风险经济资本 (Economic Capital) : ${EC_m:,.2f}M")
print("=" * 50)

# ==========================================
# 6. 结合 Entities Info 进行极端尾部多因子风险归因
# ==========================================
print("\n正在利用实体基础信息(Entities Info)进行多因子极端尾部违约归因分析...")

# 锁定击穿 99% VaR 的极端宏观压力情景切片
bad_scenarios_idx = np.where(portfolio_losses_millions >= VaR_99_m)[0]
tail_defaults = default_matrix[bad_scenarios_idx, :]

# 提取各个债务人个体在尾部情景下的平均条件违约率
df_portfolio['tail_default_rate'] = tail_defaults.mean(axis=0)

print("\n[极端尾部压力情景下不同地域(Region)的平均条件违约压力]:")
region_summary = df_portfolio.groupby('region')['tail_default_rate'].mean()
region_summary_pct = region_summary.map(lambda x: f"{x*100:.2f}%")
print(region_summary_pct.to_string())

print("\n[极端尾部压力情景下不同规模(Size Bucket)的平均条件违约压力]:")
size_summary = df_portfolio.groupby('size_bucket')['tail_default_rate'].mean()
size_summary_pct = size_summary.map(lambda x: f"{x*100:.2f}%")
print(size_summary_pct.to_string())

# ==========================================
# 7. 可视化多因子模拟损失分布
# ==========================================
plt.figure(figsize=(11, 6))
sns.histplot(portfolio_losses_millions, bins=150, stat='density',
             color='teal', alpha=0.4, label='Portfolio_3 Multi-Factor Loss')

sns.kdeplot(portfolio_losses_millions, color='darkcyan', linewidth=1.5, label='Multi-Factor Smooth KDE')

plt.axvline(EL_m, color='darkgreen', linestyle='--', linewidth=1.2, label=f'EL: ${EL_m:,.2f}M')
plt.axvline(VaR_99_m, color='crimson', linestyle='-', linewidth=1.2, label=f'99% VaR: ${VaR_99_m:,.2f}M')
plt.axvline(ES_99_m, color='darkorange', linestyle=':', linewidth=1.2, label=f'99% ES: ${ES_99_m:,.2f}M')

plt.title('Portfolio_3 Multi-Factor Simulated Loss Distribution & Risk Attribution', fontsize=13, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, color='gainsboro', alpha=0.5)
plt.tight_layout()
plt.show()


# ==============================================
# 将Portfolio 1 (One-Factor) 与 Portfolio 1 (Multi—Factor) 合并研究
# ==============================================

# 1. 组装数据字典
portfolio_data = {
    'Portfolio 1 (One-Factor)': {'data': losses_p1, 'color': 'royalblue', 'line': '-'},
    'Portfolio 1 (Multi—Factor)': {'data': losses_p4, 'color': 'emerald', 'line': '--'},

}

plt.figure(figsize=(12, 7))

# 2. 循环绘制每个组合的 KDE 曲线和核心指标
colors = ['#4F7CAC', '#2C9E4B']  # 优雅的学术风配色：钢蓝、内敛绿、朱红
styles = ['-', '--']

for i, (name, p_info) in enumerate(portfolio_data.items()):
    losses = p_info['data']
    color = colors[i]
    style = styles[i]

    # 计算核心指标
    el = np.mean(losses)
    var_99 = np.percentile(losses, 99)

    # 绘制平滑密度曲线 (去掉不透明的直方图，避免画面太满，改用 shade/fill)
    sns.kdeplot(losses, color=color, linestyle=style, linewidth=1, label=name, fill=True, alpha=0.05)

    # 标出各自的 EL 和 99% VaR
    plt.axvline(el, color=color, linestyle=':', linewidth=0.6, alpha=0.8)
    plt.axvline(var_99, color=color, linestyle=style, linewidth=0.6, alpha=0.9)

    # 在图形上方动态添加文本标注，避免线堆在一起分不清
    plt.text(var_99, plt.gca().get_ylim()[1] * (0.85 - i * 0.08), f"{name} 99% VaR: ${var_99:.2f}M",
             color=color, fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 3. 美化图表
plt.title('Comparison of Simulated Loss Distributions (Portfolio 1)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)

plt.grid(True, color='gainsboro', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.show()


# ==============================================
# 将Portfolio 2 (One-Factor) 与 Portfolio 2 (Multi—Factor) 合并研究
# ==============================================

# 1. 组装数据字典
portfolio_data = {
    'Portfolio 2 (One-Factor)': {'data': losses_p1, 'color': 'royalblue', 'line': '-'},
    'Portfolio 2 (Multi—Factor)': {'data': losses_p5, 'color': 'emerald', 'line': '--'},

}

plt.figure(figsize=(12, 7))

# 2. 循环绘制每个组合的 KDE 曲线和核心指标
colors = ['#4F7CAC', '#2C9E4B']  # 优雅的学术风配色：钢蓝、内敛绿、朱红
styles = ['-', '--']

for i, (name, p_info) in enumerate(portfolio_data.items()):
    losses = p_info['data']
    color = colors[i]
    style = styles[i]

    # 计算核心指标
    el = np.mean(losses)
    var_99 = np.percentile(losses, 99)

    # 绘制平滑密度曲线 (去掉不透明的直方图，避免画面太满，改用 shade/fill)
    sns.kdeplot(losses, color=color, linestyle=style, linewidth=1, label=name, fill=True, alpha=0.05)

    # 标出各自的 EL 和 99% VaR
    plt.axvline(el, color=color, linestyle=':', linewidth=0.6, alpha=0.8)
    plt.axvline(var_99, color=color, linestyle=style, linewidth=0.6, alpha=0.9)

    # 在图形上方动态添加文本标注，避免线堆在一起分不清
    plt.text(var_99, plt.gca().get_ylim()[1] * (0.85 - i * 0.08), f"{name} 99% VaR: ${var_99:.2f}M",
             color=color, fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 3. 美化图表
plt.title('Comparison of Simulated Loss Distributions (Portfolio 2)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)

plt.grid(True, color='gainsboro', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.show()


# ==============================================
# 将Portfolio 3 (One-Factor) 与 Portfolio 3 (Multi—Factor) 合并研究
# ==============================================

# 1. 组装数据字典
portfolio_data = {
    'Portfolio 3 (One-Factor)': {'data': losses_p1, 'color': 'royalblue', 'line': '-'},
    'Portfolio 3 (Multi—Factor)': {'data': losses_p6, 'color': 'emerald', 'line': '--'},

}

plt.figure(figsize=(12, 7))

# 2. 循环绘制每个组合的 KDE 曲线和核心指标
colors = ['#4F7CAC', '#2C9E4B']  # 优雅的学术风配色：钢蓝、内敛绿、朱红
styles = ['-', '--']

for i, (name, p_info) in enumerate(portfolio_data.items()):
    losses = p_info['data']
    color = colors[i]
    style = styles[i]

    # 计算核心指标
    el = np.mean(losses)
    var_99 = np.percentile(losses, 99)

    # 绘制平滑密度曲线 (去掉不透明的直方图，避免画面太满，改用 shade/fill)
    sns.kdeplot(losses, color=color, linestyle=style, linewidth=1, label=name, fill=True, alpha=0.05)

    # 标出各自的 EL 和 99% VaR
    plt.axvline(el, color=color, linestyle=':', linewidth=0.6, alpha=0.8)
    plt.axvline(var_99, color=color, linestyle=style, linewidth=0.6, alpha=0.9)

    # 在图形上方动态添加文本标注，避免线堆在一起分不清
    plt.text(var_99, plt.gca().get_ylim()[1] * (0.85 - i * 0.08), f"{name} 99% VaR: ${var_99:.2f}M",
             color=color, fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# 3. 美化图表
plt.title('Comparison of Simulated Loss Distributions (Portfolio 3)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Total Loss (Millions of Currency Units)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)

plt.grid(True, color='gainsboro', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.show()


## 生成一个portfolio在个模型下的VaR表格

# 1. 构建结构化字典数据
summary_data = {
    "Portfolio": ["Portfolio 1", "Portfolio 2", "Portfolio 3"],
    "One-Factor Monte Carlo (VaR 99%)": [f"${single_p1_VaR_99_m:.2f}M", f"${single_p2_VaR_99_m:.2f}M", f"${single_p3_VaR_99_m:.2f}M"],
    "Vasicek Analytical (VaR 99%)": [f"${analytical_p1_var:.2f}M", f"${analytical_p2_var:.2f}M", "N/A (Single-Factor Only)"],
    "Multi-Factor Monte Carlo (VaR 99%)": [f"${multi_p1_VaR_99_m:.2f}M", f"${multi_p2_VaR_99_m:.2f}M", f"${multi_p3_VaR_99_m:.2f}M"]
}

# 2. 转化为 Pandas DataFrame 并将 Portfolio 设为纵向索引
df_var_comparison = pd.DataFrame(summary_data).set_index("Portfolio")

# 3. 打印并保存表格
print("\n" + "*"*20 + " 最终 VaR 风险指标对比表 " + "*"*20)
print(df_var_comparison)
print("*"*65)

# 4. 自动将表格导出为 CSV，保存在你的 results 目录下
output_summary_path = os.path.join(DIR_RESULTS, 'report', "portfolio_var_comparison_matrix.csv")
df_var_comparison.to_csv(output_summary_path, encoding='utf-8-sig')
print(f"[成功] 对比表格已优雅导出至: {output_summary_path}")



##  Beta_Binomial_Mixture_Model

## portfolio_1
# -------------------------------------------------------------
# 步骤 1: 读取数据并从历史违约数据中估计 Beta-Binomial 参数
# -------------------------------------------------------------
# 读取20年历史违约数据
df_history = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_default_history_20y.csv'))

# 解析日期以提取年份 (处理空值并标准化)
df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
# 如果日期列存在缺失，通常历史违约表是按年汇总或有特定年份标示，此处按年聚合
df_history['Year'] = df_history['date'].dt.year

# 处理违约事件列中的缺失值（视为空或0）并计算每年的实际违约率
df_history['default_event'] = pd.to_numeric(df_history['default_event'], errors='coerce').fillna(0)

# 计算每年总样本数和违约数，得到每年的违约率序列
annual_stats = df_history.groupby('Year')['default_event'].agg(['count', 'sum'])
annual_stats['default_rate'] = annual_stats['sum'] / annual_stats['count']

# 计算历史违约率的均值和方差 (矩估计)
mu_p = annual_stats['default_rate'].mean()
sigma2_p = annual_stats['default_rate'].var()

print(f"--- 历史数据统计 ---")
print(f"长期平均违约率 (Mean PD): {mu_p:.4f}")
print(f"违约率历史方差 (Variance of PD): {sigma2_p:.6f}")

# 矩估计法求解 Beta 分布参数 alpha 和 beta
M = (mu_p * (1 - mu_p) / sigma2_p) - 1
alpha = mu_p * M
beta = (1 - mu_p) * M
rho = 1 / (alpha + beta + 1)

print(f"\n--- Beta-Binomial 模型估计参数 ---")
print(f"Alpha (α): {alpha:.4f}")
print(f"Beta (β): {beta:.4f}")
print(f"隐含的资产违约相关性 (Default Correlation ρ): {rho * 100:.2f}%")

# -------------------------------------------------------------
# 步骤 2: 读取当前 Portfolio 数据并计算风险指标
# -------------------------------------------------------------
# 读取资产组合数据
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables','cleaned_portfolio_1.csv_merged.csv'))

# 确保 EAD 和 LGD 为数值型
df_portfolio['EAD'] = pd.to_numeric(df_portfolio['EAD'], errors='coerce').fillna(0)
df_portfolio['LGD'] = pd.to_numeric(df_portfolio['LGD'], errors='coerce').fillna(0)

# 过滤掉敞口为0的无效数据
df_portfolio = df_portfolio[df_portfolio['EAD'] > 0].reset_index(drop=True)
N_assets = len(df_portfolio)
total_ead = df_portfolio['EAD'].sum()

print(f"\n--- 资产组合概况 ---")
print(f"资产总数 (N): {N_assets}")
print(f"总风险敞口 (Total EAD): {total_ead:,.2f}")

# -------------------------------------------------------------
# 步骤 3: 蒙特卡洛模拟组合损失分布
# -------------------------------------------------------------
np.random.seed(42)  # 设置随机种子以保证结果可重复
N_simulations = 100000  # 模拟10万次经济情景

# 1. 模拟宏观经济驱动下的系统性违约率 p (服从 Beta 分布)
simulated_p = stats.beta.rvs(alpha, beta, size=N_simulations)

# 2. 对每个场景进行资产级别的违约模拟
# 创建损失矩阵：行对应模拟场景，列对应资产单个损失
portfolio_losses = np.zeros(N_simulations)

# 为了提升计算效率，采用向量化模拟：
# 每个资产的违约率在场景中独立抽样（条件独立设定）
for i in range(N_assets):
    ead_i = df_portfolio.loc[i, 'EAD']
    lgd_i = df_portfolio.loc[i, 'LGD']

    # 在每种场景下，判断资产 i 是否违约 (1表示违约，0表示未违约)
    # simulated_p 是大小为 (N_simulations,) 的数组
    defaults = np.random.rand(N_simulations) < simulated_p

    # 累加每个资产在该场景下的损失 = 违约(0或1) * EAD * LGD
    portfolio_losses += defaults * ead_i * lgd_i

portfolio_losses_millions = portfolio_losses / 1_000_000  # 在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

# -------------------------------------------------------------
# 步骤 4: 计算组合风险度量指标 (EL, VaR, ES)
# -------------------------------------------------------------
# 预期损失 (Expected Loss)
el = np.mean(portfolio_losses)
el_m = el / 1_000_000
el_pct = el / total_ead

# 风险价值 (Value at Risk - 95% 和 99% 置信度)
var_95 = np.percentile(portfolio_losses, 95)
var_95_m = var_95 / 1_000_000
var_99 = np.percentile(portfolio_losses, 99)
var_99_m = var_99 / 1_000_000

# 预期尾部损失 (Expected Shortfall / Conditional VaR)
es_95 = np.mean(portfolio_losses[portfolio_losses >= var_95])
es_95_m = es_95 / 1_000_000
es_99 = np.mean(portfolio_losses[portfolio_losses >= var_99])
es_99_m = es_99 / 1_000_000

# 非预期损失 (Unexpected Loss = 资产组合损失的标准差)
ul = np.std(portfolio_losses)
ul_m = ul / 1_000_000

print(f"\n--- 资产组合风险计量结果 ---")
print(f"预期损失 (Expected Loss, EL): {el_m:,.2f}M ({el_pct * 100:.2f}%)")
print(f"非预期损失 (Unexpected Loss, UL): {ul_m:,.2f}M")
print(f"95% 风险价值 (95% VaR): {var_95_m:,.2f}M")
print(f"99% 风险价值 (99% VaR): {var_99_m:,.2f}M")
print(f"95% 预期尾部损失 (95% ES): {es_95_m:,.2f}M")
print(f"99% 预期尾部损失 (99% ES): {es_99_m:,.2f}M")

# -------------------------------------------------------------
# 步骤 5: 绘制组合损失分布图
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
# 步骤一：使用 seaborn 绘制半透明的钢蓝色直方图
sns.histplot(portfolio_losses_millions, bins=100, stat='density',
             color='steelblue', alpha=0.4, label='Simulated Losses')

# 步骤二：叠加一条霸道的、绝对独立的royalblue色 KDE 平滑曲线（规避底层冲突）
sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='KDE Curve')


#plt.hist(portfolio_losses, bins=100, density=True, alpha=0.6, color='navy', label='Simulated Losses')
plt.axvline(el_m, color='navy', linestyle='--', linewidth=1, label=f'EL: {el:,.0f}')
plt.axvline(var_95_m, color='orange', linestyle='-', linewidth=1, label=f'95% VaR: {var_95:,.0f}')
plt.axvline(var_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: {var_99:,.0f}')

plt.title('Credit Portfolio Loss Distribution (Portfolio 1)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Loss (Amount)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


## portfolio_2
# -------------------------------------------------------------
# 步骤 1: 读取数据并从历史违约数据中估计 Beta-Binomial 参数
# -------------------------------------------------------------
# 读取20年历史违约数据
df_history = pd.read_csv(os.path.join(DIR_RESULTS, 'cleaned_default_history_20y.csv'))

# 解析日期以提取年份 (处理空值并标准化)
df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
# 如果日期列存在缺失，通常历史违约表是按年汇总或有特定年份标示，此处按年聚合
df_history['Year'] = df_history['date'].dt.year

# 处理违约事件列中的缺失值（视为空或0）并计算每年的实际违约率
df_history['default_event'] = pd.to_numeric(df_history['default_event'], errors='coerce').fillna(0)

# 计算每年总样本数和违约数，得到每年的违约率序列
annual_stats = df_history.groupby('Year')['default_event'].agg(['count', 'sum'])
annual_stats['default_rate'] = annual_stats['sum'] / annual_stats['count']

# 计算历史违约率的均值和方差 (矩估计)
mu_p = annual_stats['default_rate'].mean()
sigma2_p = annual_stats['default_rate'].var()

print(f"--- 历史数据统计 ---")
print(f"长期平均违约率 (Mean PD): {mu_p:.4f}")
print(f"违约率历史方差 (Variance of PD): {sigma2_p:.6f}")

# 矩估计法求解 Beta 分布参数 alpha 和 beta
M = (mu_p * (1 - mu_p) / sigma2_p) - 1
alpha = mu_p * M
beta = (1 - mu_p) * M
rho = 1 / (alpha + beta + 1)

print(f"\n--- Beta-Binomial 模型估计参数 ---")
print(f"Alpha (α): {alpha:.4f}")
print(f"Beta (β): {beta:.4f}")
print(f"隐含的资产违约相关性 (Default Correlation ρ): {rho * 100:.2f}%")

# -------------------------------------------------------------
# 步骤 2: 读取当前 Portfolio 数据并计算风险指标
# -------------------------------------------------------------
# 读取资产组合数据
df_portfolio = pd.read_csv(os.path.join(DIR_RESULTS, 'merged_tables','cleaned_portfolio_2.csv_merged.csv'))

# 确保 EAD 和 LGD 为数值型
df_portfolio['EAD'] = pd.to_numeric(df_portfolio['EAD'], errors='coerce').fillna(0)
df_portfolio['LGD'] = pd.to_numeric(df_portfolio['LGD'], errors='coerce').fillna(0)

# 过滤掉敞口为0的无效数据
df_portfolio = df_portfolio[df_portfolio['EAD'] > 0].reset_index(drop=True)
N_assets = len(df_portfolio)
total_ead = df_portfolio['EAD'].sum()

print(f"\n--- 资产组合概况 ---")
print(f"资产总数 (N): {N_assets}")
print(f"总风险敞口 (Total EAD): {total_ead:,.2f}")

# -------------------------------------------------------------
# 步骤 3: 蒙特卡洛模拟组合损失分布
# -------------------------------------------------------------
np.random.seed(42)  # 设置随机种子以保证结果可重复
N_simulations = 100000  # 模拟10万次经济情景

# 1. 模拟宏观经济驱动下的系统性违约率 p (服从 Beta 分布)
simulated_p = stats.beta.rvs(alpha, beta, size=N_simulations)

# 2. 对每个场景进行资产级别的违约模拟
# 创建损失矩阵：行对应模拟场景，列对应资产单个损失
portfolio_losses = np.zeros(N_simulations)

# 为了提升计算效率，采用向量化模拟：
# 每个资产的违约率在场景中独立抽样（条件独立设定）
for i in range(N_assets):
    ead_i = df_portfolio.loc[i, 'EAD']
    lgd_i = df_portfolio.loc[i, 'LGD']

    # 在每种场景下，判断资产 i 是否违约 (1表示违约，0表示未违约)
    # simulated_p 是大小为 (N_simulations,) 的数组
    defaults = np.random.rand(N_simulations) < simulated_p

    # 累加每个资产在该场景下的损失 = 违约(0或1) * EAD * LGD
    portfolio_losses += defaults * ead_i * lgd_i

portfolio_losses_millions = portfolio_losses / 1_000_000  # 在计算完总损失后，整体除以 1,000,000，转化为“百万元”单位

# -------------------------------------------------------------
# 步骤 4: 计算组合风险度量指标 (EL, VaR, ES)
# -------------------------------------------------------------
# 预期损失 (Expected Loss)
el = np.mean(portfolio_losses)
el_m = el / 1_000_000
el_pct = el / total_ead

# 风险价值 (Value at Risk - 95% 和 99% 置信度)
var_95 = np.percentile(portfolio_losses, 95)
var_95_m = var_95 / 1_000_000
var_99 = np.percentile(portfolio_losses, 99)
var_99_m = var_99 / 1_000_000

# 预期尾部损失 (Expected Shortfall / Conditional VaR)
es_95 = np.mean(portfolio_losses[portfolio_losses >= var_95])
es_95_m = es_95 / 1_000_000
es_99 = np.mean(portfolio_losses[portfolio_losses >= var_99])
es_99_m = es_99 / 1_000_000

# 非预期损失 (Unexpected Loss = 资产组合损失的标准差)
ul = np.std(portfolio_losses)
ul_m = ul / 1_000_000

print(f"\n--- 资产组合风险计量结果 ---")
print(f"预期损失 (Expected Loss, EL): {el_m:,.2f}M ({el_pct * 100:.2f}%)")
print(f"非预期损失 (Unexpected Loss, UL): {ul_m:,.2f}M")
print(f"95% 风险价值 (95% VaR): {var_95_m:,.2f}M")
print(f"99% 风险价值 (99% VaR): {var_99_m:,.2f}M")
print(f"95% 预期尾部损失 (95% ES): {es_95_m:,.2f}M")
print(f"99% 预期尾部损失 (99% ES): {es_99_m:,.2f}M")

# -------------------------------------------------------------
# 步骤 5: 绘制组合损失分布图
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
# 步骤一：使用 seaborn 绘制半透明的钢蓝色直方图
sns.histplot(portfolio_losses_millions, bins=100, stat='density',
             color='steelblue', alpha=0.4, label='Simulated Losses')

# 步骤二：叠加一条霸道的、绝对独立的royalblue色 KDE 平滑曲线（规避底层冲突）
sns.kdeplot(portfolio_losses_millions, color='royalblue', linewidth=1, label='KDE Curve')


#plt.hist(portfolio_losses, bins=100, density=True, alpha=0.6, color='navy', label='Simulated Losses')
plt.axvline(el_m, color='navy', linestyle='--', linewidth=1, label=f'EL: {el:,.0f}')
plt.axvline(var_95_m, color='orange', linestyle='-', linewidth=1, label=f'95% VaR: {var_95:,.0f}')
plt.axvline(var_99_m, color='crimson', linestyle='-', linewidth=1, label=f'99% VaR: {var_99:,.0f}')

plt.title('Credit Portfolio Loss Distribution (Portfolio 2)', fontsize=14, fontweight='bold')
plt.xlabel('Portfolio Loss (Amount)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

