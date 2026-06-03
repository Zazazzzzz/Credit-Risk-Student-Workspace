import pandas as pd
import numpy as np
from scipy.stats import norm, linregress

DATA_DIR = "./Data/"

def run_modeling_pipeline():

    master_df = pd.read_csv(DATA_DIR + 'cleaned_master_table.csv')
    stock_df = pd.read_csv(DATA_DIR + 'stock_returns.csv')
    ind_df = pd.read_csv(DATA_DIR + 'industry_index_returns.csv')

    stock_df['date'] = pd.to_datetime(stock_df['date'], format='mixed')
    ind_df['date'] = pd.to_datetime(ind_df['date'], format='mixed')

    master_df['entity_code'] = master_df['entity_code'].astype(str).str.strip().str.upper()
    stock_df['entity_code'] = stock_df['entity_code'].astype(str).str.strip().str.upper()

    def map_industry_by_code(code):
        code = str(code).upper()
        if code.startswith('A'): return 'Industry A'
        if code.startswith('B'): return 'Industry B'
        return 'Unknown'

    master_df['industry'] = master_df['entity_code'].apply(map_industry_by_code)
    stock_df['industry'] = stock_df['entity_code'].apply(map_industry_by_code)

    def standardize_ind_name(name):
        name = str(name).lower().replace(' ', '').replace('.', '').replace('-', '')
        if 'a' in name: return 'Industry A'
        if 'b' in name: return 'Industry B'
        return name

    ind_df['industry'] = ind_df['industry'].apply(standardize_ind_name)

    master_df['default_threshold'] = norm.ppf(master_df['avg_pd'].clip(0.001, 0.999))

    stock_df['date'] = pd.to_datetime(stock_df['date'], format='mixed')
    ind_df['date'] = pd.to_datetime(ind_df['date'], format='mixed')

    merged = pd.merge(stock_df, ind_df, on=['date', 'industry'], how='inner')

    def calc_beta(group):
        clean_group = group.dropna(subset=['index_return', 'stock_return'])

        if len(clean_group) < 10:
            return np.nan

        slope, _, _, _, _ = linregress(clean_group['index_return'], clean_group['stock_return'])
        return slope

    betas = merged.groupby('entity_code').apply(calc_beta).reset_index(name='beta')
    master_df = master_df.merge(betas, on='entity_code', how='left')
    master_df['beta'] = master_df.groupby('industry')['beta'].transform(lambda x: x.fillna(x.mean()))

    out_file = DATA_DIR + 'final_modeling_data.csv'
    master_df.to_csv(out_file, index=False)

if __name__ == "__main__":
    run_modeling_pipeline()