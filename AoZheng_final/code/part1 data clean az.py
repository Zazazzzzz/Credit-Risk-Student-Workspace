import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = "./Data/"

def standardize_entity_code(df, col_name='entity_code'):
    df[col_name] = df[col_name].astype(str).str.strip().str.upper()
    return df

def standardize_industry_labels(df, col_name='industry'):
    df[col_name] = df[col_name].astype(str).str.strip().str.title()

    mapping = {
        'Ind. A': 'Industry A', 'Industrya': 'Industry A', 'Industry A': 'Industry A',
        'Ind. B': 'Industry B', 'Industryb': 'Industry B', 'Industry B': 'Industry B'
    }
    df[col_name] = df[col_name].map(mapping).fillna(df[col_name])

    return df

#data clean(portfolio)
def clean_portfolio(file_path):
    df = pd.read_csv(file_path)
    df = standardize_entity_code(df)
    df['EAD'] = pd.to_numeric(df['EAD'], errors='coerce')
    df = df.dropna(subset=['EAD'])
    df = df.drop_duplicates(subset='entity_code')
    df['booking_date'] = pd.to_datetime(df['booking_date'], format='mixed', errors='coerce')
    return df

#data clean(default history)
def clean_default_history(file_path):
    df = pd.read_csv(file_path)
    df = standardize_entity_code(df)
    df['date'] = pd.to_datetime(df['date'],format='mixed',errors='coerce')
    df = df.sort_values(['entity_code','date'])
    df['default_event'] = df.groupby('entity_code')['default_event'].ffill().fillna(0)
    return df

#data clean(entities_info)
def clean_entities_info(file_path):
    df = pd.read_csv(file_path)
    df = standardize_entity_code(df)
    df = standardize_industry_labels(df)
    df['lgd'] = pd.to_numeric(df['lgd'], errors='coerce')
    df['lgd'] = df['lgd'].fillna(df['lgd'].median())
    df = df.drop_duplicates(subset=['entity_code'])
    return df

#CLEAN
def build_master_table(base_path):

    p1 = clean_portfolio(base_path + 'portfolio_1.csv')
    p2 = clean_portfolio(base_path + 'portfolio_2.csv')
    info = clean_entities_info(base_path + 'entities_info.csv')
    hist = clean_default_history(base_path + 'default_history_20y.csv')

    portfolio_all = pd.concat([p1, p2], ignore_index=True)
    portfolio_all = portfolio_all.drop_duplicates(subset=['entity_code'])

    portfolio_all = standardize_entity_code(portfolio_all)
    info = standardize_entity_code(info)

    fix_dict = {'BWMAEKB': 'B-WMAEKB', 'BSQ4OLC': 'B-SQ4OLC', 'BJS2LBB': 'B-JS2LBB', 'BY6J2QG': 'B-Y6J2QG'}
    portfolio_all['entity_code'] = portfolio_all['entity_code'].replace(fix_dict)

#Average PD
    hist_pd = hist.groupby('entity_code')['default_event'].mean().reset_index()
    hist_pd = hist_pd.rename(columns={'default_event': 'avg_pd'})

#Merge
    master_df = portfolio_all.merge(info, on='entity_code', how='left')
    master_df = master_df.merge(hist_pd, on='entity_code', how='left')

    global_avg_pd = master_df['avg_pd'].mean()
    master_df['avg_pd'] = master_df['avg_pd'].fillna(global_avg_pd)

    return master_df

DATA_DIR = "./Data/"
#table 1
master_table = build_master_table(DATA_DIR)
master_table.to_csv(DATA_DIR + 'cleaned_master_table.csv', index=False)
print("ok")