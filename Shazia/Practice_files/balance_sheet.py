# ---------------------------------------------------------------
# Script Name: balance_sheet_explained.py
# Author: Shazia Ishaq
# Description: Understanding Balance Sheets for Credit Risk Analysis
#
# This file explains:
# 1. What a balance sheet is and how it looks
# 2. How to download real balance sheet data using yfinance
# 3. Which items matter for credit risk analysis
# 4. How risk analysts use balance sheets at RWE/Deutsche Bank
# ---------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===============================================================
# PART 1: WHAT IS A BALANCE SHEET?
# ===============================================================
# A balance sheet is a snapshot of a company's financial position
# at ONE specific point in time (usually December 31st each year).
#
# It answers the question:
# "If we stopped the company today, what does it own and owe?"
#
# THE FUNDAMENTAL EQUATION:
# ASSETS = LIABILITIES + EQUITY
#
# Assets     = everything the company OWNS (factories, cash, etc.)
# Liabilities= everything the company OWES (debts, loans, etc.)
# Equity     = what's left for shareholders (Assets - Liabilities)
#
# REAL LIFE ANALOGY:
# Think of your own personal balance sheet:
#
# YOUR ASSETS:                YOUR LIABILITIES:
# Car:         €15,000        Car loan:     €10,000
# Savings:     €5,000         Student loan: €20,000
# Laptop:      €1,000         Credit card:  €2,000
# ─────────────              ─────────────────────
# Total:       €21,000        Total debt:   €32,000
#
# YOUR EQUITY = €21,000 - €32,000 = -€11,000 (negative = you owe more than you own!)
#
# For E.ON SE (2023 Annual Report):
# Total Assets:      ~€90 billion
# Total Liabilities: ~€70 billion
# Total Equity:      ~€20 billion
# (Assets = Liabilities + Equity → 90 = 70 + 20 ✓)

print("=" * 70)
print("   BALANCE SHEET STRUCTURE - EXPLAINED FOR CREDIT RISK")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║              E.ON SE SIMPLIFIED BALANCE SHEET (2023)                ║
║              Source: E.ON Annual Report 2023 (eon.com)              ║
╠══════════════════════════════════════════════════════════════════════╣
║  ASSETS (what E.ON OWNS)        LIABILITIES (what E.ON OWES)        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CURRENT ASSETS (short-term)    CURRENT LIABILITIES (due <1 year)   ║
║  Cash & Equivalents:  €5.2B     Short-term debt:        €3.8B       ║
║  Trade Receivables:   €8.1B     Trade Payables:         €4.2B       ║
║  Inventories:         €1.8B     Other current liab.:    €9.5B       ║
║  Other current:       €4.2B     ─────────────────────────────       ║
║  ─────────────────────────      Total Current Liab.:   €17.5B       ║
║  Total Current:      €19.3B                                          ║
║                                 LONG-TERM LIABILITIES (>1 year)      ║
║  NON-CURRENT ASSETS             Long-term debt (bonds): €32.5B      ║
║  (long-term investments)        Pension obligations:     €5.2B       ║
║  Property & Equipment:€18.5B   Deferred tax:            €2.8B       ║
║  Intangibles/Goodwill:€12.3B   Other long-term:         €7.0B       ║
║  Financial investments:€28.4B  ─────────────────────────────        ║
║  Deferred tax assets:  €4.2B   Total Long-term Liab.:  €47.5B       ║
║  Other non-current:    €7.3B                                         ║
║  ─────────────────────────      ─────────────────────────────        ║
║  Total Non-current:   €70.7B   TOTAL LIABILITIES:      €65.0B       ║
║                                                                      ║
║                                 SHAREHOLDERS EQUITY                  ║
║                                 Share capital:          €2.0B        ║
║                                 Retained earnings:      €12.0B       ║
║                                 Other equity:           €11.0B       ║
║                                 ─────────────────────────────        ║
║                                 TOTAL EQUITY:           €25.0B       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  TOTAL ASSETS:       €90.0B    TOTAL LIABILITIES + EQUITY: €90.0B  ║
║                                (must be equal - the fundamental rule)║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ===============================================================
# PART 2: DOWNLOAD REAL BALANCE SHEET DATA USING YFINANCE
# ===============================================================
# yfinance gives us the actual balance sheet data from companies
# It comes from SEC filings (USA) or local regulatory filings
#
# HOW TO DOWNLOAD:
# company = yf.Ticker('EOAN.DE')
# balance_sheet = company.balance_sheet
# This gives a DataFrame where:
# - Rows = different balance sheet items
# - Columns = different years (usually 4 years of history)

print("=" * 70)
print("   PART 2: DOWNLOADING REAL BALANCE SHEET DATA")
print("=" * 70)

print("\nDownloading E.ON SE balance sheet from Yahoo Finance...")

try:
    eon = yf.Ticker('EOAN.DE')
    balance_sheet = eon.balance_sheet

    if balance_sheet is not None and not balance_sheet.empty:
        print("\nSUCCESS! Real E.ON balance sheet loaded.")
        print(f"Years available: {[str(c.date()) for c in balance_sheet.columns]}")
        print(f"Number of items: {len(balance_sheet)}")
        print("\nALL available balance sheet items:")
        print("-" * 50)
        for item in balance_sheet.index:
            latest_val = balance_sheet.iloc[
                balance_sheet.index.get_loc(item), 0]
            if pd.notna(latest_val) and latest_val != 0:
                print(f"  {item}: €{latest_val/1e9:.2f}B")
    else:
        print("Note: Balance sheet not available via API.")
        print("Using illustrative data based on E.ON Annual Report 2023.")
        print("In PyCharm with internet: run company.balance_sheet")

except Exception as e:
    print(f"Note: {e}")
    print("Using illustrative data instead.")

# ===============================================================
# PART 3: CREATE REALISTIC BALANCE SHEET DATA
# ===============================================================
# Below we create a realistic E.ON balance sheet
# based on their published Annual Report 2023
# This is what you would see when yfinance works properly
#
# All values in billions of euros (€B)
# Source: E.ON SE Annual Report 2023 (eon.com/investors)

print("\n" + "=" * 70)
print("   PART 3: E.ON BALANCE SHEET - 4 YEARS OF HISTORY")
print("   Based on E.ON Annual Reports 2020-2023")
print("=" * 70)

# Real E.ON balance sheet data (approximate from annual reports)
# Years as columns, items as rows
eon_balance_sheet = pd.DataFrame({
    '2023': {
        # ASSETS
        'Cash And Cash Equivalents':        5.2e9,
        'Receivables':                      8.1e9,
        'Inventory':                        1.8e9,
        'Other Current Assets':             4.2e9,
        'Total Current Assets':            19.3e9,
        'Net PPE':                         18.5e9,   # Property, Plant, Equipment
        'Goodwill':                        12.3e9,   # Value of acquisitions
        'Investments And Advances':        28.4e9,
        'Other Non Current Assets':        11.5e9,
        'Total Non Current Assets':        70.7e9,
        'Total Assets':                    90.0e9,
        # LIABILITIES
        'Current Debt':                     3.8e9,   # Short-term debt
        'Accounts Payable':                 4.2e9,
        'Other Current Liabilities':        9.5e9,
        'Total Current Liabilities':       17.5e9,
        'Long Term Debt':                  32.5e9,   # KEY for credit risk!
        'Long Term Provisions':             5.2e9,
        'Other Non Current Liabilities':    9.8e9,
        'Total Non Current Liabilities':   47.5e9,
        'Total Liabilities':               65.0e9,
        # EQUITY
        'Common Stock':                     2.0e9,
        'Retained Earnings':               12.0e9,
        'Total Stockholder Equity':        25.0e9,
    },
    '2022': {
        'Cash And Cash Equivalents':        4.8e9,
        'Receivables':                      9.5e9,
        'Inventory':                        2.1e9,
        'Other Current Assets':             5.8e9,
        'Total Current Assets':            22.2e9,
        'Net PPE':                         17.8e9,
        'Goodwill':                        12.5e9,
        'Investments And Advances':        26.8e9,
        'Other Non Current Assets':        10.9e9,
        'Total Non Current Assets':        68.0e9,
        'Total Assets':                    90.2e9,
        'Current Debt':                     4.2e9,
        'Accounts Payable':                 5.1e9,
        'Other Current Liabilities':       11.8e9,
        'Total Current Liabilities':       21.1e9,
        'Long Term Debt':                  33.8e9,
        'Long Term Provisions':             5.5e9,
        'Other Non Current Liabilities':   10.8e9,
        'Total Non Current Liabilities':   50.1e9,
        'Total Liabilities':               71.2e9,
        'Common Stock':                     2.0e9,
        'Retained Earnings':               10.5e9,
        'Total Stockholder Equity':        19.0e9,
    },
    '2021': {
        'Cash And Cash Equivalents':        5.5e9,
        'Receivables':                      7.2e9,
        'Inventory':                        1.5e9,
        'Other Current Assets':             3.8e9,
        'Total Current Assets':            18.0e9,
        'Net PPE':                         17.2e9,
        'Goodwill':                        12.8e9,
        'Investments And Advances':        24.5e9,
        'Other Non Current Assets':        10.5e9,
        'Total Non Current Assets':        65.0e9,
        'Total Assets':                    83.0e9,
        'Current Debt':                     3.5e9,
        'Accounts Payable':                 3.8e9,
        'Other Current Liabilities':        8.2e9,
        'Total Current Liabilities':       15.5e9,
        'Long Term Debt':                  31.5e9,
        'Long Term Provisions':             5.8e9,
        'Other Non Current Liabilities':    9.2e9,
        'Total Non Current Liabilities':   46.5e9,
        'Total Liabilities':               62.0e9,
        'Common Stock':                     2.0e9,
        'Retained Earnings':               10.0e9,
        'Total Stockholder Equity':        21.0e9,
    },
    '2020': {
        'Cash And Cash Equivalents':        4.2e9,
        'Receivables':                      6.8e9,
        'Inventory':                        1.4e9,
        'Other Current Assets':             3.5e9,
        'Total Current Assets':            15.9e9,
        'Net PPE':                         16.5e9,
        'Goodwill':                        13.0e9,
        'Investments And Advances':        22.8e9,
        'Other Non Current Assets':        10.0e9,
        'Total Non Current Assets':        62.3e9,
        'Total Assets':                    78.2e9,
        'Current Debt':                     3.2e9,
        'Accounts Payable':                 3.5e9,
        'Other Current Liabilities':        7.5e9,
        'Total Current Liabilities':       14.2e9,
        'Long Term Debt':                  30.0e9,
        'Long Term Provisions':             6.0e9,
        'Other Non Current Liabilities':    8.5e9,
        'Total Non Current Liabilities':   44.5e9,
        'Total Liabilities':               58.7e9,
        'Common Stock':                     2.0e9,
        'Retained Earnings':               9.5e9,
        'Total Stockholder Equity':        19.5e9,
    }
})

print("\nE.ON SE Balance Sheet (€ Billions):")
print("-" * 65)
print(f"{'Item':<35} {'2023':>8} {'2022':>8} {'2021':>8} {'2020':>8}")
print("-" * 65)

key_items = [
    'Cash And Cash Equivalents',
    'Total Current Assets',
    'Total Assets',
    '─── LIABILITIES ───',
    'Current Debt',
    'Long Term Debt',
    'Total Liabilities',
    '─── EQUITY ───',
    'Total Stockholder Equity',
]

for item in key_items:
    if item.startswith('───'):
        print(f"\n{item}")
        continue
    if item in eon_balance_sheet.index:
        row = eon_balance_sheet.loc[item]
        print(f"  {item:<33} "
              f"{row['2023']/1e9:>7.1f}B "
              f"{row['2022']/1e9:>7.1f}B "
              f"{row['2021']/1e9:>7.1f}B "
              f"{row['2020']/1e9:>7.1f}B")

# ===============================================================
# PART 4: CREDIT RISK METRICS FROM BALANCE SHEET
# ===============================================================
# Risk analysts extract SPECIFIC items from the balance sheet
# to calculate credit risk metrics.
#
# KEY ITEMS FOR CREDIT RISK:
#
# 1. TOTAL DEBT = Current Debt + Long-Term Debt
#    This is the DEFAULT THRESHOLD in Merton's model (B)
#    If asset value falls below total debt → DEFAULT!
#
# 2. DEBT-TO-EQUITY RATIO = Total Debt / Total Equity
#    Measures financial leverage (how much debt vs equity)
#    Higher ratio = more indebted = riskier!
#    Rule of thumb: >3.0 = concerning, >5.0 = very risky
#
# 3. DEBT-TO-ASSETS RATIO = Total Debt / Total Assets
#    What % of assets are financed by debt?
#    Higher = less buffer before insolvency
#
# 4. CURRENT RATIO = Current Assets / Current Liabilities
#    Can the company pay its bills in the next 12 months?
#    <1.0 = danger! Cannot cover short-term obligations
#    >1.5 = comfortable
#
# 5. INTEREST COVERAGE = EBIT / Interest Expense
#    Can the company afford its interest payments?
#    (Needs income statement too - see Part 5)
#
# 6. CASH RATIO = Cash / Current Liabilities
#    Most liquid safety measure
#    Can the company survive a cash crunch?

print("\n" + "=" * 70)
print("   PART 4: CREDIT RISK METRICS FROM BALANCE SHEET")
print("   These are what risk analysts calculate every quarter!")
print("=" * 70)

print(f"\n{'Metric':<35} {'2023':>8} {'2022':>8} {'2021':>8} "
      f"{'Trend':>10} {'Risk Signal':>15}")
print("-" * 90)

years = ['2023', '2022', '2021', '2020']

# Metric 1: Total Debt
total_debt = {}
for yr in years:
    current = eon_balance_sheet.loc['Current Debt', yr]
    longterm = eon_balance_sheet.loc['Long Term Debt', yr]
    total_debt[yr] = current + longterm

print(f"\n  1. TOTAL DEBT (€B):")
vals = [total_debt[yr]/1e9 for yr in ['2023','2022','2021']]
trend = "↑ Rising" if vals[0] > vals[2] else "↓ Falling"
risk = "CAUTION" if vals[0] > 38 else "OK"
print(f"     {'Total Debt':<31} "
      f"{vals[0]:>7.1f}B {vals[1]:>7.1f}B {vals[2]:>7.1f}B "
      f"{trend:>10} {risk:>15}")
print(f"     → Merton threshold B = {vals[0]:.1f}B "
      f"(company defaults if assets < {vals[0]:.1f}B)")

# Metric 2: Debt-to-Equity
print(f"\n  2. DEBT-TO-EQUITY RATIO:")
for yr in ['2023', '2022', '2021']:
    td = total_debt[yr]
    equity = eon_balance_sheet.loc['Total Stockholder Equity', yr]
    de_ratio = td / equity
    risk_level = "HIGH RISK" if de_ratio > 2.5 else \
                 "MODERATE" if de_ratio > 1.5 else "LOW RISK"
    print(f"     {yr}: €{td/1e9:.1f}B debt / €{equity/1e9:.1f}B equity "
          f"= {de_ratio:.2f}x  → {risk_level}")
print(f"     → Rule: >2.5x = concerning for credit analysts")
print(f"     → Energy companies typically have higher D/E than others")
print(f"       because they own expensive infrastructure")

# Metric 3: Current Ratio
print(f"\n  3. CURRENT RATIO (liquidity check):")
for yr in ['2023', '2022', '2021']:
    ca = eon_balance_sheet.loc['Total Current Assets', yr]
    cl = eon_balance_sheet.loc['Total Current Liabilities', yr]
    cr = ca / cl
    signal = "DANGER: cannot cover short-term debts!" if cr < 1.0 else \
             "CAUTION: tight liquidity" if cr < 1.2 else \
             "OK: comfortable liquidity"
    print(f"     {yr}: {ca/1e9:.1f}B / {cl/1e9:.1f}B = {cr:.2f}  → {signal}")

# Metric 4: Cash Ratio
print(f"\n  4. CASH RATIO (most conservative liquidity):")
for yr in ['2023', '2022', '2021']:
    cash = eon_balance_sheet.loc['Cash And Cash Equivalents', yr]
    cl = eon_balance_sheet.loc['Total Current Liabilities', yr]
    cash_ratio = cash / cl
    signal = "LOW cash cushion" if cash_ratio < 0.2 else \
             "Adequate cash" if cash_ratio < 0.5 else "Strong cash"
    print(f"     {yr}: €{cash/1e9:.1f}B cash / €{cl/1e9:.1f}B "
          f"current liab = {cash_ratio:.2f}  → {signal}")

# ===============================================================
# PART 5: HOW RISK ANALYSTS ACTUALLY USE THIS DATA
# ===============================================================
# At RWE, Deutsche Bank, or any credit institution, the workflow is:
#
# STEP 1: COLLECT DATA (quarterly, when companies publish reports)
#   - Download balance sheet, income statement, cash flow statement
#   - Use Bloomberg, Refinitiv, or yfinance
#
# STEP 2: CALCULATE RATIOS (automated Python/Excel)
#   - All the metrics above
#   - Track changes over time (trending better or worse?)
#
# STEP 3: ASSIGN INTERNAL CREDIT RATING
#   - Based on ratios + qualitative factors
#   - "E.ON is BBB because D/E = 1.4x, current ratio = 1.1x,
#      strong market position, government-linked"
#
# STEP 4: CALCULATE EXPECTED LOSS
#   - EL = PD (from rating) × LGD × Exposure
#
# STEP 5: SET CREDIT LIMITS
#   - Maximum exposure allowed based on EL and VaR
#   - "RWE can trade up to €100M with E.ON based on their BBB rating"
#
# STEP 6: MONITOR CONTINUOUSLY
#   - Watch for rating downgrades
#   - Monitor quarterly balance sheet changes
#   - Alert when ratios deteriorate

print("\n" + "=" * 70)
print("   PART 5: HOW RISK ANALYSTS WORK WITH BALANCE SHEETS")
print("=" * 70)

print("""
REAL WORKFLOW AT RWE GENERATION SE CREDIT RISK TEAM:

Monday morning, 8:00 AM:
━━━━━━━━━━━━━━━━━━━━━━
1. Automated system downloads new financial data
   (When E.ON publishes quarterly results)

2. Python script calculates all credit ratios:
   - Debt/Equity, Current Ratio, Cash Ratio etc.
   - Compares to previous quarter: improving or deteriorating?

3. If any ratio crosses a threshold → ALERT sent to risk team:
   "E.ON current ratio fell from 1.15 to 0.98 - BELOW 1.0!"
   "Action required: review credit limit"

4. Credit analyst opens Python dashboard:
   - Sees 4 years of balance sheet trends
   - Identifies: "E.ON's debt increased by €2B this quarter"
   - Runs Merton model to update PD estimate

5. Committee meeting at 10:00 AM:
   "Should we reduce our €100M exposure to E.ON to €70M?"
   Decision based on updated credit analysis.

6. End of day: Updated credit report filed in system
   Next review scheduled: next quarter earnings date.
""")

# ===============================================================
# PART 6: COMPARE BALANCE SHEETS OF MULTIPLE COMPANIES
# ===============================================================
# Risk analysts never look at just one company in isolation
# They compare a counterparty to its PEERS
# "Is E.ON more or less risky than Engie?"

print("=" * 70)
print("   PART 6: COMPARING COMPANIES - PEER ANALYSIS")
print("   How risk analysts benchmark counterparties")
print("=" * 70)

# Simplified 2023 data for all 5 companies
# Based on published annual reports
companies_bs = {
    'E.ON SE': {
        'total_assets':   90.0e9,
        'total_debt':     36.3e9,  # 3.8 + 32.5
        'total_equity':   25.0e9,
        'cash':            5.2e9,
        'current_assets': 19.3e9,
        'current_liab':   17.5e9,
        'revenue':        93.0e9,  # from income statement
        'ebit':            3.5e9,  # earnings before interest & tax
        'interest_exp':    1.2e9,
    },
    'Engie SA': {
        'total_assets':  180.0e9,
        'total_debt':     65.0e9,
        'total_equity':   42.0e9,
        'cash':            8.5e9,
        'current_assets': 55.0e9,
        'current_liab':   50.0e9,
        'revenue':        82.0e9,
        'ebit':            6.2e9,
        'interest_exp':    2.1e9,
    },
    'Enel SpA': {
        'total_assets':  195.0e9,
        'total_debt':     72.0e9,
        'total_equity':   52.0e9,
        'cash':            7.2e9,
        'current_assets': 48.0e9,
        'current_liab':   45.0e9,
        'revenue':       140.0e9,
        'ebit':            9.5e9,
        'interest_exp':    2.8e9,
    },
    'Iberdrola': {
        'total_assets':  150.0e9,
        'total_debt':     55.0e9,
        'total_equity':   45.0e9,
        'cash':            5.8e9,
        'current_assets': 28.0e9,
        'current_liab':   25.0e9,
        'revenue':        42.0e9,
        'ebit':            8.2e9,
        'interest_exp':    1.9e9,
    },
    'Vattenfall': {
        'total_assets':   55.0e9,
        'total_debt':     18.0e9,
        'total_equity':   20.0e9,
        'cash':            3.5e9,
        'current_assets': 15.0e9,
        'current_liab':   12.0e9,
        'revenue':        25.0e9,
        'ebit':            4.2e9,
        'interest_exp':    0.6e9,
    },
}

print(f"\n{'Metric':<25} {'E.ON':>10} {'Engie':>10} "
      f"{'Enel':>10} {'Iberdrola':>10} {'Vattenfall':>11}")
print("=" * 80)

metrics_to_show = [
    ('Total Assets (€B)', lambda d: d['total_assets']/1e9, '{:.0f}B'),
    ('Total Debt (€B)', lambda d: d['total_debt']/1e9, '{:.0f}B'),
    ('Equity (€B)', lambda d: d['total_equity']/1e9, '{:.0f}B'),
    ('Debt/Equity ratio', lambda d: d['total_debt']/d['total_equity'], '{:.2f}x'),
    ('Debt/Assets ratio', lambda d: d['total_debt']/d['total_assets'], '{:.1%}'),
    ('Current ratio', lambda d: d['current_assets']/d['current_liab'], '{:.2f}x'),
    ('Cash ratio', lambda d: d['cash']/d['current_liab'], '{:.2f}x'),
    ('Interest coverage', lambda d: d['ebit']/d['interest_exp'], '{:.1f}x'),
]

for metric_name, calc_func, fmt in metrics_to_show:
    print(f"{metric_name:<25}", end='')
    for company in companies_bs.keys():
        val = calc_func(companies_bs[company])
        print(f" {fmt.format(val):>10}", end='')
    print()

print()
print("INTERPRETATION GUIDE:")
print("Debt/Equity:    Lower = safer. <1.5x = good, >2.5x = concerning")
print("Debt/Assets:    Lower = safer. <50% = good, >70% = concerning")
print("Current Ratio:  Higher = safer. >1.2 = good, <1.0 = danger!")
print("Cash Ratio:     Higher = safer. >0.2 = adequate")
print("Interest Cov.:  Higher = safer. >3x = good, <1.5x = danger!")

# ===============================================================
# PART 7: VISUALIZE BALANCE SHEET DATA
# ===============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E.ON SE Balance Sheet Analysis for Credit Risk\n'
             'Based on Annual Reports 2020-2023',
             fontsize=14, fontweight='bold')

company_names = list(companies_bs.keys())
colors = ['#1a3a5c', '#2e6da4', '#4a90d9', '#7bb3e8', '#aed0f0']

# Chart 1: Assets vs Debt vs Equity trend for E.ON
years_plot = ['2020', '2021', '2022', '2023']
assets = [eon_balance_sheet.loc['Total Assets', yr]/1e9
          for yr in years_plot]
debts_plot = [(eon_balance_sheet.loc['Current Debt', yr] +
               eon_balance_sheet.loc['Long Term Debt', yr])/1e9
              for yr in years_plot]
equity = [eon_balance_sheet.loc['Total Stockholder Equity', yr]/1e9
          for yr in years_plot]

x = np.arange(len(years_plot))
width = 0.25
axes[0,0].bar(x - width, assets, width, label='Total Assets',
              color='green', alpha=0.8, edgecolor='black')
axes[0,0].bar(x, debts_plot, width, label='Total Debt',
              color='red', alpha=0.8, edgecolor='black')
axes[0,0].bar(x + width, equity, width, label='Equity',
              color='blue', alpha=0.8, edgecolor='black')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(years_plot)
axes[0,0].set_title('E.ON: Assets vs Debt vs Equity (2020-2023)')
axes[0,0].set_ylabel('€ Billions')
axes[0,0].legend()
axes[0,0].grid(True, axis='y', alpha=0.4)

# Chart 2: Debt/Equity ratio trend
de_ratios = [d/e for d, e in zip(debts_plot, equity)]
color_bars = ['green' if r < 1.5 else 'orange' if r < 2.5 else 'red'
              for r in de_ratios]
axes[0,1].bar(years_plot, de_ratios, color=color_bars, edgecolor='black')
axes[0,1].axhline(y=1.5, color='orange', linestyle='--',
                   linewidth=2, label='Caution threshold (1.5x)')
axes[0,1].axhline(y=2.5, color='red', linestyle='--',
                   linewidth=2, label='Danger threshold (2.5x)')
axes[0,1].set_title('E.ON: Debt-to-Equity Ratio Trend')
axes[0,1].set_ylabel('D/E Ratio (x)')
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, axis='y', alpha=0.4)

# Chart 3: Peer comparison - Debt/Assets
peer_de = [companies_bs[n]['total_debt'] /
           companies_bs[n]['total_assets'] * 100
           for n in company_names]
bar_colors_peer = ['red' if r > 50 else 'orange' if r > 40 else 'green'
                   for r in peer_de]
bars = axes[1,0].bar(company_names, peer_de,
                      color=bar_colors_peer, edgecolor='black')
axes[1,0].axhline(y=50, color='red', linestyle='--',
                   linewidth=2, label='50% threshold')
axes[1,0].set_title('Peer Comparison: Debt/Assets Ratio (2023)')
axes[1,0].set_ylabel('Debt as % of Assets')
axes[1,0].tick_params(axis='x', rotation=30)
axes[1,0].legend()
axes[1,0].grid(True, axis='y', alpha=0.4)
for bar, val in zip(bars, peer_de):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=9)

# Chart 4: Interest coverage comparison
int_cov = [companies_bs[n]['ebit'] /
           companies_bs[n]['interest_exp']
           for n in company_names]
bar_colors_ic = ['red' if r < 2 else 'orange' if r < 3 else 'green'
                 for r in int_cov]
bars = axes[1,1].bar(company_names, int_cov,
                      color=bar_colors_ic, edgecolor='black')
axes[1,1].axhline(y=3.0, color='orange', linestyle='--',
                   linewidth=2, label='Minimum comfort (3x)')
axes[1,1].axhline(y=1.5, color='red', linestyle='--',
                   linewidth=2, label='Danger level (1.5x)')
axes[1,1].set_title('Peer Comparison: Interest Coverage (2023)')
axes[1,1].set_ylabel('EBIT / Interest Expense (x)')
axes[1,1].tick_params(axis='x', rotation=30)
axes[1,1].legend(fontsize=9)
axes[1,1].grid(True, axis='y', alpha=0.4)
for bar, val in zip(bars, int_cov):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.05,
                   f'{val:.1f}x', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('balance_sheet_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as balance_sheet_analysis.png")

# ===============================================================
# PART 8: FINAL CREDIT ASSESSMENT BASED ON BALANCE SHEET
# ===============================================================

print("\n" + "=" * 70)
print("   PART 8: CREDIT ASSESSMENT BASED ON BALANCE SHEET")
print("   This is what a risk analyst writes in their report!")
print("=" * 70)

for name, data in companies_bs.items():
    de_ratio = data['total_debt'] / data['total_equity']
    current_ratio = data['current_assets'] / data['current_liab']
    int_coverage = data['ebit'] / data['interest_exp']
    debt_assets = data['total_debt'] / data['total_assets']

    # Simple scoring system (real models are more complex!)
    score = 0
    if de_ratio < 1.5: score += 2
    elif de_ratio < 2.5: score += 1

    if current_ratio > 1.2: score += 2
    elif current_ratio > 1.0: score += 1

    if int_coverage > 4: score += 2
    elif int_coverage > 2.5: score += 1

    if debt_assets < 0.40: score += 2
    elif debt_assets < 0.55: score += 1

    if score >= 7: rating = "A (Low Risk)"
    elif score >= 5: rating = "BBB (Moderate Risk)"
    elif score >= 3: rating = "BB (Elevated Risk)"
    else: rating = "B (High Risk)"

    print(f"\n  {name}:")
    print(f"    Debt/Equity:       {de_ratio:.2f}x")
    print(f"    Current Ratio:     {current_ratio:.2f}x")
    print(f"    Interest Coverage: {int_coverage:.1f}x")
    print(f"    Debt/Assets:       {debt_assets:.1%}")
    print(f"    → Suggested Rating: {rating} (Score: {score}/8)")

print("""
NOTE: Real credit ratings combine:
  1. Quantitative: balance sheet ratios (what we calculated above)
  2. Qualitative:  management quality, market position, regulation
  3. Macroeconomic: country risk, industry outlook, energy transition
  4. Historical:   track record of paying debts on time

At Deutsche Bank or RWE, dedicated credit analysts spend days
on each major counterparty to produce a comprehensive credit report.
Our Python code automates the quantitative part!
""")