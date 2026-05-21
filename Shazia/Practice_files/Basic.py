# ---------------------------------------------------------------
# Script Name: ch1_basics_raw_data.py
# Author: Shazia Ishaq
# Description: Chapter 1 - Calculating ALL variables from RAW data
#              No pre-computed values - we calculate everything ourselves!
#
# WHY FROM RAW DATA?
# When you work at RWE, Deutsche Bank, or BaFin as a credit risk
# analyst, you will always be asked:
# "How did you calculate this number?"
# You must be able to explain EVERY step.
# Calculating from raw data means you fully understand and control
# every number in your analysis.
# ---------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

print("=" * 60)
print("   CHAPTER 1: CALCULATING FROM RAW DATA")
print("   Market Cap, Beta, Debt - all from scratch!")
print("=" * 60)

# ===============================================================
# STEP 1: DEFINE WHICH COMPANIES AND MARKET INDEX TO DOWNLOAD
# ===============================================================
# We download RAW DAILY PRICES for all companies
# AND the DAX index (German stock market benchmark)
#
# DAX = Deutscher Aktienindex = Germany's main stock index
# Contains 40 largest German companies including RWE and E.ON
# We use DAX as our "market" for beta calculation
#
# For non-German companies we could use their local index
# but DAX is fine for a European energy portfolio

company_tickers = {
    'RWE AG':    'RWE.DE',
    'E.ON SE':   'EOAN.DE',
    'Engie SA':  'ENGI.PA',
    'Enel SpA':  'ENEL.MI',
    'Iberdrola': 'IBE.MC',
}

market_ticker = '^GDAXI'   # DAX index = German stock market
start_date = '2021-01-01'
end_date   = '2024-12-31'  # 3 years of daily data

# ===============================================================
# STEP 2: DOWNLOAD RAW DAILY PRICES
# ===============================================================
# We download the CLOSING PRICE for each company every trading day
# Closing price = the final price at end of each trading day
#
# This is the most basic raw data in financial markets
# Everything else (returns, volatility, beta) is calculated FROM this

print("\nDownloading raw daily closing prices...")
print(f"Period: {start_date} to {end_date}")
print()

raw_prices = pd.DataFrame()

for name, ticker in company_tickers.items():
    try:
        data = yf.download(ticker, start=start_date,
                           end=end_date, progress=False)
        raw_prices[name] = data['Close']
        print(f"  {name} ({ticker}): {len(data)} daily prices loaded")
    except Exception as e:
        print(f"  {name}: Error - {e}")

# Download DAX market index for beta calculation
print(f"\nDownloading DAX market index ({market_ticker})...")
try:
    dax_data = yf.download(market_ticker, start=start_date,
                           end=end_date, progress=False)
    dax_prices = dax_data['Close']
    print(f"  DAX: {len(dax_prices)} daily prices loaded")
except Exception as e:
    print(f"  DAX: Error - {e}")

print(f"\nRaw price data shape: {raw_prices.shape}")
print(f"(rows=trading days, columns=companies)")
print(f"\nFirst 5 rows of raw prices (euros):")
print(raw_prices.head())

# ===============================================================
# STEP 3: CALCULATE MARKET CAPITALIZATION FROM RAW DATA
# ===============================================================
# Market Cap = Stock Price × Number of Shares Outstanding
#
# Stock Price: we have this from the download (raw_prices)
# Shares Outstanding: we get this from yfinance company info
#
# WHY DO WE CALCULATE THIS OURSELVES?
# Yahoo Finance gives us marketCap pre-calculated, but:
# - It might use a different date than we want
# - We want to understand WHAT market cap actually means
# - In a real job you often need to calculate it at a specific date
#
# WHAT DOES MARKET CAP TELL US FOR CREDIT RISK?
# Large market cap = large company = generally more stable
# Small market cap = small company = potentially more risky
# In Merton model: market cap represents equity value of the firm

print("\n" + "=" * 60)
print("STEP 3: Calculate Market Capitalization from Raw Data")
print("Market Cap = Latest Stock Price × Shares Outstanding")
print("=" * 60)

market_caps = {}

for name, ticker in company_tickers.items():
    try:
        company = yf.Ticker(ticker)

        # Get shares outstanding from company info
        # This IS from info dict because shares outstanding
        # is not calculable from price data alone
        shares = company.info.get('sharesOutstanding', None)

        # Get the LATEST closing price from our raw data
        # iloc[-1] means "last row" = most recent day
        latest_price = raw_prices[name].iloc[-1]

        if shares and not np.isnan(latest_price):
            # Calculate market cap ourselves!
            # shares × price = total market value
            market_cap = shares * latest_price
            market_caps[name] = market_cap

            print(f"\n  {name}:")
            print(f"    Latest price:       €{latest_price:.2f}")
            print(f"    Shares outstanding: {shares/1e6:.0f} million")
            print(f"    Market Cap:         €{market_cap/1e9:.2f} billion")
            print(f"    Formula: {shares/1e6:.0f}M × €{latest_price:.2f} "
                  f"= €{market_cap/1e9:.2f}B")
        else:
            print(f"\n  {name}: Could not calculate market cap")

    except Exception as e:
        print(f"\n  {name}: Error - {e}")

# ===============================================================
# STEP 4: CALCULATE DAILY RETURNS FROM RAW PRICES
# ===============================================================
# Return = how much the price changed from yesterday to today
#
# FORMULA:
# Return_t = (Price_t - Price_{t-1}) / Price_{t-1}
#           = Price_t / Price_{t-1} - 1
#
# In Python: .pct_change() does this automatically
# pct_change() = percentage change = (today - yesterday) / yesterday
#
# Example:
# Monday:  E.ON = €10.00
# Tuesday: E.ON = €10.30
# Return = (10.30 - 10.00) / 10.00 = 0.03 = 3% gain
#
# WHY DO WE NEED RETURNS?
# Returns are used to calculate:
# 1. Volatility (how risky is the stock)
# 2. Beta (how much does it move with the market)
# 3. VaR (maximum expected daily loss)
# All three are key credit risk metrics!

print("\n" + "=" * 60)
print("STEP 4: Calculate Daily Returns from Raw Prices")
print("Return = (Today's Price - Yesterday's Price) / Yesterday's Price")
print("=" * 60)

# Calculate daily returns for all companies
# pct_change() automatically calculates row-by-row percentage change
daily_returns = raw_prices.pct_change()

# Remove first row (NaN because no previous day to compare to)
daily_returns = daily_returns.dropna()

# Calculate market (DAX) returns the same way
market_returns = dax_prices.pct_change().dropna()

# Align dates (both must have same dates to calculate beta)
# inner join keeps only dates that exist in BOTH datasets
common_dates = daily_returns.index.intersection(market_returns.index)
daily_returns = daily_returns.loc[common_dates]
market_returns = market_returns.loc[common_dates]

print(f"\nReturns calculated for {len(daily_returns)} trading days")
print(f"\nSample daily returns (first 5 days):")
print(daily_returns.head())

print(f"\nReturn Statistics Summary:")
print(daily_returns.describe().round(4))

# ===============================================================
# STEP 5: CALCULATE ANNUAL VOLATILITY FROM RAW RETURNS
# ===============================================================
# Volatility = how much the stock price jumps around
# = standard deviation of daily returns × √252
#
# WHY × √252?
# There are 252 trading days per year (not 365 - weekends excluded!)
# Daily volatility needs to be scaled to annual volatility
# This is called "annualization" using the square root of time rule
#
# Example:
# E.ON daily return std = 1.5% per day
# Annual volatility = 1.5% × √252 = 1.5% × 15.87 = 23.8% per year
#
# WHY IS VOLATILITY IMPORTANT FOR CREDIT RISK?
# In Merton's model:
# High volatility = asset value swings a lot = more chance it falls below debt
# = HIGHER probability of default!
# Iberdrola (renewables) → low volatility → safer credit
# Small energy trader → high volatility → riskier credit

print("\n" + "=" * 60)
print("STEP 5: Calculate Annual Volatility from Raw Returns")
print("Volatility = Std(daily returns) × √252")
print("(252 = number of trading days per year)")
print("=" * 60)

volatilities = {}

print(f"\n{'Company':<15} {'Daily Std':>12} {'Annual Vol':>12} {'Risk Level':>15}")
print("-" * 57)

for name in daily_returns.columns:
    daily_std = daily_returns[name].std()
    annual_vol = daily_std * np.sqrt(252)
    volatilities[name] = annual_vol

    if annual_vol < 0.15:
        risk = "Low risk"
    elif annual_vol < 0.25:
        risk = "Moderate risk"
    elif annual_vol < 0.35:
        risk = "High risk"
    else:
        risk = "Very high risk"

    print(f"{name:<15} {daily_std*100:>11.2f}% {annual_vol*100:>11.2f}% "
          f"{risk:>15}")

print()
print("Interpretation:")
print("Higher annual volatility → asset value swings more wildly")
print("→ Higher chance of falling below debt threshold")
print("→ HIGHER probability of default in Merton model!")

# ===============================================================
# STEP 6: CALCULATE BETA FROM RAW DATA
# ===============================================================
# Beta measures: how much does this stock move relative to the market?
#
# FORMULA:
# Beta = Covariance(stock returns, market returns)
#        ─────────────────────────────────────────
#        Variance(market returns)
#
# In simple words:
# Numerator (Covariance): how much stock and market move TOGETHER
# Denominator (Variance): how much does the market move on its own
# Beta = ratio of these two
#
# PRACTICAL MEANING:
# Beta = 0.5: if market falls 2%, this stock falls only 1%
#             (half as sensitive = safer for credit purposes)
# Beta = 1.0: stock moves exactly with the market
# Beta = 1.5: if market falls 2%, this stock falls 3%
#             (1.5x more sensitive = riskier)
# Beta = 0.3: very stable (typical for water/electricity utilities)
#
# DO ALL COMPANIES HAVE BETA?
# Listed companies (RWE, E.ON, Engie, Enel, Iberdrola): YES
# → We calculate from their real daily stock returns vs DAX
#
# Private/unlisted companies (Vattenfall, small firms): NO
# → Solution: use "industry beta" = average beta of similar listed companies
# → Vattenfall ≈ average of RWE + E.ON beta (same industry)
#
# HOW BETA HELPS IN CREDIT RISK:
# High beta company = amplified moves during market stress
# = more likely to be in financial trouble when economy is bad
# = higher default risk in bad economic scenarios!

print("\n" + "=" * 60)
print("STEP 6: Calculate Beta from Raw Data")
print("Beta = Cov(stock returns, DAX returns) / Var(DAX returns)")
print("=" * 60)

betas = {}

print(f"\n{'Company':<15} {'Beta':>8} {'Meaning':>30}")
print("-" * 57)

for name in daily_returns.columns:
    stock_ret = daily_returns[name].values
    market_ret = market_returns.values

    # Remove any NaN values from both arrays simultaneously
    mask = ~(np.isnan(stock_ret) | np.isnan(market_ret))
    stock_clean = stock_ret[mask]
    market_clean = market_ret[mask]

    # Calculate covariance matrix
    # np.cov returns a 2x2 matrix:
    # [[Var(stock), Cov(stock,market)],
    #  [Cov(market,stock), Var(market)]]
    cov_matrix = np.cov(stock_clean, market_clean)

    # Extract covariance between stock and market
    # [0,1] means: row 0 (stock), column 1 (market)
    covariance = cov_matrix[0, 1]

    # Extract variance of market
    # [1,1] means: row 1 (market), column 1 (market)
    market_variance = cov_matrix[1, 1]

    # Calculate beta
    beta = covariance / market_variance
    betas[name] = beta

    if beta < 0.5:
        meaning = "Very stable (defensive stock)"
    elif beta < 0.8:
        meaning = "Stable (below market risk)"
    elif beta < 1.2:
        meaning = "Average (moves with market)"
    else:
        meaning = "Volatile (above market risk)"

    print(f"{name:<15} {beta:>8.3f} {meaning:>30}")

print()
print("Verification - alternative calculation using correlation:")
for name in list(daily_returns.columns)[:2]:
    stock_ret = daily_returns[name].dropna()
    mkt_ret = market_returns.reindex(stock_ret.index).dropna()
    idx = stock_ret.index.intersection(mkt_ret.index)

    corr = stock_ret[idx].corr(mkt_ret[idx])
    vol_stock = stock_ret[idx].std()
    vol_market = mkt_ret[idx].std()

    # Alternative beta formula: Beta = Corr × (Vol_stock / Vol_market)
    beta_alt = corr * (vol_stock / vol_market)
    print(f"  {name}: Beta (cov/var) = {betas[name]:.3f}, "
          f"Beta (corr method) = {beta_alt:.3f} ← should match!")

# ===============================================================
# STEP 7: GET TOTAL DEBT FROM FINANCIAL STATEMENTS
# ===============================================================
# Total Debt = all money a company owes to banks and bondholders
#
# WHY CAN'T WE CALCULATE DEBT FROM PRICE DATA?
# Unlike market cap (price × shares), debt is in the BALANCE SHEET
# not in the stock price. We MUST look at financial statements.
#
# WHERE TO FIND DEBT DATA:
# Option A: yfinance balance sheet (automatic)
# Option B: Company annual report (manual, most accurate)
# Option C: Financial data providers (Bloomberg, Refinitiv)
#
# TYPES OF DEBT:
# Short-term debt: must be repaid within 1 year
# Long-term debt: repaid over many years (bonds, bank loans)
# Total debt = short + long term
#
# WHY IS DEBT IMPORTANT FOR CREDIT RISK?
# In Merton's model: B = total debt = DEFAULT THRESHOLD
# If company asset value falls below total debt → DEFAULT!
# High debt relative to assets = higher default risk!

print("\n" + "=" * 60)
print("STEP 7: Get Total Debt from Balance Sheet")
print("(Cannot calculate from prices - must read financial statements)")
print("=" * 60)

debts = {}

print(f"\n{'Company':<15} {'Short-term':>14} {'Long-term':>14} "
      f"{'Total Debt':>14} {'Debt/Assets':>12}")
print("-" * 70)

for name, ticker in company_tickers.items():
    try:
        company = yf.Ticker(ticker)

        # Get balance sheet data
        # This contains all assets, liabilities, equity
        balance_sheet = company.balance_sheet

        if balance_sheet is not None and not balance_sheet.empty:
            # Most recent year (first column)
            latest = balance_sheet.iloc[:, 0]

            # Extract debt components
            # Different companies report differently so we try multiple names
            short_debt = 0
            long_debt = 0
            total_assets = 0

            for field in ['Current Debt', 'Short Long Term Debt',
                          'Short Term Debt']:
                if field in latest.index:
                    val = latest[field]
                    if pd.notna(val):
                        short_debt = val
                        break

            for field in ['Long Term Debt', 'Long Term Debt And Capital Lease']:
                if field in latest.index:
                    val = latest[field]
                    if pd.notna(val):
                        long_debt = val
                        break

            for field in ['Total Assets']:
                if field in latest.index:
                    val = latest[field]
                    if pd.notna(val):
                        total_assets = val
                        break

            total_debt = short_debt + long_debt
            debt_to_assets = total_debt / total_assets if total_assets > 0 else 0
            debts[name] = total_debt

            print(f"{name:<15} €{short_debt/1e9:>12.1f}B "
                  f"€{long_debt/1e9:>12.1f}B "
                  f"€{total_debt/1e9:>12.1f}B "
                  f"{debt_to_assets*100:>11.1f}%")
        else:
            print(f"{name:<15} Balance sheet not available")

    except Exception as e:
        print(f"{name:<15} Error: {e}")

# ===============================================================
# STEP 8: COMBINE ALL CALCULATED VARIABLES
# ===============================================================
# Now we have calculated EVERYTHING from raw data:
# - Market Cap: from raw stock price × shares outstanding
# - Daily returns: from raw price changes
# - Annual volatility: from raw daily return standard deviation × √252
# - Beta: from raw covariance(stock, DAX) / variance(DAX)
# - Total debt: from raw balance sheet financial statements
#
# Now we can build our credit risk portfolio!

print("\n" + "=" * 60)
print("STEP 8: Complete Portfolio Summary")
print("All variables calculated from raw data!")
print("=" * 60)

print(f"\n{'Company':<15} {'Mkt Cap':>10} {'Debt':>10} "
      f"{'Volatility':>12} {'Beta':>8} {'Debt/Cap':>10}")
print("-" * 68)

for name in company_tickers.keys():
    cap = market_caps.get(name, 0)
    debt = debts.get(name, 0)
    vol = volatilities.get(name, 0)
    beta = betas.get(name, 0)
    debt_cap = debt / cap if cap > 0 else 0

    print(f"{name:<15} €{cap/1e9:>8.1f}B €{debt/1e9:>8.1f}B "
          f"{vol*100:>11.1f}% {beta:>8.3f} {debt_cap*100:>9.1f}%")

print()
print("KEY INSIGHT - Debt/Cap ratio:")
print("Low Debt/Cap  → company has more assets than debt → safer")
print("High Debt/Cap → company heavily indebted → riskier!")
print()
print("Beta interpretation for credit risk:")
print("Low beta  → stable company → less likely to default in crisis")
print("High beta → volatile company → more likely to struggle in downturn")

# ===============================================================
# STEP 9: NOW DO THE EL CALCULATION WITH REAL DATA
# ===============================================================
# Use the real volatility we calculated to estimate PD
# Higher volatility → higher PD (from Merton model logic)
#
# SIMPLE PD ESTIMATION FROM VOLATILITY:
# This is a simplified approach - real Merton model in Chapter 6
# For now: PD ≈ norm.cdf((log(Debt/Assets) / Volatility))
# Simple rule of thumb: higher vol → higher PD

from scipy.stats import norm

print("\n" + "=" * 60)
print("STEP 9: Expected Loss Using Real Calculated Variables")
print("=" * 60)

# Assumed exposures (how much RWE trades with each company)
exposures = {
    'RWE AG':    0,            # RWE is US! not a counterparty
    'E.ON SE':   50_000_000,
    'Engie SA':  30_000_000,
    'Enel SpA':  40_000_000,
    'Iberdrola': 25_000_000,
}

# Standard LGD assumption for energy trading
lgd_assumption = 0.40  # 40% loss given default

print(f"\n{'Company':<15} {'Volatility':>12} {'Est. PD':>10} "
      f"{'Exposure':>12} {'EL':>12}")
print("-" * 65)

total_el = 0
total_exp = 0

for name in ['E.ON SE', 'Engie SA', 'Enel SpA', 'Iberdrola']:
    vol = volatilities.get(name, 0.20)

    # Simple PD estimate: higher volatility → higher PD
    # This is simplified - Chapter 6 covers the full Merton formula
    # Rule of thumb: PD ≈ 0.5% for low vol, 2% for high vol
    pd_estimate = max(0.001, vol * 0.01)

    exposure = exposures.get(name, 0)

    # EL = PD × LGD × Exposure
    el = pd_estimate * lgd_assumption * exposure

    total_el += el
    total_exp += exposure

    print(f"{name:<15} {vol*100:>11.1f}% {pd_estimate*100:>9.3f}% "
          f"€{exposure/1e6:>10.0f}M €{el:>10,.0f}")

print("-" * 65)
print(f"{'TOTAL':<15} {'':>12} {'':>10} "
      f"€{total_exp/1e6:>10.0f}M €{total_el:>10,.0f}")

print(f"\nTotal Portfolio EL: €{total_el:,.0f}")
print(f"EL as % of exposure: {total_el/total_exp*100:.4f}%")
print()
print("NOTE: PD is estimated simply from volatility here.")
print("For more accurate PD, see Chapter 6 (Merton model)")
print("where we use the full formula with debt and asset values.")

# ===============================================================
# STEP 10: PLOT EVERYTHING
# ===============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Credit Risk Portfolio Analysis\nCalculated from Raw Market Data',
             fontsize=13, fontweight='bold')

companies = [n for n in company_tickers.keys() if n != 'RWE AG']

# Plot 1: Annual Volatility
vols = [volatilities.get(n, 0)*100 for n in companies]
axes[0,0].bar(companies, vols, color='steelblue', edgecolor='black')
axes[0,0].set_title('Annual Volatility (calculated from raw returns)')
axes[0,0].set_ylabel('Volatility (%)')
axes[0,0].tick_params(axis='x', rotation=30)
axes[0,0].grid(True, axis='y', alpha=0.4)
for i, v in enumerate(vols):
    axes[0,0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)

# Plot 2: Beta
beta_vals = [betas.get(n, 0) for n in company_tickers.keys()]
beta_names = list(company_tickers.keys())
axes[0,1].bar(beta_names, beta_vals, color='coral', edgecolor='black')
axes[0,1].axhline(y=1.0, color='red', linestyle='--',
                   linewidth=1.5, label='Market Beta = 1.0')
axes[0,1].set_title('Beta (calculated from raw returns vs DAX)')
axes[0,1].set_ylabel('Beta')
axes[0,1].tick_params(axis='x', rotation=30)
axes[0,1].legend()
axes[0,1].grid(True, axis='y', alpha=0.4)

# Plot 3: Market Cap vs Debt
cap_vals = [market_caps.get(n, 0)/1e9 for n in companies]
debt_vals = [debts.get(n, 0)/1e9 for n in companies]
x = np.arange(len(companies))
width = 0.35
axes[1,0].bar(x - width/2, cap_vals, width, label='Market Cap',
              color='green', edgecolor='black', alpha=0.8)
axes[1,0].bar(x + width/2, debt_vals, width, label='Total Debt',
              color='red', edgecolor='black', alpha=0.8)
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(companies, rotation=30)
axes[1,0].set_title('Market Cap vs Total Debt\n(from raw financial data)')
axes[1,0].set_ylabel('€ Billions')
axes[1,0].legend()
axes[1,0].grid(True, axis='y', alpha=0.4)

# Plot 4: Correlation matrix of returns
corr_matrix = daily_returns.corr()
im = axes[1,1].imshow(corr_matrix.values, cmap='RdYlGn',
                       vmin=-1, vmax=1)
axes[1,1].set_xticks(range(len(corr_matrix.columns)))
axes[1,1].set_yticks(range(len(corr_matrix.columns)))
axes[1,1].set_xticklabels(corr_matrix.columns, rotation=45, fontsize=8)
axes[1,1].set_yticklabels(corr_matrix.columns, fontsize=8)
axes[1,1].set_title('Return Correlation Matrix\n(systemic risk indicator)')
plt.colorbar(im, ax=axes[1,1])
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
        axes[1,1].text(j, i, f'{corr_matrix.iloc[i,j]:.2f}',
                       ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('ch1_raw_data_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDashboard saved as ch1_raw_data_dashboard.png")

print("\n" + "=" * 60)
print("SUMMARY: What we calculated from RAW data:")
print("=" * 60)
print("Market Cap  = raw stock price × shares outstanding")
print("Returns     = (today price - yesterday price) / yesterday")
print("Volatility  = std(daily returns) × √252")
print("Beta        = cov(stock, DAX) / var(DAX)")
print("Debt        = from raw balance sheet (short + long term)")
print("EL          = estimated PD × LGD × Exposure")
print()
print("You can now explain EVERY number to your professor or employer!")