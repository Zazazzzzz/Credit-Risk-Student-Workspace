# ---------------------------------------------------------------
# Script Name: Download Real Financial Data
# Author: Shazia
# Description: Script 2 Tasks - Get Real Data
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Script Name: RWE Credit Risk Analysis - Real Data
# Author: Shazia
# Description: Download real RWE and counterparty data
# ---------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np

# ── Download RWE real stock data ──────────────────────────────
print("=== Downloading RWE Generation SE Market Data ===\n")

# RWE AG - main company
rwe = yf.Ticker("RWE.DE")
rwe_prices = rwe.history(start="2023-01-01", end="2024-12-31")
rwe_prices = rwe_prices[['Close']].rename(columns={'Close': 'Price'})
rwe_prices.index = rwe_prices.index.strftime('%Y-%m-%d')
rwe_prices.to_csv('rwe_prices.csv')
print("✅ RWE prices saved!")

# ── Download counterparty data ────────────────────────────────
# These are companies RWE trades electricity with
counterparties = {
    'EOAN.DE':  'E.ON SE',        # ← fixed ticker!
    'ENEL.MI':  'Enel SpA',
    'ENGI.PA':  'Engie SA',
    'IBE.MC':   'Iberdrola',
}

print("\n=== Downloading Counterparty Data ===\n")
all_prices = pd.DataFrame()

for ticker, name in counterparties.items():
    try:
        company = yf.Ticker(ticker)
        prices = company.history(start="2023-01-01", end="2024-12-31")
        all_prices[name] = prices['Close']
        print(f"✅ {name} data downloaded!")
    except:
        print(f"⚠️ Could not download {name}")

all_prices.to_csv('counterparty_prices.csv')
print("\n✅ All counterparty prices saved!")
print(all_prices.head())

# ── Get RWE Financial Data ────────────────────────────────────
print("\n=== RWE Financial Summary ===\n")
rwe_info = rwe.info
print(f"Company:          {rwe_info.get('longName', 'RWE AG')}")
print(f"Market Cap:       €{rwe_info.get('marketCap', 0)/1e9:.1f} Billion")
print(f"Revenue:          €{rwe_info.get('totalRevenue', 0)/1e9:.1f} Billion")
print(f"Debt to Equity:   {rwe_info.get('debtToEquity', 'N/A')}")
print(f"Current Ratio:    {rwe_info.get('currentRatio', 'N/A')}")

# Save RWE cash flows
try:
    cashflow = rwe.cashflow
    operating_cf = cashflow.loc['Operating Cash Flow'].values.tolist()
    years = list(cashflow.columns.strftime('%Y'))
    cf_df = pd.DataFrame({
        'Year': years,
        'Operating_CF_Millions': [x/1e6 for x in operating_cf]
    })
    cf_df.to_csv('rwe_cashflows.csv', index=False)
    print(f"\n✅ RWE Cash Flows saved!")
    print(cf_df)
except:
    print("⚠️ Cash flow data not available")