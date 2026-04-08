# PHASE 1: SCRAPE THE DATA

import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://bana290-assignment1.netlify.app/"

# Fetch the live page and raise an error if the request fails
response = requests.get(URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Locate the HTML table and extract column headers from the first row
table   = soup.find('table')
rows    = table.find_all('tr')
headers = [td.get_text(strip=True) for td in rows[0].find_all(['th', 'td'])]

# Iterate over data rows — extract firm name from <strong>
# tag in column 0, and raw text from all other columns
records = []
for row in rows[1:]:
    cells = row.find_all('td')
    if not cells:
        continue
    row_data = []
    for i, cell in enumerate(cells):
        if i == 0:
            strong = cell.find('strong')
            row_data.append(strong.get_text(strip=True) if strong else cell.get_text(strip=True))
        else:
            row_data.append(cell.get_text(strip=True))
    if len(row_data) == len(headers):
        records.append(row_data)

df_raw = pd.DataFrame(records, columns=headers)
print(f"[SCRAPE] {len(df_raw)} rows, {len(df_raw.columns)} columns scraped.")
print(df_raw.head(3).to_string())

# Save raw scraped data to CSV for Phase 2 cleaning
df_raw.to_csv('raw_data.csv', index=False)
print("[SCRAPE] Saved raw_data.csv")

# PHASE 2: CLEAN THE DATA
 
import pandas as pd
import numpy as np

df = pd.read_csv('raw_data.csv')

df.columns = [
    'FIRM', 'SEGMENT', 'HQ_REGION', 'FOUNDED', 'TEAM_SIZE',
    'ANNUAL_REV', 'REV_GROWTH', 'RD_SPEND', 'AI_STATUS',
    'CLOUD_STACK', 'DIGITAL_SALES', 'COMPLIANCE_TIER',
    'FRAUD_EXPOSURE', 'FUNDING_STAGE', 'CUSTOMER_ACCTS'
]

def parse_revenue(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A', 'Unknown']:
        return np.nan
    s = str(val).lower().strip()
    s = s.replace('usd', '').replace('$', '').replace(',', '').strip()
    multiplier = 1
    if 'billion' in s or 'bn' in s:
        multiplier = 1_000_000_000
        s = s.replace('billion', '').replace('bn', '').strip()
    elif 'million' in s or ' mn' in s:
        multiplier = 1_000_000
        s = s.replace('million', '').replace('mn', '').strip()
    elif s.endswith('m'):
        multiplier = 1_000_000
        s = s[:-1].strip()
    try:
        return float(s) * multiplier
    except:
        return np.nan
 
df['ANNUAL_REV'] = df['ANNUAL_REV'].apply(parse_revenue)
 
# Convert REV_GROWTH to numeric float, stripping % and + signs
def parse_growth(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A', 'Unknown']:
        return np.nan
    try:
        return float(str(val).replace('%', '').replace('+', '').strip())
    except:
        return np.nan
 
df['REV_GROWTH'] = df['REV_GROWTH'].apply(parse_growth)

def parse_rd_spend(row):
    val = row['RD_SPEND']
    rev = row['ANNUAL_REV']
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A', 'Unknown']:
        return np.nan
    s = str(val).lower().strip()
    if 'rev' in s:
        s_clean = s.replace('% rev', '').replace('%rev', '').replace('rev', '').replace('%', '').strip()
        try:
            return (float(s_clean) / 100) * rev if not pd.isna(rev) else np.nan
        except:
            return np.nan
    s = s.replace('usd', '').replace('$', '').replace(',', '').strip()
    multiplier = 1
    if 'million' in s or ' mn' in s:
        multiplier = 1_000_000
        s = s.replace('million', '').replace('mn', '').strip()
    elif s.endswith('m'):
        multiplier = 1_000_000
        s = s[:-1].strip()
    try:
        return float(s) * multiplier
    except:
        return np.nan
 
df['RD_SPEND'] = df.apply(parse_rd_spend, axis=1)
 
ADOPTED     = {'yes', 'ai enabled', 'adopted', 'live', 'production', 'pilot'}
NOT_ADOPTED = {'no', 'not yet', 'legacy only', 'manual only'}
 
def map_ai_status(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s in ADOPTED:
        return 1
    elif s in NOT_ADOPTED:
        return 0
    return np.nan
 
df['AI_ADOPTED'] = df['AI_STATUS'].apply(map_ai_status)
 
# Convert TEAM_SIZE to numeric, handling K suffix (e.g., 1.2K)
def parse_team_size(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A']:
        return np.nan
    s = str(val).lower().replace(',', '').strip()
    if s.endswith('k'):
        try:
            return float(s[:-1]) * 1000
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan
 
df['TEAM_SIZE'] = df['TEAM_SIZE'].apply(parse_team_size)
 
# Convert DIGITAL_SALES percentage string to numeric float
def parse_pct(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A']:
        return np.nan
    try:
        return float(str(val).replace('%', '').strip())
    except:
        return np.nan
 
df['DIGITAL_SALES'] = df['DIGITAL_SALES'].apply(parse_pct)
 
# Convert CUSTOMER_ACCTS to numeric, handling K and M suffixes
def parse_customers(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A']:
        return np.nan
    s = str(val).lower().replace(',', '').strip()
    if s.endswith('k'):
        try:
            return float(s[:-1]) * 1000
        except:
            return np.nan
    if s.endswith('m'):
        try:
            return float(s[:-1]) * 1_000_000
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan
 
df['CUSTOMER_ACCTS'] = df['CUSTOMER_ACCTS'].apply(parse_customers)
 
# Convert FOUNDED to integer year
df['FOUNDED'] = pd.to_numeric(df['FOUNDED'], errors='coerce')
 
# Drop rows missing any critical variable before analysis
df.dropna(subset=['REV_GROWTH', 'AI_ADOPTED', 'ANNUAL_REV', 'TEAM_SIZE', 'FOUNDED'], inplace=True)
df.reset_index(drop=True, inplace=True)
 
print(f"[CLEAN] Final dataset: {len(df)} rows after cleaning.")
print(f"AI Adopted (1): {int(df['AI_ADOPTED'].sum())} | Not Adopted (0): {int((df['AI_ADOPTED']==0).sum())}")
print(df[['FIRM', 'ANNUAL_REV', 'REV_GROWTH', 'RD_SPEND', 'AI_ADOPTED']].head(8).to_string())
 
# Save cleaned data to CSV for Phase 3 analysis
df.to_csv('cleaned_data.csv', index=False)
print("[CLEAN] Saved cleaned_data.csv")