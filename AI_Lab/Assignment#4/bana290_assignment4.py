# BANA290 - Assignment 4: AI Intensity & Firm Innovation

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')


# STAGE 1: SCRAPE THE DATA
# Fetch the Smart Campus Incubator index page, extract
# all brief hrefs using BeautifulSoup, then loop through each
# URL to scrape its table into a separate DataFrame. Merge all
# three DataFrames on TEAM_REF to build one master dataset.

BASE_URL  = "https://bana290-assignment4.netlify.app"
INDEX_URL = BASE_URL

# Scrape index page and collect all brief page hrefs
response = requests.get(INDEX_URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

brief_links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if '/briefs/' in href:
        full_url = href if href.startswith('http') else BASE_URL + href
        if full_url not in brief_links:
            brief_links.append(full_url)

print(f"[SCRAPE] Found {len(brief_links)} brief pages.")

# Loop through each brief URL and scrape its HTML table
# extracting TEAM_REF from <strong> tag in column 0
def scrape_table(url):
    page  = requests.get(url)
    psoup = BeautifulSoup(page.text, 'html.parser')
    table = psoup.find('table')
    rows  = table.find_all('tr')
    headers = [td.get_text(strip=True) for td in rows[0].find_all(['th','td'])]
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
    return pd.DataFrame(records, columns=headers)

dfs = {}
for url in brief_links:
    df_page = scrape_table(url)
    tag = url.split('/')[-1]
    dfs[tag] = df_page
    print(f"  Scraped {len(df_page)} rows from {tag}")

# Merge all three DataFrames on TEAM_REF common identifier
df_infra   = dfs['fiber-access-bulletin']
df_metrics = dfs['builder-metrics-ledger']
df_grants  = dfs['anteater-fund-panel']

df_master = df_infra.merge(df_metrics, on='TEAM_REF') \
                    .merge(df_grants,  on='TEAM_REF')
df_master.to_csv('raw_data_a4.csv', index=False)
print(f"\n[SCRAPE] Master dataset: {len(df_master)} rows, {len(df_master.columns)} columns.")


# STAGE 2: CLEAN THE DATA
# Extract numeric values from DISTANCE_TO_NODE using
# regex, convert km to meters for a uniform unit, parse
# AI_INTENSITY and INNOVATION_SCORE removing text suffixes,
# parse ELIGIBILITY_SCORE stripping label prefixes, handle
# outliers using the IQR capping method on AI_INTENSITY and
# INNOVATION_SCORE, and create ABOVE_CUTOFF dummy for RDD.

df = df_master.copy()

# Clean DISTANCE_TO_NODE — extract numeric value and
# convert all units to meters (km * 1000)
def parse_distance(val):
    if pd.isna(val): return np.nan
    s = str(val).lower()
    num_match = re.search(r'[\d,]+\.?\d*', s.replace(',', ''))
    if not num_match: return np.nan
    num = float(num_match.group().replace(',', ''))
    if 'km' in s or 'kilometer' in s:
        return num * 1000
    return num  # already in meters

df['DISTANCE_M'] = df['DISTANCE_TO_NODE'].apply(parse_distance)

# Parse AI_INTENSITY — strip text units using regex
def parse_numeric(val):
    if pd.isna(val): return np.nan
    s = str(val).replace('~', '').replace(',', '').strip()
    match = re.search(r'[\d.]+', s)
    try: return float(match.group()) if match else np.nan
    except: return np.nan

df['AI_INTENSITY']    = df['AI_INTENSITY'].apply(parse_numeric)
df['INNOVATION_SCORE'] = df['INNOVATION_SCORE'].apply(parse_numeric)

# Parse ELIGIBILITY_SCORE — remove label prefixes like
# "Pitch rating =", "panel avg", "Score:", "points" etc.
def parse_eligibility(val):
    if pd.isna(val): return np.nan
    s = str(val)
    match = re.search(r'[\d.]+', s)
    try: return float(match.group()) if match else np.nan
    except: return np.nan

df['ELIGIBILITY_SCORE'] = df['ELIGIBILITY_SCORE'].apply(parse_eligibility)

# Handle outliers in AI_INTENSITY and INNOVATION_SCORE
# using the IQR method — clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
def iqr_clip(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr    = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series.clip(lower=lower, upper=upper)

df['AI_INTENSITY']     = iqr_clip(df['AI_INTENSITY'])
df['INNOVATION_SCORE'] = iqr_clip(df['INNOVATION_SCORE'])

# Create ABOVE_CUTOFF dummy for RDD — 1 if ELIGIBILITY_SCORE >= 85
CUTOFF = 85
df['ABOVE_CUTOFF'] = (df['ELIGIBILITY_SCORE'] >= CUTOFF).astype(int)

# Create centered running variable for RDD
df['SCORE_CENTERED'] = df['ELIGIBILITY_SCORE'] - CUTOFF

df.dropna(subset=['DISTANCE_M','AI_INTENSITY','INNOVATION_SCORE','ELIGIBILITY_SCORE'],
          inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('cleaned_data_a4.csv', index=False)

print(f"\n[CLEAN] {len(df)} rows after cleaning.")
print(df[['TEAM_REF','DISTANCE_M','AI_INTENSITY','INNOVATION_SCORE',
          'ELIGIBILITY_SCORE','ABOVE_CUTOFF']].head(8).to_string())


