# Assignment 3: AI Training Subsidy & Employment

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
# Fetch the Rust Belt Revival Labor Portal index page,
# extract all brief page hrefs using BeautifulSoup, then loop
# through each URL to scrape its employment table into a
# DataFrame, store in a list, and combine with pd.concat().

BASE_URL  = "https://bana290-assignment3.netlify.app"
INDEX_URL = BASE_URL

# Scrape the main index page to extract all brief URLs
response = requests.get(INDEX_URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Find all href links that point to individual brief pages
brief_links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if '/briefs/' in href:
        full_url = href if href.startswith('http') else BASE_URL + href
        if full_url not in brief_links:
            brief_links.append(full_url)

print(f"[SCRAPE] Found {len(brief_links)} brief pages: {brief_links}")

# Loop through each brief URL, scrape the HTML table,
# extract clerk names from <strong> tags in column 0 (REGION),
# and append each DataFrame to a list for later concatenation
all_dfs = []
for url in brief_links:
    page   = requests.get(url)
    psoup  = BeautifulSoup(page.text, 'html.parser')
    table  = psoup.find('table')
    rows   = table.find_all('tr')
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
    df_page = pd.DataFrame(records, columns=headers)
    all_dfs.append(df_page)
    print(f"  Scraped {len(df_page)} rows from {url}")

# Combine all brief DataFrames into one wide table
df_wide = pd.concat(all_dfs, ignore_index=True)
df_wide.to_csv('raw_data_a3.csv', index=False)
print(f"\n[SCRAPE] Combined: {len(df_wide)} rows, {len(df_wide.columns)} columns.")
print(df_wide[['REGION','PROGRAM_STATUS','2018','2022','2025']].to_string())


# STAGE 2: CLEAN THE DATA
# Standardize region names, assign TREATED dummy based
# on STATE_GROUP, reshape from wide to long format using melt,
# convert employment figures from mixed string formats (k, K,
# thousand, commas, ~ markers) to numeric integers, and create
# POST_POLICY dummy for years >= 2022.

df = df_wide.copy()

# Standardize region names by stripping extra whitespace
df['REGION'] = df['REGION'].str.strip()

# Assign TREATED dummy — Ohio funded corridor = 1, PA = 0
def assign_treated(state_group):
    s = str(state_group).lower()
    if 'ohio' in s:
        return 1
    elif 'pennsylvania' in s:
        return 0
    return np.nan

df['TREATED'] = df['STATE_GROUP'].apply(assign_treated)

# Reshape from wide format (years as columns) to long format
# (each row is one region-year observation) using pd.melt
year_cols = [str(y) for y in range(2018, 2026)]
df_long = df.melt(
    id_vars=['REGION', 'STATE_GROUP', 'PROGRAM_STATUS', 'ANCHOR_INDUSTRY', 'TREATED'],
    value_vars=year_cols,
    var_name='YEAR',
    value_name='EMPLOYMENT_RAW'
)
df_long['YEAR'] = df_long['YEAR'].astype(int)

# Convert employment figures from messy string formats to
# numeric integers using regex — handle k/K, thousand, commas, ~
def parse_employment(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'N/A']:
        return np.nan
    s = str(val).lower().strip()
    s = s.replace('~', '').replace('approximately', '').replace(',', '').strip()
    multiplier = 1
    if 'thousand' in s:
        multiplier = 1000
        s = s.replace('thousand', '').strip()
    elif s.endswith('k'):
        multiplier = 1000
        s = s[:-1].strip()
    # Extract numeric part using regex
    match = re.search(r'[\d.]+', s)
    if match:
        try:
            return int(float(match.group()) * multiplier)
        except:
            return np.nan
    return np.nan

df_long['EMPLOYMENT'] = df_long['EMPLOYMENT_RAW'].apply(parse_employment)

# Create POST_POLICY dummy — 1 for years >= 2022 (policy start)
df_long['POST_POLICY'] = (df_long['YEAR'] >= 2022).astype(int)

# Create DID interaction term TREATED x POST_POLICY
df_long['DID'] = df_long['TREATED'] * df_long['POST_POLICY']

# Drop rows missing employment values
df_long.dropna(subset=['EMPLOYMENT'], inplace=True)
df_long.reset_index(drop=True, inplace=True)
df_long.to_csv('cleaned_data_a3.csv', index=False)

print(f"\n[CLEAN] Long format: {len(df_long)} rows.")
print(df_long[['REGION','YEAR','EMPLOYMENT','TREATED','POST_POLICY','DID']].head(10).to_string())


