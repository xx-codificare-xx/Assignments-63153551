# ============================================================
# BANA290 - Assignment 2: AI Tool Impact on Worker Productivity
# Randomized Controlled Trial (RCT) Analysis
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')


# STAGE 1: SCRAPE THE DATA

# Fetch the Loan Operations Tracker page using requests,
# parse the HTML table with BeautifulSoup, extract clerk names
# from <strong> tags, and save raw scraped data to CSV.

URL = "https://bana290-assignment2.netlify.app/"

response = requests.get(URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Locate the table and extract headers from first row
table   = soup.find('table')
rows    = table.find_all('tr')
headers = [td.get_text(strip=True) for td in rows[0].find_all(['th', 'td'])]

# Iterate over data rows — extract clerk name from <strong>
# tag in column 0, raw text from all other columns
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
df_raw.to_csv('raw_data_a2.csv', index=False)
print(f"[SCRAPE] {len(df_raw)} rows, {len(df_raw.columns)} columns scraped.")
print(df_raw.head(3).to_string())


# STAGE 2: CLEAN THE DATA

# Rename columns, use regex to extract numeric values
# from messy text fields, map TREATMENT to binary using
# predefined keyword lists, parse inconsistent timestamps
# with pd.to_datetime to compute shift duration, and drop
# rows missing critical variables.

df = df_raw.copy()

# Rename columns to clean snake_case names
df.columns = [
    'CLERK', 'CLERK_ID', 'QUEUE', 'SITE', 'SHIFT',
    'YEARS_EXP', 'BASELINE_TASKS', 'BASELINE_ERROR',
    'TRAINING_SCORE', 'TREATMENT', 'SHIFT_START', 'SHIFT_END',
    'TASKS_COMPLETED', 'ERROR_RATE'
]

# Use regex to safely extract numeric values — strip all
# characters except digits and decimal points from text fields
def extract_numeric(val):
    if pd.isna(val) or str(val).strip() in ['', '--', 'TBD', 'N/A', 'pending log']:
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(val))
    # Handle multiple dots edge case
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    try:
        return float(cleaned)
    except:
        return np.nan

# Apply regex extractor to all numeric columns
df['YEARS_EXP']       = df['YEARS_EXP'].apply(extract_numeric)
df['BASELINE_TASKS']  = df['BASELINE_TASKS'].apply(extract_numeric)
df['BASELINE_ERROR']  = df['BASELINE_ERROR'].apply(extract_numeric)
df['TRAINING_SCORE']  = df['TRAINING_SCORE'].apply(extract_numeric)
df['TASKS_COMPLETED'] = df['TASKS_COMPLETED'].apply(extract_numeric)
df['ERROR_RATE']      = df['ERROR_RATE'].apply(extract_numeric)

# Map TREATMENT to binary using predefined keyword lists
# Treatment (1) = AI tool active; Control (0) = manual / no tool
TREATMENT_LABELS = {'ai extract', 'treatment', 'assist-on',
                    'prefill enabled', 'group a'}
CONTROL_LABELS   = {'control', 'none', 'manual entry',
                    'typing only', 'group b'}

def map_treatment(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s in TREATMENT_LABELS:
        return 1
    elif s in CONTROL_LABELS:
        return 0
    return np.nan

df['TREATMENT'] = df['TREATMENT'].apply(map_treatment)

# Parse inconsistent timestamp formats using pd.to_datetime
# with errors='coerce' to handle malformed entries like 'pending log'
# and '--', then subtract to compute SHIFT_DURATION in hours
df['SHIFT_START'] = pd.to_datetime(df['SHIFT_START'], errors='coerce')
df['SHIFT_END']   = pd.to_datetime(df['SHIFT_END'],   errors='coerce', infer_datetime_format=True)
df['SHIFT_HOURS'] = (df['SHIFT_END'] - df['SHIFT_START']).dt.total_seconds() / 3600

# Drop rows with missing critical variables before analysis
df.dropna(subset=['TREATMENT', 'TASKS_COMPLETED', 'ERROR_RATE',
                  'YEARS_EXP', 'BASELINE_TASKS', 'TRAINING_SCORE'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('cleaned_data_a2.csv', index=False)

treat = df[df['TREATMENT'] == 1]
ctrl  = df[df['TREATMENT'] == 0]
print(f"\n[CLEAN] {len(df)} rows after cleaning.")
print(f"Treatment (AI): {len(treat)} | Control (Manual): {len(ctrl)}")



