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



