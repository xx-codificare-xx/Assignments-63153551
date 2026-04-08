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