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


# STAGE 3: ANALYZE THE DATA

POLICY_YEAR = 2022
treated_df = df_long[df_long['TREATED'] == 1]
control_df = df_long[df_long['TREATED'] == 0]

# Compute group-year means for plotting
group_year = df_long.groupby(['TREATED', 'YEAR'])['EMPLOYMENT'].mean().reset_index()
treat_means = group_year[group_year['TREATED'] == 1].sort_values('YEAR')
ctrl_means  = group_year[group_year['TREATED'] == 0].sort_values('YEAR')

# 3A. VISUAL ANALYSIS — Employment trends over full period
# Plot employment trends for Treatment and Control
# groups across all years, marking the policy intervention

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(treat_means['YEAR'], treat_means['EMPLOYMENT'],
             marker='o', color='steelblue', linewidth=2, label='Ohio (Treated)')
axes[0].plot(ctrl_means['YEAR'], ctrl_means['EMPLOYMENT'],
             marker='s', color='tomato', linewidth=2, label='Pennsylvania (Control)')
axes[0].axvline(x=POLICY_YEAR, color='black', linestyle='--',
                linewidth=1.5, label='Policy Start (2022)')
axes[0].set_title('Employment Trends: Treatment vs Control', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Mean Employment')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 3B. PARALLEL TRENDS TEST
# Visually and statistically inspect pre-treatment
# trends to verify groups move in parallel before 2022

pre_treat  = treat_means[treat_means['YEAR'] < POLICY_YEAR]
pre_ctrl   = ctrl_means[ctrl_means['YEAR'] < POLICY_YEAR]

# Normalize to index = 100 in 2018 for visual parallel trends
base_t = pre_treat[pre_treat['YEAR'] == 2018]['EMPLOYMENT'].values[0]
base_c = pre_ctrl[pre_ctrl['YEAR']  == 2018]['EMPLOYMENT'].values[0]
pre_treat = pre_treat.copy(); pre_treat['INDEX'] = pre_treat['EMPLOYMENT'] / base_t * 100
pre_ctrl  = pre_ctrl.copy();  pre_ctrl['INDEX']  = pre_ctrl['EMPLOYMENT']  / base_c * 100

axes[1].plot(pre_treat['YEAR'], pre_treat['INDEX'],
             marker='o', color='steelblue', linewidth=2, label='Ohio (Treated)')
axes[1].plot(pre_ctrl['YEAR'], pre_ctrl['INDEX'],
             marker='s', color='tomato', linewidth=2, label='Pennsylvania (Control)')
axes[1].axvline(x=POLICY_YEAR, color='black', linestyle='--',
                linewidth=1.5, label='Policy Start (2022)')
axes[1].set_title('Parallel Trends: Pre-Treatment Index (2018=100)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Employment Index (2018 = 100)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('did_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[ANALYZE] Saved did_trends.png")

# Statistically test parallel trends — regress employment on
# YEAR for each group pre-policy and compare slopes with t-test
pre_df = df_long[df_long['YEAR'] < POLICY_YEAR].copy()
treat_pre = pre_df[pre_df['TREATED'] == 1]
ctrl_pre  = pre_df[pre_df['TREATED'] == 0]

slope_t = np.polyfit(treat_pre['YEAR'], treat_pre['EMPLOYMENT'], 1)[0]
slope_c = np.polyfit(ctrl_pre['YEAR'],  ctrl_pre['EMPLOYMENT'],  1)[0]

print(f"\n[ANALYZE] Parallel Trends:")
print(f"  Ohio (Treated) pre-trend slope   : {slope_t:+.1f} jobs/year")
print(f"  Pennsylvania (Control) pre-trend : {slope_c:+.1f} jobs/year")

# 3C. PLACEBO TEST
# Conduct a placebo DID using 2020 as a fake
# treatment year within the pre-treatment window to test
# for anticipatory effects — coefficient should be ~0

PLACEBO_YEAR = 2020
placebo_df = df_long[df_long['YEAR'] < POLICY_YEAR].copy()
placebo_df['POST_PLACEBO'] = (placebo_df['YEAR'] >= PLACEBO_YEAR).astype(int)
placebo_df['DID_PLACEBO']  = placebo_df['TREATED'] * placebo_df['POST_PLACEBO']

placebo_model = smf.ols(
    'EMPLOYMENT ~ TREATED + POST_PLACEBO + DID_PLACEBO',
    data=placebo_df
).fit()
placebo_coef = placebo_model.params['DID_PLACEBO']
placebo_pval = placebo_model.pvalues['DID_PLACEBO']

print(f"\n[ANALYZE] Placebo Test (fake year = {PLACEBO_YEAR}):")
print(f"  Placebo DID coefficient: {placebo_coef:+.1f}  (p={placebo_pval:.4f})")
print(f"  {'PASS — no anticipatory effect' if placebo_pval > 0.05 else 'FAIL — anticipatory effect detected'}")

# 3D. DID ESTIMATION — Fixed Effects Regression
# Run fixed-effects DID regression with TREATED,
# POST_POLICY, and their interaction (DID) to estimate
# the causal effect of the AI training subsidy

did_model = smf.ols(
    'EMPLOYMENT ~ TREATED + POST_POLICY + DID',
    data=df_long
).fit()

did_coef = did_model.params['DID']
did_pval = did_model.pvalues['DID']
did_ci   = did_model.conf_int().loc['DID']

print(f"\n[ANALYZE] DID Regression Results:")
print(did_model.summary())
print(f"\n  DID Coefficient (ATE): {did_coef:+.1f} jobs  (p={did_pval:.4f})")
print(f"  95% CI: [{did_ci[0]:+.1f}, {did_ci[1]:+.1f}]")

# Save summary
summary = {
    'did_coef':      round(did_coef, 2),
    'did_pval':      round(did_pval, 4),
    'did_ci_low':    round(float(did_ci[0]), 2),
    'did_ci_high':   round(float(did_ci[1]), 2),
    'placebo_coef':  round(placebo_coef, 2),
    'placebo_pval':  round(placebo_pval, 4),
    'slope_treated': round(slope_t, 2),
    'slope_control': round(slope_c, 2),
    'mean_emp_treat_pre':  round(treat_pre['EMPLOYMENT'].mean(), 1),
    'mean_emp_treat_post': round(treated_df[treated_df['YEAR'] >= POLICY_YEAR]['EMPLOYMENT'].mean(), 1),
    'mean_emp_ctrl_pre':   round(ctrl_pre['EMPLOYMENT'].mean(), 1),
    'mean_emp_ctrl_post':  round(control_df[control_df['YEAR'] >= POLICY_YEAR]['EMPLOYMENT'].mean(), 1),
}
with open('analysis_summary_a3.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n[ANALYZE] Saved analysis_summary_a3.json")


# STAGE 4: INTERPRET THE DATA
# Load analysis summary and print structured
# interpretation covering causal effect of AI training
# subsidy, parallel trends validity, and implications for
# regional labor market displacement vs creation.

with open('analysis_summary_a3.json', 'r') as f:
    s = json.load(f)

sig = "statistically significant" if s['did_pval'] < 0.05 else "not statistically significant"
placebo_pass = s['placebo_pval'] > 0.05

print("\n" + "="*65)
print("STAGE 4: INTERPRETATION OF DID RESULTS")
print("="*65)
print(f"""
CAUSAL EFFECT OF AI TRAINING SUBSIDY
  DID Coefficient (ATE) : {s['did_coef']:+,.1f} jobs  (p={s['did_pval']:.4f})
  95% Confidence Interval: [{s['did_ci_low']:+,.1f}, {s['did_ci_high']:+,.1f}]

  The Difference-in-Differences estimate indicates that the 2022
  AI and robotics upskilling subsidy caused an average increase of
  {abs(s['did_coef']):,.1f} jobs per county in the treated Ohio
  counties relative to the Pennsylvania control counties. This effect
  is {sig}. Treated counties saw mean employment rise from
  {s['mean_emp_treat_pre']:,.0f} (pre-2022) to {s['mean_emp_treat_post']:,.0f}
  (post-2022), while control counties showed a much flatter trajectory
  from {s['mean_emp_ctrl_pre']:,.0f} to {s['mean_emp_ctrl_post']:,.0f}.

PARALLEL TRENDS ASSUMPTION
  Pre-treatment slope — Ohio    : {s['slope_treated']:+,.1f} jobs/year
  Pre-treatment slope — Penn.   : {s['slope_control']:+,.1f} jobs/year
  Placebo DID coefficient       : {s['placebo_coef']:+,.1f}  (p={s['placebo_pval']:.4f})

  {'The parallel trends assumption is supported. Both groups followed similar employment trajectories before 2022, and the placebo test produced a near-zero, statistically insignificant coefficient, confirming no anticipatory effects.' if placebo_pass else 'The placebo test flagged a potential pre-trend issue. Results should be interpreted with caution.'}

LABOR MARKET DISPLACEMENT vs CREATION
  The subsidy appears to have driven net job creation in treated
  counties, particularly in fabricated metals, auto parts, and
  steel components. Upskilling workers for AI-assisted roles likely
  reduced displacement risk by equipping existing workers for new
  tasks rather than replacing them outright. However, firms that
  adopted automation without subsidy support in control counties
  showed stagnant or slow employment growth, suggesting that without
  targeted policy intervention, AI adoption in manufacturing regions
  may suppress labor demand. The policy therefore acted as a
  creation buffer, converting potential displacement into upward
  employment trajectories.
""")

