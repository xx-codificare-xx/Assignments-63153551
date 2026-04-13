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


# STAGE 3: ANALYZE THE DATA
# Run balance tests to verify randomization, perform
# t-tests for ignorability assumption, justify SUTVA, and
# compute ATE for productivity (TASKS_COMPLETED) and quality
# (ERROR_RATE). Save boxplots for the tex file.

print("\n" + "="*60)
print("STAGE 3A: BALANCE TEST")
print("="*60)

# Compare baseline characteristics between treatment and
# control groups to verify that randomization was successful
baseline_vars = ['YEARS_EXP', 'BASELINE_TASKS', 'BASELINE_ERROR', 'TRAINING_SCORE']

balance_rows = []
for var in baseline_vars:
    t_mean = treat[var].mean()
    c_mean = ctrl[var].mean()
    t_std  = treat[var].std()
    c_std  = ctrl[var].std()
    t_stat, p_val = stats.ttest_ind(treat[var].dropna(), ctrl[var].dropna())
    balance_rows.append({
        'Variable':        var,
        'Treatment Mean':  round(t_mean, 3),
        'Control Mean':    round(c_mean, 3),
        'Treatment Std':   round(t_std, 3),
        'Control Std':     round(c_std, 3),
        't-stat':          round(t_stat, 3),
        'p-value':         round(p_val, 4),
        'Balanced?':       'Yes' if p_val > 0.05 else 'No'
    })

balance_df = pd.DataFrame(balance_rows)
print(balance_df.to_string(index=False))

print("\n" + "="*60)
print("STAGE 3B: IGNORABILITY ASSUMPTION (t-tests)")
print("="*60)
# Verify ignorability — no statistically significant
# differences in pre-treatment variables between groups
for _, row in balance_df.iterrows():
    status = "PASS" if row['p-value'] > 0.05 else "FAIL"
    print(f"  {row['Variable']:<22}: p={row['p-value']:.4f}  [{status}]")

print("\n" + "="*60)
print("STAGE 3C: SUTVA JUSTIFICATION")
print("="*60)
print("""
  SUTVA (Stable Unit Treatment Value Assumption) requires:
  1. No spillover effects between units.
  2. Only one version of the treatment exists.

  Justification: In this RCT, clerks process loan applications
  independently on individual workstations. Each clerk's output
  is self-contained — one clerk's use of the AI tool does not
  affect another clerk's task queue or error rate. The AI tool
  (PDF pre-fill extraction) operates at the individual session
  level, and clerks were physically separated across Day and
  Evening shifts at two sites (Irvine and Phoenix). Group
  assignments were randomized within each site and queue,
  further minimizing any risk of spillover. SUTVA is satisfied.
""")

print("="*60)
print("STAGE 3D: ATE ESTIMATION")
print("="*60)

# Calculate ATE for productivity — mean difference in
# TASKS_COMPLETED between treatment and control groups
ate_tasks = treat['TASKS_COMPLETED'].mean() - ctrl['TASKS_COMPLETED'].mean()
t_stat_tasks, p_tasks = stats.ttest_ind(treat['TASKS_COMPLETED'], ctrl['TASKS_COMPLETED'])

# Calculate ATE for quality — mean difference in
# ERROR_RATE between treatment and control groups
ate_error = treat['ERROR_RATE'].mean() - ctrl['ERROR_RATE'].mean()
t_stat_err, p_error = stats.ttest_ind(treat['ERROR_RATE'], ctrl['ERROR_RATE'])

print(f"\n  PRODUCTIVITY (TASKS_COMPLETED):")
print(f"    Treatment Mean : {treat['TASKS_COMPLETED'].mean():.2f} tasks")
print(f"    Control Mean   : {ctrl['TASKS_COMPLETED'].mean():.2f} tasks")
print(f"    ATE            : {ate_tasks:+.2f} tasks  (p={p_tasks:.4f})")

print(f"\n  QUALITY (ERROR_RATE):")
print(f"    Treatment Mean : {treat['ERROR_RATE'].mean():.2f}%")
print(f"    Control Mean   : {ctrl['ERROR_RATE'].mean():.2f}%")
print(f"    ATE            : {ate_error:+.2f}%  (p={p_error:.4f})")

# Run OLS regression of TASKS_COMPLETED on TREATMENT
# controlling for baseline covariates to confirm ATE estimate
X_prod = sm.add_constant(df[['TREATMENT', 'YEARS_EXP', 'BASELINE_TASKS', 'TRAINING_SCORE']])
ols_prod = sm.OLS(df['TASKS_COMPLETED'], X_prod).fit()

X_qual = sm.add_constant(df[['TREATMENT', 'YEARS_EXP', 'BASELINE_ERROR', 'TRAINING_SCORE']])
ols_qual = sm.OLS(df['ERROR_RATE'], X_qual).fit()

print(f"\n  OLS (controlling for baseline covariates):")
print(f"    TASKS_COMPLETED ~ TREATMENT coef: {ols_prod.params['TREATMENT']:+.4f}  (p={ols_prod.pvalues['TREATMENT']:.4f})")
print(f"    ERROR_RATE      ~ TREATMENT coef: {ols_qual.params['TREATMENT']:+.4f}  (p={ols_qual.pvalues['TREATMENT']:.4f})")

# Generate boxplots of productivity and error rate
# by treatment group for inclusion in the tex report
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Productivity boxplot
data_tasks = [ctrl['TASKS_COMPLETED'].dropna(), treat['TASKS_COMPLETED'].dropna()]
bp1 = axes[0].boxplot(data_tasks, patch_artist=True,
                       tick_labels=['Control\n(Manual)', 'Treatment\n(AI Tool)'])
bp1['boxes'][0].set_facecolor('#f28b82')
bp1['boxes'][1].set_facecolor('#87c5f5')
axes[0].set_title('Productivity: Tasks Completed per Shift', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Tasks Completed')
axes[0].grid(axis='y', alpha=0.3)
axes[0].annotate(f'ATE = {ate_tasks:+.2f}\np = {p_tasks:.4f}',
                 xy=(0.72, 0.05), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Error rate boxplot
data_error = [ctrl['ERROR_RATE'].dropna(), treat['ERROR_RATE'].dropna()]
bp2 = axes[1].boxplot(data_error, patch_artist=True,
                       tick_labels=['Control\n(Manual)', 'Treatment\n(AI Tool)'])
bp2['boxes'][0].set_facecolor('#f28b82')
bp2['boxes'][1].set_facecolor('#87c5f5')
axes[1].set_title('Quality: Error Rate per Shift', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Error Rate (%)')
axes[1].grid(axis='y', alpha=0.3)
axes[1].annotate(f'ATE = {ate_error:+.2f}%\np = {p_error:.4f}',
                 xy=(0.72, 0.88), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('rct_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[ANALYZE] Saved rct_boxplots.png")

# Save summary for interpretation and tex file
summary = {
    'n_treatment': int(len(treat)),
    'n_control':   int(len(ctrl)),
    'ate_tasks':   round(ate_tasks, 4),
    'p_tasks':     round(p_tasks, 4),
    'ate_error':   round(ate_error, 4),
    'p_error':     round(p_error, 4),
    'mean_tasks_treat': round(treat['TASKS_COMPLETED'].mean(), 2),
    'mean_tasks_ctrl':  round(ctrl['TASKS_COMPLETED'].mean(), 2),
    'mean_error_treat': round(treat['ERROR_RATE'].mean(), 2),
    'mean_error_ctrl':  round(ctrl['ERROR_RATE'].mean(), 2),
    'ols_treat_coef_tasks': round(float(ols_prod.params['TREATMENT']), 4),
    'ols_treat_pval_tasks': round(float(ols_prod.pvalues['TREATMENT']), 4),
    'ols_treat_coef_error': round(float(ols_qual.params['TREATMENT']), 4),
    'ols_treat_pval_error': round(float(ols_qual.pvalues['TREATMENT']), 4),
    'balance': [
        {'var': r['Variable'], 't_mean': r['Treatment Mean'],
         'c_mean': r['Control Mean'], 'pval': r['p-value'], 'balanced': r['Balanced?']}
        for _, r in balance_df.iterrows()
    ]
}
with open('analysis_summary_a2.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("[ANALYZE] Saved analysis_summary_a2.json")


# STAGE 4: INTERPRET THE DATA
# Load analysis summary and print structured
# interpretation covering marginal productivity gain,
# quantity-quality trade-off, and implications for firms
# considering broad deployment of AI assistants.

with open('analysis_summary_a2.json', 'r') as f:
    s = json.load(f)

print("\n" + "="*65)
print("STAGE 4: INTERPRETATION OF RCT RESULTS")
print("="*65)

sig_tasks = "statistically significant" if s['p_tasks'] < 0.05 else "not statistically significant"
sig_error = "statistically significant" if s['p_error'] < 0.05 else "not statistically significant"
direction = "increase" if s['ate_error'] > 0 else "decrease"

print(f"""
MARGINAL PRODUCTIVITY GAIN
  Treatment Mean Tasks : {s['mean_tasks_treat']} tasks/shift
  Control Mean Tasks   : {s['mean_tasks_ctrl']} tasks/shift
  ATE (Productivity)   : {s['ate_tasks']:+.2f} tasks  (p={s['p_tasks']:.4f}, {sig_tasks})

  Clerks assigned to the AI extraction tool completed on average
  {abs(s['ate_tasks']):.2f} more tasks per shift than manual clerks.
  This represents a {abs(s['ate_tasks']/s['mean_tasks_ctrl']*100):.1f}% productivity gain
  attributable directly to the AI tool.

QUANTITY-QUALITY TRADE-OFF
  Treatment Mean Error : {s['mean_error_treat']}%
  Control Mean Error   : {s['mean_error_ctrl']}%
  ATE (Error Rate)     : {s['ate_error']:+.2f}%  (p={s['p_error']:.4f}, {sig_error})

  The AI tool was associated with a {abs(s['ate_error']):.2f} percentage point
  {direction} in error rates. {'This suggests a quantity-quality trade-off — clerks processed more applications but with slightly higher errors.' if s['ate_error'] > 0 else 'This is a favorable result — the AI tool improved both quantity and quality simultaneously.'}

IMPLICATIONS FOR FIRMS
  The RCT design ensures the estimated ATE is causal. AI tool
  deployment yields meaningful productivity gains with {'a modest quality cost worth monitoring.' if s['ate_error'] > 0 else 'no quality degradation.'} Firms should
  consider targeted training to help clerks verify AI-prefilled
  fields before submission, which could preserve speed gains
  while closing any quality gap.
""")
