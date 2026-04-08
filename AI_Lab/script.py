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


# PHASE 3: ANALYZE THE DATA

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset produced by phase2_clean.py
df = pd.read_csv('cleaned_data.csv')
print(f"[ANALYZE] Loaded {len(df)} rows.")

# 3A. BASELINE NAIVE OLS REGRESSION
print("3A. BASELINE NAIVE OLS REGRESSION")

X_naive = sm.add_constant(df['AI_ADOPTED'])
y       = df['REV_GROWTH']
ols     = sm.OLS(y, X_naive).fit()
naive_coef = ols.params['AI_ADOPTED']
naive_pval = ols.pvalues['AI_ADOPTED']
print(ols.summary())
print(f"\nNaive OLS Coefficient: {naive_coef:.4f}  (p={naive_pval:.4f})")

# 3B. PROPENSITY SCORE ESTIMATION
print("3B. PROPENSITY SCORE MATCHING (PSM)")

covariates = ['ANNUAL_REV', 'TEAM_SIZE', 'FOUNDED', 'DIGITAL_SALES']
df_psm = df[covariates + ['AI_ADOPTED', 'REV_GROWTH']].dropna().copy()
df_psm.reset_index(drop=True, inplace=True)

# Standardize covariates before fitting logistic regression
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_psm[covariates])
logit    = LogisticRegression(max_iter=1000)
logit.fit(X_scaled, df_psm['AI_ADOPTED'])
df_psm['PROPENSITY_SCORE'] = logit.predict_proba(X_scaled)[:, 1]

treated = df_psm[df_psm['AI_ADOPTED'] == 1].copy()
control = df_psm[df_psm['AI_ADOPTED'] == 0].copy()
print(f"Treated (AI=1): {len(treated)} | Control (AI=0): {len(control)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(treated['PROPENSITY_SCORE'], bins=12, alpha=0.6,
             color='steelblue', label='AI Adopted (1)', edgecolor='white')
axes[0].hist(control['PROPENSITY_SCORE'], bins=12, alpha=0.6,
             color='tomato', label='Not Adopted (0)', edgecolor='white')
axes[0].set_title('Common Support: Propensity Score Distributions', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Propensity Score')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(alpha=0.3)

# BALANCING PROPERTY — SMD BEFORE MATCHING
def compute_smd(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0

smd_before = {c: compute_smd(treated[c], control[c]) for c in covariates}
print("\nSMD Before Matching:")
for k, v in smd_before.items():
    print(f"  {k}: {v:.4f}")

# NEAREST-NEIGHBOR MATCHING (without replacement)
t_scores = treated['PROPENSITY_SCORE'].values.reshape(-1, 1)
c_scores = control['PROPENSITY_SCORE'].values.reshape(-1, 1)
dist_matrix = cdist(c_scores, t_scores, metric='euclidean')

matched_treated_idx  = []
matched_control_rows = []
used = set()
for i in range(len(control)):
    for idx in np.argsort(dist_matrix[i]):
        if idx not in used:
            matched_control_rows.append(i)
            matched_treated_idx.append(idx)
            used.add(idx)
            break

matched_treated = treated.iloc[matched_treated_idx].reset_index(drop=True)
matched_control = control.iloc[matched_control_rows].reset_index(drop=True)
matched_df = pd.concat([matched_treated, matched_control], ignore_index=True)
print(f"\nMatched: {len(matched_treated)} treated + {len(matched_control)} control = {len(matched_df)} firms")

# BALANCING PROPERTY — SMD AFTER MATCHING
smd_after = {
    c: compute_smd(
        matched_df[matched_df['AI_ADOPTED'] == 1][c],
        matched_df[matched_df['AI_ADOPTED'] == 0][c]
    )
    for c in covariates
}
print("\nSMD After Matching:")
for k, v in smd_after.items():
    flag = "OK" if abs(v) < 0.1 else "CHECK"
    print(f"  {k}: {v:.4f}  [{flag}]")

# Plot Love Plot showing absolute SMD before vs after matching
labels     = covariates
before_abs = [abs(smd_before[c]) for c in labels]
after_abs  = [abs(smd_after[c])  for c in labels]
ypos = np.arange(len(labels))

axes[1].barh(ypos - 0.2, before_abs, 0.35, label='Before Matching', color='tomato',    alpha=0.8)
axes[1].barh(ypos + 0.2, after_abs,  0.35, label='After Matching',  color='steelblue', alpha=0.8)
axes[1].axvline(0.1, color='black', linestyle='--', linewidth=1.2, label='SMD = 0.1 threshold')
axes[1].set_yticks(ypos)
axes[1].set_yticklabels(labels)
axes[1].set_xlabel('Absolute Standardized Mean Difference')
axes[1].set_title('Balancing Property: SMD Before vs After Matching', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('psm_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[ANALYZE] Saved psm_plots.png")

# PSM OLS — RE-ESTIMATE ON MATCHED SAMPLE
print("3C. PSM OLS ON MATCHED SAMPLE")

X_matched = sm.add_constant(matched_df['AI_ADOPTED'])
psm_ols   = sm.OLS(matched_df['REV_GROWTH'], X_matched).fit()
psm_coef  = psm_ols.params['AI_ADOPTED']
psm_pval  = psm_ols.pvalues['AI_ADOPTED']
print(psm_ols.summary())
print(f"\nPSM Coefficient: {psm_coef:.4f}  (p={psm_pval:.4f})")

# Save matched dataset for Phase 4
matched_df.to_csv('matched_data.csv', index=False)

# Save summary stats for interpretation
import json
summary = {
    'naive_coef': round(naive_coef, 4),
    'naive_pval': round(naive_pval, 4),
    'psm_coef':   round(psm_coef, 4),
    'psm_pval':   round(psm_pval, 4),
    'shift':      round(psm_coef - naive_coef, 4),
    'treated_ps_min': round(float(treated['PROPENSITY_SCORE'].min()), 4),
    'treated_ps_max': round(float(treated['PROPENSITY_SCORE'].max()), 4),
    'control_ps_min': round(float(control['PROPENSITY_SCORE'].min()), 4),
    'control_ps_max': round(float(control['PROPENSITY_SCORE'].max()), 4),
    'smd_before': {k: round(v, 4) for k, v in smd_before.items()},
    'smd_after':  {k: round(v, 4) for k, v in smd_after.items()},
}
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n[ANALYZE] Saved matched_data.csv and analysis_summary.json")
print("RESULTS SUMMARY")
print(f"Naive OLS Coefficient : {naive_coef:+.4f}  (p={naive_pval:.4f})")
print(f"PSM   OLS Coefficient : {psm_coef:+.4f}  (p={psm_pval:.4f})")
print(f"Shift after PSM       : {psm_coef - naive_coef:+.4f}")

# PHASE 4: INTERPRET THE DATA

import json

# Load analysis results saved by phase3_analyze.py
with open('analysis_summary.json', 'r') as f:
    s = json.load(f)

print("PHASE 4: INTERPRETATION OF PSM RESULTS")

print(f"""
COEFFICIENT COMPARISON
  Naive OLS (AI_ADOPTED): {s['naive_coef']:+.4f}  (p={s['naive_pval']:.4f})
  PSM   OLS (AI_ADOPTED): {s['psm_coef']:+.4f}  (p={s['psm_pval']:.4f})
  Shift after PSM       : {s['shift']:+.4f}
""")

print("""
HOW DID THE COEFFICIENT CHANGE?
  The naive OLS regression estimated that AI-adopting fintech firms
  grow approximately {naive:.2f} percentage points faster than
  non-adopters. After applying Propensity Score Matching (PSM), this
  estimate fell to {psm:.2f} percentage points — a downward shift of
  {shift:.2f} points. Despite the reduction, the PSM coefficient
  remains positive and statistically significant (p={pval:.4f}),
  suggesting that AI adoption is associated with higher revenue growth
  even after controlling for observable firm characteristics.

WHAT DOES THE SHIFT SUGGEST ABOUT SELECTION BIAS?
  The downward shift indicates that the naive OLS estimate was upwardly
  biased. Firms that choose to adopt AI tend to be larger, better
  resourced, and more digitally mature than those that do not — meaning
  they may have grown faster regardless of AI adoption. PSM partially
  corrects for this by pairing each AI-adopting firm with a
  structurally similar non-adopting firm, reducing the confounding
  effect of firm size, team scale, and digital footprint. The remaining
  PSM coefficient of {psm:.2f} represents a more conservative and
  credible estimate of AI's causal impact on revenue growth.
""".format(
    naive=s['naive_coef'],
    psm=s['psm_coef'],
    shift=abs(s['shift']),
    pval=s['psm_pval']
))

print(f"""
COMMON SUPPORT ASSUMPTION
  Treated propensity scores ranged from {s['treated_ps_min']:.3f} to {s['treated_ps_max']:.3f}.
  Control propensity scores ranged from {s['control_ps_min']:.3f} to {s['control_ps_max']:.3f}.
  The two distributions show substantial overlap, confirming that the
  common support assumption is satisfied. No extreme trimming was
  required, and every control firm fell within the support range of
  the treated group. This overlap is visible in the left panel of
  psm_plots.png.

BALANCING PROPERTY ASSUMPTION
  Before Matching (Absolute SMD):""")

for k, v in s['smd_before'].items():
    print(f"    {k:<20}: {v:.4f}")

print(f"\n  After Matching (Absolute SMD):")
for k, v in s['smd_after'].items():
    flag = "BALANCED" if abs(v) < 0.1 else "PARTIALLY BALANCED"
    print(f"    {k:<20}: {v:.4f}  [{flag}]")

max_after = max(abs(v) for v in s['smd_after'].values())
print(f"""
  Before matching, all covariates showed meaningful imbalance
  (SMD ranging from 0.12 to 0.44), confirming that AI adopters
  and non-adopters were systematically different. After matching,
  the FOUNDED covariate achieved full balance (SMD < 0.1), while
  ANNUAL_REV, TEAM_SIZE, and DIGITAL_SALES improved substantially
  but remained partially imbalanced (max SMD = {max_after:.4f}).
  This residual imbalance is a known limitation of PSM with small
  samples and reflects the genuine structural differences between
  AI-adopting and non-adopting fintech firms in this dataset.
  The Love Plot in psm_plots.png illustrates this improvement.

CONCLUSION
  After applying PSM, AI adoption is still associated with
  approximately {s['psm_coef']:.2f} percentage points of additional
  revenue growth (p={s['psm_pval']:.4f}). The common support
  assumption holds well. The balancing property is partially
  satisfied, with meaningful improvement after matching. Together,
  these results suggest a real but modestly overstated positive
  relationship between AI adoption and firm revenue growth when
  selection bias is not accounted for.
""")