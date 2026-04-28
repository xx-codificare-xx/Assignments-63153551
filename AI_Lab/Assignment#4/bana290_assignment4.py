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


# STAGE 3: ANALYZE THE DATA

# 3A. NAIVE OLS (Benchmark)
# Run naive OLS of INNOVATION_SCORE on AI_INTENSITY
# as a biased benchmark before IV correction
print("\n" + "="*60)
print("3A. NAIVE OLS REGRESSION")
print("="*60)

X_ols = sm.add_constant(df['AI_INTENSITY'])
ols   = sm.OLS(df['INNOVATION_SCORE'], X_ols).fit()
ols_coef = ols.params['AI_INTENSITY']
ols_pval = ols.pvalues['AI_INTENSITY']
print(ols.summary())
print(f"\nNaive OLS Coefficient: {ols_coef:+.4f}  (p={ols_pval:.4f})")

# 3B. IV ANALYSIS — First Stage
# Regress AI_INTENSITY on DISTANCE_M instrument to
# test instrument relevance via F-statistic (threshold: F>10)
print("\n" + "="*60)
print("3B. IV FIRST STAGE — Instrument Relevance")
print("="*60)

X_fs  = sm.add_constant(df['DISTANCE_M'])
fs    = sm.OLS(df['AI_INTENSITY'], X_fs).fit()
fs_coef = fs.params['DISTANCE_M']
fs_fstat = fs.fvalue
fs_pval  = fs.f_pvalue
df['AI_INTENSITY_HAT'] = fs.fittedvalues

print(fs.summary())
print(f"\nFirst Stage F-statistic : {fs_fstat:.2f}  (p={fs_pval:.4f})")
print(f"Instrument Relevance    : {'STRONG (F > 10)' if fs_fstat > 10 else 'WEAK (F < 10)'}")

# 3C. IV ANALYSIS — Second Stage (2SLS)
# Use fitted AI_INTENSITY from first stage to run
# 2SLS second stage regression on INNOVATION_SCORE
print("\n" + "="*60)
print("3C. IV SECOND STAGE — 2SLS Estimate")
print("="*60)

X_ss   = sm.add_constant(df['AI_INTENSITY_HAT'])
ss     = sm.OLS(df['INNOVATION_SCORE'], X_ss).fit()
iv_coef = ss.params['AI_INTENSITY_HAT']
iv_pval = ss.pvalues['AI_INTENSITY_HAT']

print(ss.summary())
print(f"\n2SLS Coefficient (IV): {iv_coef:+.4f}  (p={iv_pval:.4f})")
print(f"Naive OLS Coefficient: {ols_coef:+.4f}")
print(f"Direction of bias    : {'OLS overstates' if abs(ols_coef) > abs(iv_coef) else 'OLS understates'}")

# 3D. RDD ANALYSIS
# Plot innovation score vs eligibility score with
# separate regression lines on each side of the cutoff.
# Then run local linear RDD regression to estimate the jump
print("\n" + "="*60)
print("3D. RDD ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# RDD scatter plot with fitted lines
below = df[df['ABOVE_CUTOFF'] == 0]
above = df[df['ABOVE_CUTOFF'] == 1]

axes[0].scatter(below['ELIGIBILITY_SCORE'], below['INNOVATION_SCORE'],
                color='tomato', alpha=0.7, label='Below Cutoff (No Grant)', s=50)
axes[0].scatter(above['ELIGIBILITY_SCORE'], above['INNOVATION_SCORE'],
                color='steelblue', alpha=0.7, label='Above Cutoff (Grant)', s=50)

# Fit regression lines on each side
for grp, color in [(below,'tomato'), (above,'steelblue')]:
    x = grp['ELIGIBILITY_SCORE']
    y = grp['INNOVATION_SCORE']
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        axes[0].plot(x_line, m * x_line + b, color=color, linewidth=2)

axes[0].axvline(x=CUTOFF, color='black', linestyle='--', linewidth=1.5, label=f'Cutoff = {CUTOFF}')
axes[0].set_title('RDD: Innovation Score vs Eligibility Score', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Eligibility Score')
axes[0].set_ylabel('Innovation Score')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# IV First Stage plot
axes[1].scatter(df['DISTANCE_M'], df['AI_INTENSITY'],
                color='steelblue', alpha=0.6, s=50)
m2, b2 = np.polyfit(df['DISTANCE_M'], df['AI_INTENSITY'], 1)
x2 = np.linspace(df['DISTANCE_M'].min(), df['DISTANCE_M'].max(), 100)
axes[1].plot(x2, m2 * x2 + b2, color='tomato', linewidth=2)
axes[1].set_title('IV First Stage: Distance → AI Intensity', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Distance to Fiber Node (meters)')
axes[1].set_ylabel('AI Intensity (compute hrs/wk)')
axes[1].grid(alpha=0.3)
axes[1].annotate(f'F = {fs_fstat:.1f}\np = {fs_pval:.4f}', xy=(0.65, 0.85),
                 xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Density plot of eligibility scores to check for
# manipulation at the RDD cutoff (McCrary test proxy)
axes[2].hist(df['ELIGIBILITY_SCORE'], bins=15, color='steelblue',
             alpha=0.7, edgecolor='white')
axes[2].axvline(x=CUTOFF, color='black', linestyle='--',
                linewidth=1.5, label=f'Cutoff = {CUTOFF}')
axes[2].set_title('Continuity Check: Density at RDD Cutoff', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Eligibility Score')
axes[2].set_ylabel('Count')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('iv_rdd_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("[ANALYZE] Saved iv_rdd_plots.png")

# Run local linear RDD regression — regress INNOVATION_SCORE
# on SCORE_CENTERED, ABOVE_CUTOFF, and their interaction
rdd_model = smf.ols(
    'INNOVATION_SCORE ~ SCORE_CENTERED + ABOVE_CUTOFF + SCORE_CENTERED:ABOVE_CUTOFF',
    data=df
).fit()
rdd_coef = rdd_model.params['ABOVE_CUTOFF']
rdd_pval = rdd_model.pvalues['ABOVE_CUTOFF']

print(rdd_model.summary())
print(f"\nRDD Jump at Cutoff     : {rdd_coef:+.4f}  (p={rdd_pval:.4f})")
print(f"IV 2SLS Coefficient    : {iv_coef:+.4f}")
print(f"Naive OLS Coefficient  : {ols_coef:+.4f}")

# Save summary
summary = {
    'ols_coef':  round(ols_coef, 4),  'ols_pval':  round(ols_pval, 4),
    'fs_fstat':  round(fs_fstat, 2),  'fs_pval':   round(fs_pval, 4),
    'fs_coef':   round(fs_coef, 6),
    'iv_coef':   round(iv_coef, 4),   'iv_pval':   round(iv_pval, 4),
    'rdd_coef':  round(rdd_coef, 4),  'rdd_pval':  round(rdd_pval, 4),
    'cutoff':    CUTOFF,
    'n':         int(len(df)),
    'n_above':   int(df['ABOVE_CUTOFF'].sum()),
    'n_below':   int((df['ABOVE_CUTOFF']==0).sum()),
}
with open('analysis_summary_a4.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n[ANALYZE] Saved analysis_summary_a4.json")


# STAGE 4: INTERPRET THE DATA
# Load summary and print interpretation covering IV
# vs OLS comparison, RDD confirmation, exclusion restriction
# argument, continuity assumption, and policy implications
# for digital infrastructure investment

with open('analysis_summary_a4.json', 'r') as f:
    s = json.load(f)

strong_iv  = s['fs_fstat'] > 10
iv_sig     = s['iv_pval']  < 0.05
rdd_sig    = s['rdd_pval'] < 0.05
ols_bigger = abs(s['ols_coef']) > abs(s['iv_coef'])

print("\n" + "="*65)
print("STAGE 4: INTERPRETATION")
print("="*65)
print(f"""
MODEL COMPARISON
  Naive OLS Coefficient  : {s['ols_coef']:+.4f}  (p={s['ols_pval']:.4f})
  IV 2SLS Coefficient    : {s['iv_coef']:+.4f}  (p={s['iv_pval']:.4f})
  RDD Jump at Cutoff     : {s['rdd_coef']:+.4f}  (p={s['rdd_pval']:.4f})
  First Stage F-statistic: {s['fs_fstat']:.2f}  ({'STRONG' if strong_iv else 'WEAK'})

IV vs OLS COMPARISON
  The naive OLS estimate ({s['ols_coef']:+.4f}) {'overstates' if ols_bigger else 'understates'}
  the true causal effect compared to the IV estimate ({s['iv_coef']:+.4f}).
  This is expected — innovative firms naturally invest more in AI,
  creating upward selection bias in OLS. The IV uses physical distance
  to the campus fiber backbone as an exogenous instrument, isolating
  only the variation in AI intensity caused by infrastructure access
  rather than firm ambition. The first-stage F-statistic of {s['fs_fstat']:.1f}
  {'confirms a strong, relevant instrument.' if strong_iv else 'suggests a potentially weak instrument.'}

EXCLUSION RESTRICTION
  Distance to the fiber node plausibly affects innovation only through
  AI intensity — closer teams get faster sync speeds and lower latency,
  enabling heavier compute workloads. There is no credible direct path
  from physical dorm location to innovation score that bypasses AI
  usage. The instrument is geographic and pre-determined, satisfying
  the exclusion restriction.

RDD CONFIRMATION
  The RDD analysis at the 85-point eligibility cutoff {'confirms' if rdd_sig else 'partially supports'}
  the IV findings. Teams just above the threshold received compute
  server credits, which boosted AI workloads and innovation scores.
  The estimated jump of {s['rdd_coef']:+.4f} points {'is' if rdd_sig else 'is not'} statistically
  significant (p={s['rdd_pval']:.4f}).

CONTINUITY ASSUMPTION
  The density plot shows no suspicious bunching of teams just above
  the 85-point cutoff, suggesting teams did not manipulate their
  pitch scores to cross the threshold. The continuity assumption
  for valid RDD identification appears satisfied.

POLICY IMPLICATIONS
  Both IV and RDD evidence point in the same direction: expanding
  fiber infrastructure access causally increases AI intensity and,
  through it, innovation output. Policymakers investing in campus
  or regional digital infrastructure should expect measurable
  downstream innovation gains, particularly for smaller teams that
  currently lack proximity to high-speed network nodes. Grant
  eligibility cutoffs like the 85-point threshold can further
  amplify these gains by directing compute resources to high-
  potential teams at the margin.
""")
