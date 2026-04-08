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

# Prompt: Load the cleaned dataset produced by phase2_clean.py
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

# Prompt: Standardize covariates before fitting logistic regression
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

# Prompt: Plot Love Plot showing absolute SMD before vs after matching
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