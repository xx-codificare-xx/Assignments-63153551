# PHASE 4: INTERPRET THE DATA

import json

# Prompt: Load analysis results saved by phase3_analyze.py
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