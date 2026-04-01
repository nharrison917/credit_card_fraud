# Changelog

## [0.2.0] - 2026-04-01

### Added

**Streamlit dashboard (`app.py`)**
- Single-page interactive dashboard targeting a non-technical audience
- Sidebar holds all controls (model selector, classification threshold slider, FP cost slider) so they remain visible regardless of scroll position
- Section 2 — Bottom Line: three `st.metric` cards summarising the headline result (RF cost-optimal vs LR default vs savings)
- Section 3 — Threshold Explorer: Plotly cost-curve chart per model with vertical markers at cost-optimal and selected thresholds; four live metrics update on slider move
- Section 4 — Model Comparison Table: all six model x threshold combinations; recommended row highlighted with a mode-safe rgba tint
- Section 5 — Sensitivity Analysis: pre-computed minimum-cost curves for each model across FP base costs $1-$100; vertical marker tracks the sidebar slider
- Section 6 — Operational Reality: transaction-level interpretation scaled to 1,000,000 transactions; updates live with model and threshold selection
- Section 7 — Limitations: always expanded for transparency

**`phase2_cost_analysis/cost_fraud_analysis.py`**
- Added dashboard data export block (runs after test set evaluation, before visualisations)
- Exports `models/dashboard_data.json` containing:
  - Full validation sweep per model (threshold, total/FP/FN cost, n_fp, n_fn, recall, precision, F1 at each of 99 thresholds)
  - Test set results at default (0.50) and cost-optimal thresholds per model
  - Metadata: base FP cost, friction rate, validation and test set sizes, total fraud amount

**`requirements.txt`**
- Created with all ML dependencies pinned to current environment versions
- `streamlit` added unpinned (install latest)

### Technical notes

- Sensitivity slider recomputes costs algebraically from stored sweep data rather than re-running models. The friction component (0.01 x sum of FP transaction amounts) is recovered as `fp_cost - base_fp_cost * n_fp`, making the total cost for any new base cost exact with no approximation.
- Sensitivity curves are precomputed at load time via `@st.cache_data` to avoid recomputation on every slider move.
- Row highlight uses `rgba(0, 204, 150, 0.18)` rather than a solid colour for dark/light mode compatibility.
- Savings metric uses `delta_color="inverse"` so a negative percentage renders green.
- `use_container_width` replaced with `width='stretch'` per Streamlit deprecation notice (removal after 2025-12-31).

---

## [0.1.0] - 2026-03-XX — Initial commit

### Added

**Phase 1 — Baseline detection (`phase1_baseline/`)**
- EDA: amount distribution, time-of-day fraud patterns (2-5am spike, card-probing signal under $1)
- Three classifiers: Logistic Regression, Random Forest, XGBoost (40-combination random search)
- SMOTE strategy comparison (50/50, class weights only, 1:10); 1:10 selected on validation F1
- Data leakage prevention: train/val/test split before any preprocessing; scaler and SMOTE fit on training data only
- Best result: Random Forest F1 0.845, ROC-AUC 0.965 on test set

**Phase 2 — Cost-sensitive decision framework (`phase2_cost_analysis/`)**
- Threshold sweep (0.01-0.99) on validation set minimising total business cost
- Cost model: FP = $10 + 1% of Amount (investigation + friction); FN = full Amount (worst-case)
- Sensitivity analysis across four FP cost scenarios ($5 / $10 / $25 / $50)
- Key finding: threshold choice dominates model choice; LR at default costs 3.2x RF cost-optimal
- Stakeholder HTML report and interactive Plotly chart output

**Models (`models/`)**
- `model_metadata.json`: cost-optimal and F1-optimal thresholds, validation costs, hyperparameters per model
- Saved `.pkl` files for all six trained models (gitignored — regenerate by running the scripts)
