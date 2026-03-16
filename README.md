# Credit Card Fraud Detection

## Two-Phase Analysis: ML Detection + Cost-Sensitive Decision Framework

---

## Overview

This project applies machine learning to the Kaggle Credit Card Fraud dataset
(284,807 transactions, 492 confirmed fraud cases, 0.17% fraud rate) across two
analytical phases.

Phase 1 establishes a baseline detection pipeline — framing the class imbalance
problem, selecting an appropriate evaluation metric, and comparing three classifiers.
Phase 2 extends the analysis into business decision territory: given a model, what
classification threshold minimises total business cost rather than optimising a
statistical metric?

The central finding is that **threshold selection matters more than model selection**.
A poorly-chosen threshold can triple the business cost of an otherwise competitive model.

> This project was developed using agentic AI (Claude Code) as the analytical
> environment. Methodology decisions — SMOTE ratio selection, cost function design,
> sensitivity analysis scope, and limitation framing — were directed by the author.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions over two days (European cardholders, September 2013)
- Features V1–V28 are PCA-transformed and anonymized — not directly interpretable
- `Amount` and `Time` are the only raw features
- Class imbalance (0.17%) is the central modeling challenge

---

## Phase 1 — Baseline Detection

### Methodology

**Why F1 over accuracy:**
A model predicting "not fraud" on every transaction achieves 99.83% accuracy
and catches zero fraud. F1 is the harmonic mean of precision and recall —
it penalises models that sacrifice one for the other, making it the appropriate
single metric when both false positives (wasted investigations) and false negatives
(missed fraud) carry business cost.

**Imbalance handling:**
Three strategies were evaluated: full 50/50 SMOTE, class weights only, and modest
SMOTE at a 1:10 minority:majority ratio. The 1:10 ratio outperformed the others on
validation F1. Full 50/50 SMOTE inflates the synthetic minority class to the point
of overfitting; class weights alone produced weaker recall on this dataset.

**Data leakage prevention:**
Train/validation/test split was performed before any preprocessing. `StandardScaler`
was fit on training data only and applied to validation and test. SMOTE was applied
to training data only. The test set was held out and used exactly once.

### Results — Test Set

| Model | Recall | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.838 | 0.508 | 0.957 |
| **Random Forest** | **0.811** | **0.845** | **0.965** |
| XGBoost (tuned) | 0.824 | 0.828 | 0.965 |

Random Forest wins on F1. 40 random search combinations across XGBoost
hyperparameters did not close the gap — V1–V28 are already clean PCA components,
which reduces XGBoost's typical advantage on messy raw features. RF handles
tidy high-dimensional data naturally with default settings.

LR achieves the highest recall but lowest precision — it catches more fraud
but flags too many legitimate transactions, collapsing F1 to 0.51.

---

## Phase 2 — Cost-Sensitive Decision Framework

### Cost Model

> **All figures are illustrative.** The dataset currency is unknown and amounts
> are anonymized. Cost calculations should be interpreted as relative comparisons,
> not absolute dollar figures.

| Error Type | Cost Formula | Rationale |
|---|---|---|
| False Positive | $10 + (0.01 × Amount) | $10 investigation + friction penalty for larger blocked transactions |
| False Negative | Full Amount | Worst-case assumption — no chargeback recovery modeled |

The $10 base investigation cost is drawn from an industry estimate of $7–10
per flagged transaction. The 1% friction penalty reflects the reality that
incorrectly blocking a $500 transaction creates substantially more customer
friction than blocking a $5 one.

### Threshold Optimisation

For each model, the classification threshold was swept from 0.01 to 0.99 on
the validation set. At each threshold, total business cost was calculated using
the formula above. The cost-optimal threshold was then applied to the test set.

| Model | Threshold | Total Cost | FP Cost | FN Cost | Recall | F1 |
|---|---|---|---|---|---|---|
| LR — default (0.5) | 0.50 | $14,103 | $10,779 | $3,325 | 0.878 | 0.116 |
| LR — cost-optimal | 0.99 | $4,562 | $326 | $4,236 | 0.838 | 0.747 |
| RF — default (0.5) | 0.50 | $4,899 | $91 | $4,808 | 0.797 | 0.831 |
| **RF — cost-optimal** | **0.30** | **$4,389** | **$153** | **$4,236** | **0.851** | **0.834** |
| XGB — default (0.5) | 0.50 | $6,164 | $1,677 | $4,487 | 0.824 | 0.421 |
| XGB — cost-optimal | 0.75 | $5,396 | $908 | $4,488 | 0.811 | 0.550 |

**The key finding:** LR at its default threshold costs $14,103 — 3.2× the
RF cost-optimal result — almost entirely due to 980 false positives at $10+
each. Shifting LR's threshold to 0.99 cuts its cost to $4,562, nearly matching
RF. Threshold choice dominates model choice.

The RF cost-optimal threshold (0.30) is lower than the default 0.50 because
fraud losses dominate: missing a fraudulent transaction costs the full amount,
while a false alarm costs only ~$10–12. The model should cast a wider net.

### Operational Interpretation

At a cost-optimal threshold of **0.30**, for every 10,000 transactions
Random Forest flags approximately **18** as fraud, of which approximately
**15** are genuine fraud and **3** are false alarms. The estimated
investigation and friction cost of false alarms is **~$36**, offset against
approximately **~$992** in fraud prevented, for a net saving of approximately
**~$958** compared to no fraud detection.

### Sensitivity Analysis

The cost-optimal threshold was tested across four investigation cost scenarios
to assess robustness to assumption uncertainty.

| Scenario | FP Base Cost | Optimal Threshold | Total Cost (validation) |
|---|---|---|---|
| Low | $5 | 0.30 | $2,406 |
| Base | $10 | 0.30 | $2,506 |
| High | $25 | 0.50 | $2,697 |
| Very High | $50 | 0.50 | $2,897 |

The threshold is stable. Even as investigation cost quadruples from $5 to $50,
the optimal threshold only shifts from 0.30 to 0.50. FN cost (missed fraud)
dominates the objective throughout, confirming the framework is not sensitive
to reasonable variation in the FP cost assumption.

---

## Limitations

**Transaction independence:** The model treats every transaction independently.
Real fraud often follows sequential patterns — card testing uses small $1–5
transactions to verify a stolen card before executing larger fraud. The PCA
anonymization eliminates card identifiers needed to construct velocity features.
The cost function understates the true cost of certain false negatives: missing
a $2 card test transaction is not a $2 loss but potentially the sum of all
subsequent fraud it enables.

**Anonymized amounts:** Dollar figures are illustrative — the dataset currency
is unknown. All cost calculations should be interpreted as relative comparisons.

**SMOTE generates synthetic samples:** Synthetic minority oversampling interpolates
between real fraud cases. These may not reflect rare or novel fraud patterns.

**No temporal validation:** The data was split randomly rather than by time.
Production models should be trained on past data and evaluated on future data.
Fraud patterns also drift as adversaries adapt; this model would degrade in
deployment without retraining.

**Recovery rate not modeled:** Banks recover some fraud losses via chargebacks.
False negative cost as full Amount lost is a worst-case assumption.

**No feature engineering on raw features:** EDA identified clear signal in the
two unmasked features — fraud concentrates in overnight hours (2–5am, up to 12×
the baseline rate), 37% of fraud transactions are under $1 (card probing), and
round-dollar amounts are overrepresented in fraud relative to the retail pricing
patterns seen in legitimate transactions. None of this was translated into
engineered features. Derived inputs such as `hour_of_day`, `is_probe` (Amount < 1),
`is_round` (Amount % 1 == 0), and `log1p(Amount)` would likely improve model
performance and are fully justified by the data rather than speculative. This was
scoped out to keep the two phases focused, but represents the most accessible
performance improvement available within this dataset.

---

## What a Richer Dataset Would Enable

This dataset's PCA anonymization is the binding constraint. With card-level
relational structure the natural next features would be:

- Velocity features (transactions per hour per card)
- New merchant flags
- Location velocity (two transactions 500 miles apart in one hour)
- Device consistency checks
- Behavioral baseline per customer

The [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection)
preserves enough relational structure for sequential modeling and is the
natural next step.

---

## Project Structure

```
phase1_baseline/
    fraud_detection.py       Baseline pipeline: SMOTE, 3 models, evaluation
    eda.html                 EDA charts: amount distribution, time patterns
    results.html             ROC curves, confusion matrix, model comparison
    time_amount.md           Written analysis of Time and Amount fraud patterns

phase2_cost_analysis/
    cost_fraud_analysis.py   Cost-sensitive pipeline: threshold optimisation,
                             sensitivity analysis, stakeholder report
    cost_results.html        Interactive Plotly charts
    cost_report.html         Standalone stakeholder report
    results_summary.md       Full results tables
    xgb.md                   XGBoost tuning write-up

models/
    model_metadata.json      Optimal thresholds and hyperparameters per model
    *.pkl                    Saved model binaries (not tracked in git —
                             regenerate by running the scripts)

archive/
    fraud_analysis.py        Original strategy comparison (3 SMOTE strategies
                             × 3 models = 9 models evaluated)
    xgb_experiment.py        XGBoost hyperparameter search runs

data/                        Not tracked — see Dataset section above
```

---

## How to Run

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn plotly matplotlib seaborn joblib

# Phase 1 — baseline detection
python phase1_baseline/fraud_detection.py

# Phase 2 — cost-sensitive analysis
python phase2_cost_analysis/cost_fraud_analysis.py
```

Both scripts resolve all paths relative to their own location and can be
run from the project root or from within their directories.

The dataset (`data/creditcard.csv`) is not included in this repository.
Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in the `data/` folder before running.

---

## Technologies

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · Plotly · joblib
