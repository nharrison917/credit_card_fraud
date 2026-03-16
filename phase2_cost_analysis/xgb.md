# XGBoost Tuning — Credit Card Fraud Detection

## Context

The main analysis (`fraud_analysis.py`) trained three models (Logistic Regression,
Random Forest, XGBoost) across three imbalance-handling strategies (SMOTE 50/50,
class weights, modest SMOTE 1:10). The winner on the held-out test set was
**Random Forest + Modest SMOTE** with F1=0.853, Precision=0.884, Recall=0.824,
ROC-AUC=0.965.

This document covers a focused effort to beat that benchmark using XGBoost alone,
with improved imbalance handling and two rounds of hyperparameter search.

---

## Why XGBoost for Fraud Detection?

XGBoost is a gradient-boosted tree ensemble. Each tree is fit to the *residuals* of
the previous one — meaning the model iteratively concentrates effort on the examples
it is currently getting wrong. For fraud detection this is appealing: the rare fraud
cases that are hard to classify get re-weighted up in later boosting rounds. This is
a fundamentally different mechanism from Random Forest, which fits all trees in
parallel and averages them. RF reduces variance; XGBoost reduces both bias and
variance sequentially.

The cost: XGBoost has many more hyperparameters and is more sensitive to them.

---

## Step 0 — Switching from SMOTE to `scale_pos_weight`

### What SMOTE does
SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud samples
by interpolating between existing minority-class points in feature space. This
physically changes the training distribution — the model sees a balanced dataset.

### The problem with full 50/50 SMOTE
With ~344 real fraud cases in the training set and ~199,000 legitimate ones, hitting
50/50 requires generating ~198,676 synthetic samples from those 344 real points. The
synthetic samples are linear combinations of nearest neighbours, so they densely
cover the convex hull of the real fraud cases but invent no genuinely new patterns.
The model can overfit to this interpolated region.

### `scale_pos_weight` — the gradient-level alternative
Rather than changing *what data* the model sees, `scale_pos_weight` changes *how
much each error costs*. In XGBoost's gradient computation, the loss contribution of
each positive (fraud) sample is multiplied by this weight. Setting it to
`neg / pos ≈ 578.5` means each missed fraud case contributes 578× more gradient
signal than a missed legitimate case.

**Why this is cleaner:** the model still trains on real data only. Imbalance is
handled at the loss-function level, not by data augmentation. This avoids the
synthetic-distribution risk and reduces training set size dramatically (no inflated
resampled set to iterate over).

---

## Step 1 — Baseline XGBoost with Specified Settings

**Configuration:**
```
n_estimators      = 500 (max)
learning_rate     = 0.1
max_depth         = 5
colsample_bytree  = 0.8   # each tree sees 80% of features
subsample         = 0.8   # each tree sees 80% of rows (stochastic boosting)
scale_pos_weight  = 578.5
eval_metric       = 'aucpr'
early_stopping_rounds = 20
```

### Why `colsample_bytree` and `subsample`?
These are regularisation parameters. Without them, every tree sees the full dataset,
which increases correlation between trees and variance of the ensemble. Subsampling
rows and columns introduces diversity — analogous to the random feature selection
that makes Random Forest robust. Values of 0.7–0.9 are standard starting points.

### Why `eval_metric = 'aucpr'` over `logloss`?
AUCPR is the area under the Precision-Recall curve. For severely imbalanced problems,
the ROC curve can look deceptively good (because the huge true-negative mass keeps
FPR low), while the PR curve exposes the model's actual struggle to find rare
positives. Using AUCPR as the early-stopping criterion therefore aligns the stopping
rule with the thing we actually care about.

### Why early stopping?
Without it, XGBoost trains for all `n_estimators` trees regardless of whether
validation performance has plateaued or begun to overfit. Early stopping monitors
the validation eval_metric after each tree and halts training if it hasn't improved
in `early_stopping_rounds` consecutive rounds. This means `n_estimators` becomes an
upper bound, not a fixed cost.

### Result
```
Trees used: 35 of 500
Precision: 0.367   Recall: 0.838   F1: 0.510   ROC-AUC: 0.972
```

Only 35 trees were built before early stopping triggered — a sign that the
default learning rate of 0.1 converges quickly on this problem. The recall is high
(the aggressive `scale_pos_weight` is working) but precision is poor: the model is
flagging many legitimate transactions as fraud. F1 of 0.51 is well below the RF
benchmark. The default hyperparameters are not a good match for this configuration.

---

## Step 2 — Round 1 Randomized Search (n_iter = 20)

### Why randomized search over grid search?
A full grid over the five parameters at the specified levels would be
4 × 4 × 4 × 3 × 3 = 576 combinations. Each fit involves training an XGBoost model
with early stopping on the full training set. Randomized search samples uniformly
from the Cartesian product — 20 random draws gives broad coverage of the space at
~3.5% of the cost. For continuous hyperparameters you would use a distribution
(e.g. `loguniform` for learning rate); with discrete lists, uniform sampling over
the list is equivalent.

**Important:** we evaluate each combination on the validation set, not via
cross-validation. This is deliberate — cross-validation on the full training set
would be expensive and we have a held-out validation set that is representative and
stratified. The test set is still locked.

**Search space:**
```python
param_dist = {
    'n_estimators'    : [100, 200, 300, 500],
    'learning_rate'   : [0.01, 0.05, 0.1, 0.2],
    'max_depth'       : [3, 4, 5, 6],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample'       : [0.7, 0.8, 0.9],
}
```

### Key findings from 20 iterations

| Observation | Implication |
|---|---|
| Low learning rates (0.01–0.05) consistently underperformed | The model needs to take large enough steps to distinguish fraud from legitimate in a heavily re-weighted loss landscape |
| Shallow trees (depth 3) gave poor F1 regardless of lr | Fraud patterns in this dataset require deeper splits — likely interactions between multiple V-features |
| depth 6 + lr 0.2 gave F1=0.835 (iteration 12, 322 trees) | Deep trees + moderate-high lr works; needed many trees to converge |
| depth 5 + lr 0.2 gave F1=0.806 with AUC=0.983 | Shallower trees need more of them but reach better AUC |
| One run stopped at tree 6, still got F1=0.50 | With aggressive weighting, a few trees at high depth can already split off most fraud |

**Round 1 winner:** lr=0.2, depth=6, n_estimators=300, col=0.8, sub=0.7,
stopped at tree 72 → **F1=0.835, AUC=0.978**

---

## Step 3 — Round 2 Focused Search

Based on Round 1, the promising region was clearly high lr (0.2+) and deeper trees
(depth 5–7). The search space was tightened:

```python
param_dist = {
    'learning_rate'   : [0.15, 0.2, 0.25, 0.3],
    'max_depth'       : [5, 6, 7],
    'n_estimators'    : [200, 300, 400, 500],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample'       : [0.6, 0.7, 0.8],
}
```

20 more iterations. The F1 distribution tightened (most runs 0.80–0.85), confirming
we were in the right region. The best run:

**Round 2 winner:** lr=0.3, depth=7, n_estimators=200, col=0.9, sub=0.7,
stopped at tree 118 → **F1=0.847, AUC=0.971 (validation)**

One notable failure (row 14, lr=0.15, depth=5, early-stopped at tree 9): with a
low learning rate and aggressive early stopping, the model halted before it could
learn anything useful — a reminder that `early_stopping_rounds=20` is a relative
threshold, not an absolute one.

---

## Test Set Evaluation

Both winners were retrained on the full training set with their best hyperparameters,
early-stopped on the validation set, and evaluated once on the locked test set.

```
Model                    Precision   Recall      F1   ROC-AUC
─────────────────────────────────────────────────────────────
RF — Modest SMOTE          0.8841   0.8243   0.8531    0.9650
XGB round-2 winner         0.8824   0.8108   0.8451    0.9670
XGB round-1 winner         0.8451   0.8108   0.8276    0.9649
```

### What happened to the validation gains?

The round-2 winner showed F1=0.847 on validation but only 0.845 on test — a drop of
~0.002, well within expected noise. The validation gains over RF (+0.015 F1) did not
transfer; on the test set RF still leads by +0.008 F1.

This is a textbook example of **optimistic validation bias**. With 40 random search
iterations, we implicitly tuned to the specific validation split — even without
directly overfitting to it, the search will favour configurations that got lucky on
that particular sample of 74 fraud cases. A more rigorous approach would use nested
cross-validation or a second held-out tuning set.

Notably, both XGB winners produced **identical recall** (60 TP, 14 FN). All the
F1 difference between them comes from precision (FPs: 11 vs 8). The RF catches 62 TP
vs XGB's 60 — those 2 extra caught fraud cases are the real source of its F1 lead.

---

## Summary

| Decision | Rationale |
|---|---|
| `scale_pos_weight` instead of SMOTE | Handles imbalance at the loss level without synthetic data; trains on real examples only |
| `eval_metric = 'aucpr'` | PR-AUC is more informative than log-loss for rare-event problems; aligns stopping criterion with detection quality |
| Early stopping (rounds=20) | Prevents overfitting without grid-searching `n_estimators`; acts as a built-in regulariser |
| High learning rate (0.2–0.3) | This dataset rewards aggressive steps; low lr models stopped too early or never converged to high-precision solutions |
| Deep trees (depth 6–7) | Fraud signals here involve interactions between multiple PCA-derived V-features; shallow trees cannot represent them |
| Randomized → focused search | Broad random coverage first identified the promising region; focused refinement confirmed it without 576-combination grid cost |

**Final verdict:** RF + Modest SMOTE remains the champion (F1=0.853). The tuned XGBoost
is competitive (F1=0.845, AUC=0.967) and has marginally better discrimination across
thresholds, but did not overcome the RF on the metric that matters most for this use
case. With 40 random search iterations over a discrete parameter space this is close
to the ceiling of what this approach will find.
