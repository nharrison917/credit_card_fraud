"""
XGBoost Focused Experiment
- scale_pos_weight only (no SMOTE)
- early stopping on validation set
- randomized hyperparameter search
"""

import pandas as pd
import numpy as np
import random
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from xgboost import XGBClassifier

# ── Data & splits (identical seeds to fraud_analysis.py) ──────────────────────
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_sc = X_train.copy()
X_val_sc   = X_val.copy()
X_test_sc  = X_test.copy()

X_train_sc[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_val_sc[['Amount', 'Time']]   = scaler.transform(X_val[['Amount', 'Time']])
X_test_sc[['Amount', 'Time']]  = scaler.transform(X_test[['Amount', 'Time']])

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
spw = neg_count / pos_count

print(f"scale_pos_weight = {neg_count:,} / {pos_count} = {spw:.1f}")

# Previous best from fraud_analysis.py (RF — Modest SMOTE, validation set)
PREV_BEST = dict(name="RF — Modest SMOTE", precision=0.9048, recall=0.7703, f1=0.8321, auc=0.9593)

# ── Helper ─────────────────────────────────────────────────────────────────────
def score(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return dict(
        precision = precision_score(y, y_pred),
        recall    = recall_score(y, y_pred),
        f1        = f1_score(y, y_pred),
        auc       = roc_auc_score(y, y_prob),
    )

def print_row(label, metrics):
    print(f"  {label:<40} {metrics['precision']:>9.4f} {metrics['recall']:>9.4f}"
          f" {metrics['f1']:>9.4f} {metrics['auc']:>9.4f}")

# ── STEP 1: XGBoost with specified settings ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — XGBoost: scale_pos_weight + early stopping")
print("=" * 60)

xgb1 = XGBClassifier(
    n_estimators      = 500,    # max; early stopping will cut this short
    learning_rate     = 0.1,
    max_depth         = 5,
    colsample_bytree  = 0.8,
    subsample         = 0.8,
    scale_pos_weight  = spw,
    eval_metric       = 'aucpr',
    early_stopping_rounds = 20,
    random_state      = 42,
    n_jobs            = -1,
)
xgb1.fit(
    X_train_sc, y_train,
    eval_set=[(X_val_sc, y_val)],
    verbose=False,
)

m1 = score(xgb1, X_val_sc, y_val)
print(f"\n  Trees used (best iteration): {xgb1.best_iteration} of 500 max")
print(f"  Precision: {m1['precision']:.4f}  Recall: {m1['recall']:.4f}"
      f"  F1: {m1['f1']:.4f}  ROC-AUC: {m1['auc']:.4f}")

print(f"\n  Head-to-head vs previous best (validation set):")
print(f"  {'Model':<40} {'Precision':>9} {'Recall':>9} {'F1':>9} {'ROC-AUC':>9}")
print("  " + "-" * 76)
print_row(PREV_BEST['name'], PREV_BEST)
print_row("XGB — scale_pos_weight + early stop", m1)

# ── STEP 2: Randomized hyperparameter search ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — XGBoost Randomized Search (n_iter=20)")
print("=" * 60)

param_dist = {
    'n_estimators'   : [100, 200, 300, 500],
    'learning_rate'  : [0.01, 0.05, 0.1, 0.2],
    'max_depth'      : [3, 4, 5, 6],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'subsample'      : [0.7, 0.8, 0.9],
}

random.seed(42)
search_results = []

print(f"\n  {'#':>3}  {'lr':>5} {'depth':>5} {'n_est':>6} {'col':>5} {'sub':>5}"
      f"  {'stopped@':>9}  {'F1':>7}  {'AUC':>7}")
print("  " + "-" * 68)

for i in range(20):
    p = {k: random.choice(v) for k, v in param_dist.items()}
    model = XGBClassifier(
        **p,
        scale_pos_weight      = spw,
        eval_metric           = 'aucpr',
        early_stopping_rounds = 20,
        random_state          = 42,
        n_jobs                = -1,
    )
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_val_sc, y_val)],
        verbose=False,
    )
    m = score(model, X_val_sc, y_val)
    m['params'] = p
    m['best_iter'] = model.best_iteration
    m['model'] = model
    search_results.append(m)

    print(f"  {i+1:>3}  {p['learning_rate']:>5}  {p['max_depth']:>4}"
          f"  {p['n_estimators']:>6}  {p['colsample_bytree']:>5}  {p['subsample']:>5}"
          f"  {model.best_iteration:>8}  {m['f1']:>7.4f}  {m['auc']:>7.4f}")

best = max(search_results, key=lambda x: x['f1'])

print(f"\n  Best combination (F1={best['f1']:.4f}, AUC={best['auc']:.4f}):")
for k, v in best['params'].items():
    print(f"    {k:<20}: {v}")
print(f"    {'early stop at':<20}: tree {best['best_iter']}")

# ── Final comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COMPARISON — VALIDATION SET")
print("=" * 60)
print(f"\n  {'Model':<40} {'Precision':>9} {'Recall':>9} {'F1':>9} {'ROC-AUC':>9}")
print("  " + "-" * 76)
print_row(PREV_BEST['name'],                    PREV_BEST)
print_row("XGB — scale_pos_weight + early stop", m1)
print_row("XGB — randomized search best",        best)

delta_f1 = best['f1'] - PREV_BEST['f1']
delta_auc = best['auc'] - PREV_BEST['auc']
sign_f1  = "+" if delta_f1  >= 0 else ""
sign_auc = "+" if delta_auc >= 0 else ""
print(f"\n  XGB search best vs RF prev best:  "
      f"F1 {sign_f1}{delta_f1:.4f}   AUC {sign_auc}{delta_auc:.4f}")

# ── Save best XGBoost model ────────────────────────────────────────────────────
joblib.dump(best['model'], 'models/xgb_best.pkl')
joblib.dump(scaler,        'models/scaler.pkl')

print(f"\nModels saved to models/")
print(f"  xgb_best.pkl — randomized search winner  (F1={best['f1']:.4f})")
print(f"  scaler.pkl   — StandardScaler fit on training Amount & Time")
