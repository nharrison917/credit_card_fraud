"""
Phase 1 -- Credit Card Fraud Detection
=======================================
Baseline fraud detection pipeline using three classifiers.

Approach: Modest SMOTE (1:10 minority:majority) on training data only.
This was selected over full 50/50 SMOTE (overfits synthetic minority) and
class weights alone (weaker recall on this dataset) after exploratory comparison.
See archive/fraud_analysis.py for the full strategy comparison.

Models trained:
  - Logistic Regression  (class_weight='balanced')
  - Random Forest        (n_estimators=100, class_weight='balanced')
  - XGBoost              (scale_pos_weight = legit/fraud ratio)

Run from anywhere: python phase1_baseline/fraud_detection.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Resolve all paths relative to this script's location
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.normpath(os.path.join(SCRIPT_DIR, '../data/creditcard.csv'))
MODELS_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '../models'))

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  PHASE 1 -- FRAUD DETECTION BASELINE")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
fraud_pct = df['Class'].mean() * 100
print(f"  {len(df):,} transactions  |  "
      f"{df['Class'].sum():,} fraud ({fraud_pct:.3f}%)  |  "
      f"{(df['Class']==0).sum():,} legitimate")

# ============================================================
# SPLIT  70% train / 15% val / 15% test, stratified
# Identical seeds to phase 2 for direct comparability.
# Test set is locked here and not touched until step 5.
# ============================================================
print("\n[2/6] Splitting data (70/15/15, stratified, random_state=42)...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"  Train: {len(X_train):,}  (fraud: {y_train.sum()})")
print(f"  Val:   {len(X_val):,}  (fraud: {y_val.sum()})")
print(f"  Test:  {len(X_test):,}  (fraud: {y_test.sum()})  <-- locked")

# ============================================================
# SCALE  fit on training only, apply to all sets
# Only Amount and Time are scaled; V1-V28 are already PCA-transformed.
# ============================================================
print("\n[3/6] Scaling Amount and Time (fit on train only)...")
scaler = StandardScaler()
X_train_sc = X_train.copy()
X_val_sc   = X_val.copy()
X_test_sc  = X_test.copy()
X_train_sc[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_val_sc[['Amount', 'Time']]   = scaler.transform(X_val[['Amount', 'Time']])
X_test_sc[['Amount', 'Time']]  = scaler.transform(X_test[['Amount', 'Time']])

# ============================================================
# SMOTE  1:10 ratio on training set only
# sampling_strategy=0.1 -> minority/majority = 0.1 after resampling.
# Validation and test sets are left at their natural imbalance.
# ============================================================
print("\n[4/6] Applying SMOTE (1:10) to training set only...")
print(f"  Before: {(y_train==0).sum():,} legit / {y_train.sum()} fraud")
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"  After:  {(y_train_sm==0).sum():,} legit / {(y_train_sm==1).sum():,} fraud  "
      f"(ratio 1:{(y_train_sm==0).sum()//(y_train_sm==1).sum()})")

# ============================================================
# TRAIN
# ============================================================
print("\n[5/6] Training models...")

spw = (y_train == 0).sum() / (y_train == 1).sum()

print(f"\n  Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1,
                        class_weight='balanced')
lr.fit(X_train_sm, y_train_sm)

print(f"  Random Forest (n_estimators=100)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                             class_weight='balanced')
rf.fit(X_train_sm, y_train_sm)

print(f"  XGBoost (scale_pos_weight={spw:.0f})...")
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                    scale_pos_weight=spw, eval_metric='logloss',
                    random_state=42, n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)

# Save models and scaler
joblib.dump(lr,     os.path.join(MODELS_DIR, 'lr_best.pkl'))
joblib.dump(rf,     os.path.join(MODELS_DIR, 'rf_best.pkl'))
joblib.dump(xgb,    os.path.join(MODELS_DIR, 'xgb_best.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
print(f"\n  Saved to models/: lr_best.pkl  rf_best.pkl  xgb_best.pkl  scaler.pkl")

# ============================================================
# EVALUATE
# ============================================================
print("\n[6/6] Evaluating on validation set, then test set...")

def evaluate(name, model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'name':      name,
        'model':     model,
        'y_prob':    y_prob,
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred),
        'auc':       roc_auc_score(y_true, y_prob),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }

models = [('Logistic Regression', lr), ('Random Forest', rf), ('XGBoost', xgb)]

# Validation
val_results = [evaluate(name, m, X_val_sc, y_val) for name, m in models]
print("\n  VALIDATION SET")
print(f"  {'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("  " + "-" * 58)
for r in val_results:
    print(f"  {r['name']:<22} {r['precision']:>10.4f} {r['recall']:>8.4f} "
          f"{r['f1']:>8.4f} {r['auc']:>9.4f}")

# Pick best model by F1
best_val = max(val_results, key=lambda r: r['f1'])
print(f"\n  Best by F1: {best_val['name']}")

# Test set -- first and only use
print("\n  >>> TEST SET (first and only use) <<<")
test_results = [evaluate(name, m, X_test_sc, y_test) for name, m in models]
print(f"\n  {'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("  " + "-" * 58)
for r in test_results:
    champion = " <--" if r['name'] == best_val['name'] else ""
    print(f"  {r['name']:<22} {r['precision']:>10.4f} {r['recall']:>8.4f} "
          f"{r['f1']:>8.4f} {r['auc']:>9.4f}{champion}")

best_test = max(test_results, key=lambda r: r['f1'])

# Confusion matrix detail for best model
print(f"\n  {best_test['name']} confusion matrix (test set):")
print(f"    True Negatives  (legit correct):  {best_test['tn']:,}")
print(f"    False Positives (legit flagged):   {best_test['fp']:,}")
print(f"    False Negatives (fraud missed):    {best_test['fn']:,}")
print(f"    True Positives  (fraud caught):    {best_test['tp']:,}")

# Feature importance
print(f"\n  Top 10 features ({best_test['name']}):")
if hasattr(best_test['model'], 'feature_importances_'):
    imp = pd.Series(best_test['model'].feature_importances_, index=X_train.columns)
    for feat, val in imp.nlargest(10).items():
        print(f"    {feat:<8}  {val:.4f}")
elif hasattr(best_test['model'], 'coef_'):
    imp = pd.Series(np.abs(best_test['model'].coef_[0]), index=X_train.columns)
    for feat, val in imp.nlargest(10).items():
        print(f"    {feat:<8}  {val:.4f}  (|coef|)")

# ============================================================
# VISUALIZATIONS -> phase1_baseline/results.html
# ============================================================
print("\n  Building results.html...")
figs = []

# ROC curves
fig_roc = go.Figure()
for r in test_results:
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f"{r['name']} (AUC={r['auc']:.4f})",
        mode='lines',
    ))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
    line=dict(dash='dash', color='grey'), showlegend=False))
fig_roc.update_layout(
    title='ROC Curves -- Test Set',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    height=480,
)
figs.append(('ROC Curves', fig_roc))

# Confusion matrix -- proper 2x2 heatmap with counts and percentages
r = best_test
total = r['tn'] + r['fp'] + r['fn'] + r['tp']
# Layout: rows = Actual (Legit, Fraud), cols = Predicted (Legit, Fraud)
z      = [[r['tn'],  r['fp']],
          [r['fn'],  r['tp']]]
z_pct  = [[r['tn']/total*100, r['fp']/total*100],
          [r['fn']/total*100, r['tp']/total*100]]
labels = [['True Negative', 'False Positive'],
          ['False Negative', 'True Positive']]
annot  = [[f"<b>{z[i][j]:,}</b><br>{z_pct[i][j]:.3f}%<br><i>{labels[i][j]}</i>"
           for j in range(2)] for i in range(2)]
# Colour scale: high TN/TP green, high FP/FN red -- invert for off-diagonal
cm_z_display = [[z[i][j] if (i==j) else -z[i][j] for j in range(2)] for i in range(2)]
fig_cm = go.Figure(go.Heatmap(
    z=cm_z_display,
    x=['Predicted: Legitimate', 'Predicted: Fraud'],
    y=['Actual: Legitimate', 'Actual: Fraud'],
    text=annot,
    texttemplate='%{text}',
    textfont=dict(size=14),
    colorscale='RdYlGn',
    showscale=False,
    zmin=-max(r['fp'], r['fn']) * 3,
    zmax=max(r['tn'], r['tp']),
))
fig_cm.update_layout(
    title=f'Confusion Matrix -- {best_test["name"]} (Test Set)',
    xaxis=dict(side='top'),
    height=380,
    margin=dict(t=120),
)
figs.append(('Confusion Matrix', fig_cm))

# Precision / Recall / F1 comparison -- grouped bars with value labels
metric_colors = {'Precision': '#636EFA', 'Recall': '#00CC96', 'F1': '#EF553B'}
fig_metrics = go.Figure()
for metric, color in metric_colors.items():
    values = [r[metric.lower()] for r in test_results]
    fig_metrics.add_trace(go.Bar(
        name=metric,
        x=[r['name'] for r in test_results],
        y=values,
        marker_color=color,
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        textfont=dict(size=11),
    ))
fig_metrics.update_layout(
    title='Precision / Recall / F1 by Model -- Test Set',
    barmode='group',
    yaxis=dict(range=[0, 1.08], title='Score'),
    height=460,
    legend=dict(orientation='h', y=1.06, x=0),
)
figs.append(('Model Comparison', fig_metrics))

out_path = os.path.join(SCRIPT_DIR, 'results.html')
with open(out_path, 'w') as f:
    f.write("<html><head><title>Phase 1 -- Fraud Detection Results</title></head><body>\n")
    first = True
    for title, fig in figs:
        f.write(f"<h2>{title}</h2>\n")
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False))
        f.write("\n<hr>\n")
        first = False
    f.write("</body></html>")

print(f"  Saved -> phase1_baseline/results.html")

print("\n" + "=" * 60)
print("  PHASE 1 COMPLETE")
print("=" * 60)
print(f"  Best model: {best_test['name']}")
print(f"  Test F1: {best_test['f1']:.4f}  |  Recall: {best_test['recall']:.4f}"
      f"  |  ROC-AUC: {best_test['auc']:.4f}")
print(f"  phase1_baseline/results.html  -- ROC curves and charts")
print(f"  models/  -- saved models and scaler")
print("=" * 60)
