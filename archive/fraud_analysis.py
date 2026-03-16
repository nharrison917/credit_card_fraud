"""
Credit Card Fraud Detection - Complete Analysis
Business Analytics ML Pipeline
"""

import os
import pandas as pd
import numpy as np
import joblib

# Resolve paths relative to this script's location, so it runs correctly
# from any working directory (project root, baseline/, VS Code, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ============================================================
# CRITICAL DATA RULES - Load and split FIRST
# ============================================================
print("=" * 60)
print("LOADING DATA AND CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 60)

df = pd.read_csv(os.path.join(SCRIPT_DIR, '../data/creditcard.csv'))

# Split: 70% train, 15% val, 15% test — stratified
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train size:      {len(X_train):,} rows")
print(f"Validation size: {len(X_val):,} rows")
print(f"Test size:       {len(X_test):,} rows")
print(f"\nTest set is now locked away — will not be touched until Phase 5.\n")

# ============================================================
# PHASE 1 — Understand the data
# ============================================================
print("=" * 60)
print("PHASE 1 — DATA UNDERSTANDING")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

class_counts = y.value_counts()
fraud_pct = class_counts[1] / len(y) * 100
print("Class distribution (full dataset):")
print(f"  Legitimate (0): {class_counts[0]:,}  ({100 - fraud_pct:.4f}%)")
print(f"  Fraud      (1): {class_counts[1]:,}  ({fraud_pct:.4f}%)")

print("\nAmount statistics by Class:")
amount_stats = df.groupby('Class')['Amount'].describe()
print(amount_stats.to_string())

print("\nTime statistics by Class:")
time_stats = df.groupby('Class')['Time'].describe()
print(time_stats.to_string())

print("""
CLASS IMBALANCE EXPLANATION
----------------------------
Only {:.4f}% of transactions are fraudulent. This extreme imbalance creates
problems for standard ML models:
  - A naive model that predicts "legitimate" every time achieves {:.2f}% accuracy
    but catches ZERO fraud cases — useless in practice.
  - Standard metrics like accuracy are misleading; we need Precision, Recall,
    F1, and ROC-AUC to fairly evaluate model performance.
  - We compare three imbalance strategies: SMOTE 50/50, class weights, and
    modest SMOTE (1:10 ratio) to find what works best on this data.
""".format(fraud_pct, 100 - fraud_pct))

# ============================================================
# PHASE 2 — Exploratory Analysis → eda.html
# ============================================================
print("=" * 60)
print("PHASE 2 — EXPLORATORY ANALYSIS")
print("=" * 60)

# Use the TRAINING split for EDA to avoid data leakage
df_train = X_train.copy()
df_train['Class'] = y_train.values

figs = []

# --- Amount distribution by Class ---
print("  Plotting Amount distributions...")
fig_amt = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Amount by Class (Box Plot)", "Amount Distribution (Histogram)")
)
for cls, color, label in [(0, 'steelblue', 'Legitimate'), (1, 'crimson', 'Fraud')]:
    sub = df_train[df_train['Class'] == cls]['Amount']
    fig_amt.add_trace(
        go.Box(y=sub, name=label, marker_color=color, boxmean=True),
        row=1, col=1
    )
    fig_amt.add_trace(
        go.Histogram(x=sub, name=label, marker_color=color, opacity=0.6,
                     nbinsx=80, showlegend=False),
        row=1, col=2
    )
fig_amt.update_layout(title="Transaction Amount by Class", height=500)
figs.append(("Amount by Class", fig_amt))

# --- Time distribution by Class ---
print("  Plotting Time distributions...")
fig_time = go.Figure()
for cls, color, label in [(0, 'steelblue', 'Legitimate'), (1, 'crimson', 'Fraud')]:
    sub = df_train[df_train['Class'] == cls]['Time']
    fig_time.add_trace(go.Histogram(
        x=sub, name=label, marker_color=color, opacity=0.6, nbinsx=100
    ))
fig_time.update_layout(
    title="Time of Transaction by Class",
    xaxis_title="Time (seconds from first transaction)",
    yaxis_title="Count",
    barmode='overlay',
    height=450
)
figs.append(("Time by Class", fig_time))

# --- Top 10 V-features by class separation ---
print("  Identifying top 10 V-features by class separation...")
v_cols = [f'V{i}' for i in range(1, 29)]
fraud_means = df_train[df_train['Class'] == 1][v_cols].mean()
legit_means = df_train[df_train['Class'] == 0][v_cols].mean()
separation = (fraud_means - legit_means).abs().sort_values(ascending=False)
top10 = separation.head(10).index.tolist()
print(f"  Top 10 features: {top10}")

fig_violin = make_subplots(
    rows=2, cols=5,
    subplot_titles=top10,
    vertical_spacing=0.15
)
for i, feat in enumerate(top10):
    row, col = divmod(i, 5)
    for cls, color, label in [(0, 'steelblue', 'Legitimate'), (1, 'crimson', 'Fraud')]:
        sub = df_train[df_train['Class'] == cls][feat]
        fig_violin.add_trace(
            go.Violin(y=sub, name=label, marker_color=color,
                      showlegend=(i == 0), box_visible=True, meanline_visible=True),
            row=row + 1, col=col + 1
        )
fig_violin.update_layout(
    title="Top 10 V-Features: Distribution by Class (Violin Plots)",
    height=700, violinmode='group'
)
figs.append(("Top 10 V-Features Violin", fig_violin))

# --- Correlation of each feature with Class ---
print("  Plotting feature-Class correlations...")
all_features = v_cols + ['Amount', 'Time']
correlations = df_train[all_features + ['Class']].corr()['Class'].drop('Class')
correlations_sorted = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)
colors = ['crimson' if v > 0 else 'steelblue' for v in correlations_sorted.values]
fig_corr = go.Figure(go.Bar(
    x=correlations_sorted.index,
    y=correlations_sorted.values,
    marker_color=colors
))
fig_corr.update_layout(
    title="Feature Correlation with Class (ranked by |correlation|)",
    xaxis_title="Feature",
    yaxis_title="Pearson Correlation with Class",
    height=500
)
figs.append(("Feature Correlations", fig_corr))

# Write eda.html
print("  Saving eda.html...")
with open(os.path.join(SCRIPT_DIR, 'eda.html'), 'w') as f:
    f.write("<html><head><title>EDA - Credit Card Fraud</title></head><body>\n")
    f.write("<h1>Exploratory Data Analysis — Credit Card Fraud Detection</h1>\n")
    for title, fig in figs:
        f.write(f"<h2>{title}</h2>\n")
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if title == figs[0][0] else False))
        f.write("\n<hr>\n")
    f.write("</body></html>")
print("  eda.html saved.\n")

# ============================================================
# PHASE 3 — Preprocessing & Imbalance Strategies
# ============================================================
print("=" * 60)
print("PHASE 3 — PREPROCESSING & IMBALANCE STRATEGIES")
print("=" * 60)

# Scale Amount and Time — fit on train only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_val_scaled[['Amount', 'Time']] = scaler.transform(X_val[['Amount', 'Time']])
X_test_scaled[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

print(f"Scaler fit on training data only — applied to val and test sets.")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nClass distribution in training set:")
print(f"  Legitimate: {neg_count:,}  |  Fraud: {pos_count:,}")
print(f"  Imbalance ratio (neg/pos): {scale_pos_weight:.1f}:1")

# Strategy A: SMOTE 50/50 — full resampling to equality
print(f"\nStrategy A — SMOTE 50/50 (full resampling to equality):")
smote_full = SMOTE(random_state=42)
X_train_full, y_train_full = smote_full.fit_resample(X_train_scaled, y_train)
after_full = pd.Series(y_train_full).value_counts()
print(f"  Legitimate: {after_full[0]:,}  |  Fraud: {after_full[1]:,}")
print(f"  ({after_full[1] - pos_count:,} synthetic fraud samples generated)")

# Strategy B: Class weights — no resampling, loss function penalises minority errors
print(f"\nStrategy B — Class weights (no resampling):")
print(f"  LR / RF: class_weight='balanced'  |  XGBoost: scale_pos_weight={scale_pos_weight:.1f}")

# Strategy C: Modest SMOTE — 1:10 minority:majority ratio
print(f"\nStrategy C — Modest SMOTE (1:10 minority:majority ratio):")
smote_modest = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_mod, y_train_mod = smote_modest.fit_resample(X_train_scaled, y_train)
after_mod = pd.Series(y_train_mod).value_counts()
print(f"  Legitimate: {after_mod[0]:,}  |  Fraud: {after_mod[1]:,}")
print(f"  ({after_mod[1] - pos_count:,} synthetic fraud samples generated)")

print("""
WHY THREE STRATEGIES?
----------------------
  A (SMOTE 50/50):   Common default but risky — generates ~{:,} synthetic fraud
                     samples from only {:,} real ones. Can overfit to the
                     interpolated synthetic distribution.
  B (Class weights): Free lunch — no synthetic data, just reweights the loss
                     function. Often matches or beats SMOTE. Always try first.
  C (Modest SMOTE):  Middle ground — reduces imbalance without flooding the
                     training set. Benchmarks often show this beats 50/50.
""".format(after_full[1] - pos_count, pos_count))

# ============================================================
# PHASE 4 — Train & Compare Three Models
# ============================================================
print("=" * 60)
print("PHASE 4 — MODEL TRAINING & VALIDATION COMPARISON")
print("=" * 60)

def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    total_fraud = fn + tp
    missed_pct = fn / total_fraud * 100 if total_fraud > 0 else 0
    false_alarm_pct = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

    print(f"\n--- {name} ---")
    print(f"Confusion Matrix:")
    print(f"  True Negatives  (correct legit):  {tn:,}")
    print(f"  False Positives (legit flagged):   {fp:,}")
    print(f"  False Negatives (fraud missed):    {fn:,}")
    print(f"  True Positives  (fraud caught):    {tp:,}")
    print(f"Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}  |  ROC-AUC: {auc:.4f}")
    print(f"Plain English: This model catches {recall*100:.1f}% of actual fraud cases,")
    print(f"  misses {missed_pct:.1f}% of fraud, and raises false alarms on {false_alarm_pct:.2f}% of legitimate transactions.")

    return {'name': name, 'f1': f1, 'auc': auc, 'precision': precision,
            'recall': recall, 'model': model, 'y_prob': y_prob, 'y_pred': y_pred}

results = []

strategies = [
    ("SMOTE 50/50",   X_train_full,   y_train_full,  False),
    ("Class Weights", X_train_scaled, y_train,        True),
    ("Modest SMOTE",  X_train_mod,    y_train_mod,    False),
]

for strat_name, X_tr, y_tr, use_cw in strategies:
    print(f"\n{'=' * 55}")
    print(f"  STRATEGY: {strat_name}")
    print(f"{'=' * 55}")

    print(f"\nTraining Logistic Regression ({strat_name})...")
    lr = LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1,
        class_weight='balanced' if use_cw else None
    )
    lr.fit(X_tr, y_tr)
    results.append(evaluate_model(f"LR — {strat_name}", lr, X_val_scaled, y_val))

    print(f"\nTraining Random Forest ({strat_name})...")
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight='balanced' if use_cw else None
    )
    rf.fit(X_tr, y_tr)
    results.append(evaluate_model(f"RF — {strat_name}", rf, X_val_scaled, y_val))

    print(f"\nTraining XGBoost ({strat_name})...")
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric='logloss',
        use_label_encoder=False, n_jobs=-1,
        scale_pos_weight=scale_pos_weight if use_cw else 1.0
    )
    xgb.fit(X_tr, y_tr)
    results.append(evaluate_model(f"XGB — {strat_name}", xgb, X_val_scaled, y_val))

# Summary table grouped by strategy
print("\n\nVALIDATION SUMMARY — ALL MODELS × ALL STRATEGIES")
print(f"{'Model':<6} {'Strategy':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 68)
for strat_name, _, _, _ in strategies:
    for r in results:
        if r['name'].endswith(strat_name):
            short = r['name'].replace(f" — {strat_name}", "")
            print(f"{short:<6} {strat_name:<16} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")
    print()

# ============================================================
# PHASE 5 — Best Model & Final Test Evaluation
# ============================================================
print("\n")
print("=" * 60)
print("PHASE 5 — BEST MODEL SELECTION & TEST SET EVALUATION")
print("=" * 60)

best = max(results, key=lambda r: r['f1'])
print(f"\nBest model by validation F1: {best['name']}  (F1={best['f1']:.4f})")
best_model = best['model']

print("""
WHY F1 OVER ACCURACY?
----------------------
With {:.4f}% fraud, a model predicting "all legitimate" gets {:.2f}% accuracy
but zero real value. F1 is the harmonic mean of Precision and Recall:
  - Precision: Of flagged transactions, how many were actually fraud?
  - Recall:    Of actual fraud cases, how many did we catch?
F1 penalizes models that sacrifice one for the other, making it the right
single metric when both false positives (wasted investigations) and false
negatives (missed fraud, real losses) carry business cost.
""".format(fraud_pct, 100 - fraud_pct))

print(">>> NOW using the test set for the FIRST AND ONLY TIME <<<\n")

y_test_pred = best_model.predict(X_test_scaled)
y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]

cm_test = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm_test.ravel()
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"FINAL TEST SET RESULTS — {best['name']}")
print(f"  True Negatives  (correct legit):  {tn:,}")
print(f"  False Positives (legit flagged):   {fp:,}")
print(f"  False Negatives (fraud missed):    {fn:,}")
print(f"  True Positives  (fraud caught):    {tp:,}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")

# Build results.html
result_figs = []

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode='lines', name=f'{best["name"]} (AUC={test_auc:.4f})',
    line=dict(color='crimson', width=2)
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
    line=dict(color='gray', dash='dash')
))
fig_roc.update_layout(
    title=f"ROC Curve — {best['name']} (Test Set)",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=500
)
result_figs.append(("ROC Curve (Test Set)", fig_roc))

# Confusion Matrix heatmap
fig_cm = go.Figure(go.Heatmap(
    z=cm_test,
    x=['Predicted Legit', 'Predicted Fraud'],
    y=['Actual Legit', 'Actual Fraud'],
    text=[[str(v) for v in row] for row in cm_test],
    texttemplate="%{text}",
    colorscale='RdBu_r',
    showscale=True
))
fig_cm.update_layout(
    title=f"Confusion Matrix — {best['name']} (Test Set)",
    height=450
)
result_figs.append(("Confusion Matrix (Test Set)", fig_cm))

with open(os.path.join(SCRIPT_DIR, 'results.html'), 'w') as f:
    f.write("<html><head><title>Results - Credit Card Fraud</title></head><body>\n")
    f.write(f"<h1>Final Model Results — {best['name']}</h1>\n")
    f.write(f"<p><b>Precision:</b> {test_precision:.4f} &nbsp; "
            f"<b>Recall:</b> {test_recall:.4f} &nbsp; "
            f"<b>F1:</b> {test_f1:.4f} &nbsp; "
            f"<b>ROC-AUC:</b> {test_auc:.4f}</p>\n")
    first = True
    for title, fig in result_figs:
        f.write(f"<h2>{title}</h2>\n")
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False))
        first = False
        f.write("\n<hr>\n")
    f.write("</body></html>")
print("\nresults.html saved.")

# ============================================================
# PHASE 6 — Interpret Results
# ============================================================
print("\n")
print("=" * 60)
print("PHASE 6 — INTERPRETATION & BUSINESS CONTEXT")
print("=" * 60)

# Feature importance
print(f"\n--- Feature Importance ({best['name']}) ---")
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    print(top_features.to_string())
elif hasattr(best_model, 'coef_'):
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    print("(Logistic Regression: absolute coefficient values)")
    print(top_features.to_string())

# Operational meaning — scaled to 1000 transactions
total_test = len(y_test)
test_fraud_count = y_test.sum()
scale = 1000 / total_test

flagged = fp + tp
true_fraud_flagged = tp
false_alarms = fp

flagged_per_1k = flagged * scale
true_fraud_per_1k = true_fraud_flagged * scale
false_alarm_per_1k = false_alarms * scale
missed_per_1k = fn * scale

print(f"""
--- Operational Impact (scaled to 1,000 transactions) ---
  Transactions reviewed: 1,000
  Expected fraud cases:  ~{test_fraud_count * scale:.1f}

  Model would FLAG:      ~{flagged_per_1k:.1f} transactions for investigation
    - Genuine fraud:     ~{true_fraud_per_1k:.1f}  (correctly caught)
    - False alarms:      ~{false_alarm_per_1k:.1f}  (legitimate, wasted investigator time)

  Fraud MISSED:          ~{missed_per_1k:.1f} cases slip through undetected

This means the fraud team investigates ~{flagged_per_1k:.0f} alerts per 1,000
transactions, catching ~{true_fraud_per_1k:.1f} real fraud cases while generating
~{false_alarm_per_1k:.0f} false alarms. Each trade-off (precision vs recall) can
be tuned by adjusting the classification threshold.
""")

print("""--- Limitations of This Analysis ---
1. PCA anonymisation: V1-V28 are already transformed, so we cannot interpret
   them directly (e.g., "V14 corresponds to merchant category"). Feature
   importance shows statistical signal, not business meaning.

2. Time-based leakage risk: We split randomly rather than by time. In
   production, models should always be trained on past data and evaluated
   on future data to simulate real deployment.

3. SMOTE is synthetic: Oversampled minority examples are interpolated, not
   real transactions. Models may be overfit to the synthetic distribution.

4. Dataset age: This dataset is from 2013 European cardholders. Fraud
   patterns evolve — a model trained on it may not generalise to modern fraud.

5. No cost-sensitive learning: We treated all errors equally. In reality,
   missing $50,000 fraud is far worse than a false alarm on a $20 purchase.
   Threshold tuning or cost-sensitive training would improve real-world ROI.

6. Single dataset: No out-of-time or out-of-distribution validation was
   performed. Deployment performance should be monitored continuously.
""")

# ============================================================
# SAVE MODELS
# ============================================================
best_lr = max((r for r in results if r['name'].startswith('LR')), key=lambda r: r['f1'])
best_rf = max((r for r in results if r['name'].startswith('RF')), key=lambda r: r['f1'])

joblib.dump(best_lr['model'], os.path.join(SCRIPT_DIR, '../models/lr_best.pkl'))
joblib.dump(best_rf['model'], os.path.join(SCRIPT_DIR, '../models/rf_best.pkl'))
joblib.dump(scaler,           os.path.join(SCRIPT_DIR, '../models/scaler.pkl'))

print(f"\nModels saved to models/")
print(f"  lr_best.pkl  — {best_lr['name']}  (F1={best_lr['f1']:.4f})")
print(f"  rf_best.pkl  — {best_rf['name']}  (F1={best_rf['f1']:.4f})")
print(f"  scaler.pkl   — StandardScaler fit on training Amount & Time")

print("=" * 60)
print("ANALYSIS COMPLETE")
print("  eda.html     — Exploratory Data Analysis plots")
print("  results.html — Final model ROC curve and confusion matrix")
print("  fraud_analysis.py — Full reproducible code")
print("=" * 60)
