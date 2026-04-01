"""
Cost-Sensitive Fraud Detection Analysis
========================================
Extends the baseline fraud analysis with business cost optimisation.
Run from anywhere: python phase2_cost_analysis/cost_fraud_analysis.py

COST ASSUMPTIONS (illustrative -- not specific to any institution):
  - Base FP cost:    $10 per flagged transaction (industry est. $7-10)
  - Friction rate:    1% of Amount added to FP cost
  - Full FP cost:    $10 + (0.01 x Amount)
  - FN cost:         full transaction Amount (worst-case assumption)
"""

import os
import json

# Resolve all paths relative to this script's location so it runs correctly
# from any working directory (project root, phase2_cost_analysis/, VS Code, etc.)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '../models'))
DATA_PATH   = os.path.normpath(os.path.join(SCRIPT_DIR, '../data/creditcard.csv'))
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import joblib
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ============================================================
# SETUP -- directories
# ============================================================
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("  COST-SENSITIVE FRAUD DETECTION ANALYSIS")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/9] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} transactions")
print(f"  Fraud:  {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)")
print(f"  Legit:  {(df['Class']==0).sum():,}  ({(df['Class']==0).mean()*100:.3f}%)")

# ============================================================
# DATA SPLIT -- identical methodology to baseline for comparability
# 70% train / 15% val / 15% test, stratified, random_state=42
# ============================================================
print("\n[2/9] Splitting data (70/15/15, stratified, random_state=42)...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Preserve raw Amount values BEFORE scaling -- needed for cost calculations
amounts_train = X_train['Amount'].copy()
amounts_val   = X_val['Amount'].copy()
amounts_test  = X_test['Amount'].copy()

print(f"  Train: {len(X_train):,}  (fraud: {y_train.sum():,}  |  legit: {(y_train==0).sum():,})")
print(f"  Val:   {len(X_val):,}  (fraud: {y_val.sum():,}  |  legit: {(y_val==0).sum():,})")
print(f"  Test:  {len(X_test):,}  (fraud: {y_test.sum():,}  |  legit: {(y_test==0).sum():,})")
print("  NOTE: Test set locked -- not used until final evaluation (step 8).")

# ============================================================
# SCALING -- fit on training data only, apply to all sets
# ============================================================
print("\n[3/9] Fitting StandardScaler on training data (Amount + Time only)...")
scaler = StandardScaler()
X_train_sc = X_train.copy()
X_val_sc   = X_val.copy()
X_test_sc  = X_test.copy()

X_train_sc[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_val_sc[['Amount', 'Time']]   = scaler.transform(X_val[['Amount', 'Time']])
X_test_sc[['Amount', 'Time']]  = scaler.transform(X_test[['Amount', 'Time']])

# ============================================================
# SMOTE -- 1:10 ratio on training set only
# sampling_strategy=0.1 -> minority/majority = 0.1 after resampling
# ============================================================
print("\n[4/9] Applying SMOTE (1:10 minority:majority) to training set only...")
print(f"  Before: {(y_train==0).sum():,} legit / {(y_train==1).sum()} fraud")
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"  After:  {(y_train_sm==0).sum():,} legit / {(y_train_sm==1).sum():,} fraud")
print(f"  Ratio:  1:{(y_train_sm==0).sum() / (y_train_sm==1).sum():.1f}  (legit per fraud)")

# ============================================================
# COST FUNCTION
# ============================================================
print("\n[5/9] Cost function defined:")
print("""
  ASSUMPTIONS (illustrative -- state this in all outputs):
  --------------------------------------------------------
  These are illustrative figures for a portfolio analysis on an
  anonymized dataset. Real costs vary by institution.

  FP cost = $10  +  (0.01 x Amount)
              ^            ^
          investigation  friction penalty (larger blocked txn = more friction)

  FN cost = Amount  (full loss -- worst-case; no partial recovery assumed)
""")

def calculate_business_cost(y_true, y_pred, amounts, base_fp_cost=10, friction_rate=0.01):
    """
    Calculate total business cost of model predictions.

    Parameters
    ----------
    y_true       : array-like -- actual labels (0=legit, 1=fraud)
    y_pred       : array-like -- predicted labels
    amounts      : array-like -- raw transaction amounts (pre-scaling)
    base_fp_cost : float      -- flat investigation cost per false positive (default $10)
    friction_rate: float      -- fraction of Amount added to FP cost (default 0.01)

    Returns
    -------
    dict with keys:
        total_cost, fp_cost, fn_cost, n_fp, n_fn, cost_per_transaction
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    amounts = np.array(amounts)

    fp_mask = (y_pred == 1) & (y_true == 0)   # flagged as fraud, actually legit
    fn_mask = (y_pred == 0) & (y_true == 1)   # missed fraud

    fp_costs = base_fp_cost + friction_rate * amounts[fp_mask]
    fn_costs = amounts[fn_mask]

    total_fp_cost = fp_costs.sum()
    total_fn_cost = fn_costs.sum()
    total_cost    = total_fp_cost + total_fn_cost

    return {
        'total_cost':           total_cost,
        'fp_cost':              total_fp_cost,
        'fn_cost':              total_fn_cost,
        'n_fp':                 int(fp_mask.sum()),
        'n_fn':                 int(fn_mask.sum()),
        'cost_per_transaction': total_cost / len(y_true) if len(y_true) > 0 else 0,
    }

# ============================================================
# MODEL TRAINING
# ============================================================
print("[6/9] Training models...")

# XGBoost scale_pos_weight = legit/fraud in ORIGINAL training set
# (before SMOTE -- represents the true population imbalance)
spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  XGBoost scale_pos_weight = {spw:.1f}  (original imbalance ratio)")

print("\n  Training Logistic Regression (class_weight='balanced')...")
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
lr.fit(X_train_sm, y_train_sm)
joblib.dump(lr, os.path.join(MODELS_DIR, 'lr_cost_model.pkl'))
print("    Saved -> models/lr_cost_model.pkl")

print("\n  Training Random Forest (n_estimators=200, class_weight='balanced')...")
rf = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train_sm, y_train_sm)
joblib.dump(rf, os.path.join(MODELS_DIR, 'rf_cost_model.pkl'))
print("    Saved -> models/rf_cost_model.pkl")

print("\n  Training XGBoost (n_estimators=200, scale_pos_weight, aucpr)...")
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    colsample_bytree=0.8, subsample=0.8,
    scale_pos_weight=spw, eval_metric='aucpr',
    random_state=42, n_jobs=-1,
)
xgb.fit(X_train_sm, y_train_sm)
joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgb_cost_model.pkl'))
print("    Saved -> models/xgb_cost_model.pkl")

joblib.dump(scaler, os.path.join(MODELS_DIR, 'cost_scaler.pkl'))
print("    Saved -> models/cost_scaler.pkl")

# ============================================================
# THRESHOLD SWEEP (0.01 -> 0.99) on validation set
# ============================================================
print("\n[7/9] Threshold optimisation...")

thresholds   = np.arange(0.01, 1.00, 0.01)
models_dict  = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
sweep_store  = {}   # store full sweep DataFrames for plotting
model_metadata = {}

for model_name, model in models_dict.items():
    print(f"\n  {model_name}:")
    y_prob = model.predict_proba(X_val_sc)[:, 1]

    rows = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cost = calculate_business_cost(y_val, y_pred_t, amounts_val)
        rows.append({
            'threshold':  t,
            'total_cost': cost['total_cost'],
            'fp_cost':    cost['fp_cost'],
            'fn_cost':    cost['fn_cost'],
            'n_fp':       cost['n_fp'],
            'n_fn':       cost['n_fn'],
            'precision':  precision_score(y_val, y_pred_t, zero_division=0),
            'recall':     recall_score(y_val, y_pred_t, zero_division=0),
            'f1':         f1_score(y_val, y_pred_t, zero_division=0),
        })

    sweep_df = pd.DataFrame(rows)
    sweep_store[model_name] = sweep_df

    best_cost = sweep_df.loc[sweep_df['total_cost'].idxmin()]
    best_f1   = sweep_df.loc[sweep_df['f1'].idxmax()]

    cost_thresh = float(best_cost['threshold'])
    f1_thresh   = float(best_f1['threshold'])

    cost_at_cost = float(best_cost['total_cost'])
    cost_at_f1   = calculate_business_cost(
        y_val, (y_prob >= f1_thresh).astype(int), amounts_val
    )['total_cost']
    dollar_diff  = cost_at_f1 - cost_at_cost

    print(f"    Cost-optimal threshold: {cost_thresh:.2f}  ->  total cost: ${cost_at_cost:,.2f}")
    print(f"    F1-optimal  threshold:  {f1_thresh:.2f}  ->  total cost: ${cost_at_f1:,.2f}")
    print(f"    Dollar difference (F1-opt vs cost-opt): ${dollar_diff:,.2f}")
    print(f"    At cost-optimal:  FP={int(best_cost['n_fp'])}  FN={int(best_cost['n_fn'])}"
          f"  P={best_cost['precision']:.3f}  R={best_cost['recall']:.3f}  F1={best_cost['f1']:.3f}")

    model_metadata[model_name] = {
        'cost_optimal_threshold':              cost_thresh,
        'f1_optimal_threshold':                f1_thresh,
        'validation_cost_at_optimal_threshold': cost_at_cost,
        'dollar_cost_of_using_f1_threshold':   round(dollar_diff, 2),
        'hyperparameters': {},
    }

# Populate hyperparameters in metadata
model_metadata['Logistic Regression']['hyperparameters'] = {
    'max_iter': 1000, 'class_weight': 'balanced',
}
model_metadata['Random Forest']['hyperparameters'] = {
    'n_estimators': 200, 'class_weight': 'balanced',
}
model_metadata['XGBoost']['hyperparameters'] = {
    'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
    'colsample_bytree': 0.8, 'subsample': 0.8, 'scale_pos_weight': float(round(spw, 2)),
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(model_metadata, f, indent=2)
print("\n  Saved -> models/model_metadata.json")

# ============================================================
# SENSITIVITY ANALYSIS -- FP cost scenarios, best model only
# ============================================================
print("\n  Sensitivity analysis -- FP cost scenarios:")

# Pick best model by validation cost
best_model_name = min(
    model_metadata,
    key=lambda m: model_metadata[m]['validation_cost_at_optimal_threshold']
)
best_model      = models_dict[best_model_name]
y_prob_best_val = best_model.predict_proba(X_val_sc)[:, 1]

scenarios = [
    ('A -- Low',       5,  'Low ($5)'),
    ('B -- Base',      10, 'Base ($10)'),
    ('C -- High',      25, 'High ($25)'),
    ('D -- Very High', 50, 'Very High ($50)'),
]

sensitivity_results = {}
print(f"\n  Best model for sensitivity: {best_model_name}")
print(f"\n  {'Scenario':<20} {'FP Cost':>10} {'Opt Threshold':>14} {'Total Cost':>13}")
print("  " + "-" * 62)

for sc_id, fp_cost, sc_label in scenarios:
    rows = []
    for t in thresholds:
        y_pred_t = (y_prob_best_val >= t).astype(int)
        cost = calculate_business_cost(y_val, y_pred_t, amounts_val, base_fp_cost=fp_cost)
        rows.append({'threshold': t, 'total_cost': cost['total_cost']})
    sc_df    = pd.DataFrame(rows)
    best_row = sc_df.loc[sc_df['total_cost'].idxmin()]
    sensitivity_results[sc_label] = {
        'fp_cost':       fp_cost,
        'sweep':         sc_df,
        'opt_threshold': float(best_row['threshold']),
        'total_cost':    float(best_row['total_cost']),
    }
    print(f"  {sc_label:<20} ${fp_cost:>8}   {best_row['threshold']:>12.2f}   ${best_row['total_cost']:>10,.2f}")

# ============================================================
# FINAL EVALUATION -- test set (used here for the first time)
# ============================================================
print("\n[8/9] Final evaluation on TEST SET...")
print("  >>> Using test set for the first and only time <<<\n")

test_rows = []
for model_name, model in models_dict.items():
    y_prob_test = model.predict_proba(X_test_sc)[:, 1]
    auc         = roc_auc_score(y_test, y_prob_test)
    opt_thresh  = model_metadata[model_name]['cost_optimal_threshold']
    scale_1M    = 1_000_000 / len(y_test)

    for thresh_label, thresh in [('Cost-Optimal', opt_thresh), ('0.5 Default', 0.50)]:
        y_pred = (y_prob_test >= thresh).astype(int)
        cost   = calculate_business_cost(y_test, y_pred, amounts_test)
        test_rows.append({
            'model':          model_name,
            'threshold_type': thresh_label,
            'threshold':      thresh,
            'total_cost':     cost['total_cost'],
            'fp_cost':        cost['fp_cost'],
            'fn_cost':        cost['fn_cost'],
            'n_fp':           cost['n_fp'],
            'n_fn':           cost['n_fn'],
            'precision':      precision_score(y_test, y_pred, zero_division=0),
            'recall':         recall_score(y_test, y_pred, zero_division=0),
            'f1':             f1_score(y_test, y_pred, zero_division=0),
            'auc':            auc,
            'annual_cost_1M': cost['total_cost'] * scale_1M,
        })

test_df = pd.DataFrame(test_rows)

# Print full comparison table
hdr = (f"  {'Model':<22} {'Type':<13} {'Thr':>5} {'Total$':>9} {'FP$':>8} "
       f"{'FN$':>8} {'nFP':>5} {'nFN':>5} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for _, row in test_df.iterrows():
    print(f"  {row['model']:<22} {row['threshold_type']:<13} {row['threshold']:>5.2f}"
          f" ${row['total_cost']:>7,.0f} ${row['fp_cost']:>6,.0f} ${row['fn_cost']:>6,.0f}"
          f" {row['n_fp']:>5} {row['n_fn']:>5}"
          f" {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f} {row['auc']:>7.4f}")

# Identify best model on test set at cost-optimal threshold
cost_opt_df   = test_df[test_df['threshold_type'] == 'Cost-Optimal'].copy()
best_test_row = cost_opt_df.loc[cost_opt_df['total_cost'].idxmin()]
best_name     = best_test_row['model']

#  Operational interpretation (scaled to 10,000 transactions) 
scale_10k     = 10_000 / len(y_test)
tp_count      = y_test.sum() - best_test_row['n_fn']
flagged_10k   = (best_test_row['n_fp'] + tp_count) * scale_10k
genuine_10k   = tp_count * scale_10k
false_alarm_10k = best_test_row['n_fp'] * scale_10k
fp_cost_10k   = best_test_row['fp_cost'] * scale_10k
fn_cost_10k   = best_test_row['fn_cost'] * scale_10k
no_det_cost   = amounts_test[y_test == 1].sum()
net_saving_10k = (no_det_cost - best_test_row['total_cost']) * scale_10k

print(f"""
  OPERATIONAL INTERPRETATION -- {best_name} at threshold {best_test_row['threshold']:.2f}

  At a cost-optimal threshold of {best_test_row['threshold']:.2f}, for every 10,000 transactions
  this model flags ~{flagged_10k:.0f} as fraud, of which ~{genuine_10k:.0f} are genuine fraud and
  ~{false_alarm_10k:.0f} are false alarms. The estimated investigation and friction cost
  of false alarms is ~${fp_cost_10k:,.0f}, offset against ~${fn_cost_10k:,.0f} in fraud
  prevented, for a net saving of ~${net_saving_10k:,.0f} compared to no fraud detection.
""")

# ============================================================
# DASHBOARD DATA EXPORT -> models/dashboard_data.json
# Powers the Streamlit dashboard (app.py).
# Contains validation sweep data (per-threshold FP/FN counts and costs)
# and test set results at default + cost-optimal thresholds.
# The friction component stored implicitly in fp_cost allows the dashboard
# to recompute costs for any FP base cost without re-running models:
#   friction_part = fp_cost - BASE_FP_COST * n_fp
#   new_fp_cost   = new_base * n_fp + friction_part
# ============================================================
print("\n  Exporting dashboard data...")

dashboard_val_sweep = {}
for model_name, df in sweep_store.items():
    int_cols = {'n_fp', 'n_fn'}
    dashboard_val_sweep[model_name] = {
        col: [int(v) if col in int_cols else round(float(v), 6)
              for v in df[col]]
        for col in df.columns
    }

dashboard_test_results = {}
for _, row in test_df.iterrows():
    m     = row['model']
    ttype = 'cost_optimal' if row['threshold_type'] == 'Cost-Optimal' else 'default_0_50'
    if m not in dashboard_test_results:
        dashboard_test_results[m] = {}
    dashboard_test_results[m][ttype] = {
        'threshold':  float(row['threshold']),
        'total_cost': round(float(row['total_cost']), 4),
        'fp_cost':    round(float(row['fp_cost']), 4),
        'fn_cost':    round(float(row['fn_cost']), 4),
        'n_fp':       int(row['n_fp']),
        'n_fn':       int(row['n_fn']),
        'recall':     round(float(row['recall']), 6),
        'precision':  round(float(row['precision']), 6),
        'f1':         round(float(row['f1']), 6),
        'auc':        round(float(row['auc']), 6),
    }

dashboard_data = {
    'metadata': {
        'base_fp_cost':           10,
        'friction_rate':          0.01,
        'n_val':                  int(len(y_val)),
        'n_fraud_val':            int(y_val.sum()),
        'total_fraud_amount_val': round(float(amounts_val[y_val == 1].sum()), 4),
        'n_test':                 int(len(y_test)),
        'n_fraud_test':           int(y_test.sum()),
    },
    'validation_sweep': dashboard_val_sweep,
    'test_results':     dashboard_test_results,
}

dashboard_path = os.path.join(MODELS_DIR, 'dashboard_data.json')
with open(dashboard_path, 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, indent=2)
print("  Saved -> models/dashboard_data.json")

# ============================================================
# VISUALIZATIONS -> phase2_cost_analysis/cost_results.html
# ============================================================
print("[9/9] Building visualisations...")

MODEL_COLORS = {
    'Logistic Regression': '#636EFA',
    'Random Forest':       '#00CC96',
    'XGBoost':             '#EF553B',
}
SA_COLORS = {
    'Low ($5)':       '#636EFA',
    'Base ($10)':     '#00CC96',
    'High ($25)':     '#EF553B',
    'Very High ($50)':'#AB63FA',
}

figs = []

#  Fig 1: Cost curves per model 
fig1 = make_subplots(
    rows=1, cols=3,
    subplot_titles=list(models_dict.keys()),
    shared_yaxes=True,
)
for col_i, (model_name, _) in enumerate(models_dict.items(), 1):
    sweep  = sweep_store[model_name]
    opt_t  = model_metadata[model_name]['cost_optimal_threshold']
    f1_t   = model_metadata[model_name]['f1_optimal_threshold']
    color  = MODEL_COLORS[model_name]

    fig1.add_trace(go.Scatter(
        x=sweep['threshold'], y=sweep['total_cost'],
        name=model_name, line=dict(color=color),
        showlegend=False,
    ), row=1, col=col_i)
    fig1.add_vline(
        x=opt_t, line_dash='dash', line_color='green',
        annotation_text=f'Cost-opt {opt_t:.2f}',
        annotation_font_color='green', row=1, col=col_i,
    )
    fig1.add_vline(
        x=f1_t, line_dash='dot', line_color='orange',
        annotation_text=f'F1-opt {f1_t:.2f}',
        annotation_font_color='darkorange', row=1, col=col_i,
    )

fig1.update_layout(
    title='Business Cost Across Thresholds -- Validation Set',
    yaxis_title='Total Cost ($)', height=480,
)
fig1.update_xaxes(title_text='Threshold')
figs.append(('Cost Curves by Model', fig1))

#  Fig 2: Sensitivity analysis 
fig2 = go.Figure()
for sc_label, sc_data in sensitivity_results.items():
    color = SA_COLORS.get(sc_label, '#666')
    fig2.add_trace(go.Scatter(
        x=sc_data['sweep']['threshold'],
        y=sc_data['sweep']['total_cost'],
        name=sc_label,
        line=dict(color=color, width=2),
    ))
    fig2.add_vline(
        x=sc_data['opt_threshold'],
        line_dash='dash', line_color=color, opacity=0.6,
    )
fig2.update_layout(
    title=f'Sensitivity Analysis -- {best_model_name}: FP Cost Scenarios (Validation Set)',
    xaxis_title='Threshold', yaxis_title='Total Cost ($)',
    legend_title='FP Investigation Cost', height=480,
)
figs.append(('Sensitivity Analysis -- FP Cost Scenarios', fig2))

#  Fig 3: Cost breakdown bar charts at cost-optimal thresholds 
co_df   = test_df[test_df['threshold_type'] == 'Cost-Optimal'].copy()
fig3    = make_subplots(rows=1, cols=3,
                        subplot_titles=['Total Cost', 'FP Cost (investigation + friction)',
                                        'FN Cost (missed fraud)'])
for col_i, cost_col in enumerate(['total_cost', 'fp_cost', 'fn_cost'], 1):
    fig3.add_trace(go.Bar(
        x=co_df['model'],
        y=co_df[cost_col],
        marker_color=[MODEL_COLORS[m] for m in co_df['model']],
        showlegend=False,
        text=[f'${v:,.0f}' for v in co_df[cost_col]],
        textposition='outside',
    ), row=1, col=col_i)
fig3.update_layout(
    title='Cost Breakdown at Cost-Optimal Thresholds -- Test Set',
    height=460,
)
figs.append(('Cost Breakdown -- Test Set', fig3))

#  Fig 4: Scatter -- errors by Amount 
y_prob_best_test = best_model.predict_proba(X_test_sc)[:, 1]
opt_t_best       = model_metadata[best_name]['cost_optimal_threshold']
y_pred_best      = (y_prob_best_test >= opt_t_best).astype(int)

def classify_outcome(yt, yp):
    if   yt == 0 and yp == 0: return 'True Negative'
    elif yt == 1 and yp == 1: return 'True Positive'
    elif yt == 0 and yp == 1: return 'False Positive (investigation cost)'
    else:                      return 'False Negative (fraud missed)'

outcomes = [classify_outcome(yt, yp) for yt, yp in zip(
    y_test.values, y_pred_best
)]
scatter_df = pd.DataFrame({
    'Amount':  amounts_test.values,
    'Outcome': outcomes,
    'idx':     range(len(y_test)),
})

errors_df  = scatter_df[scatter_df['Outcome'].str.startswith('False')]
tp_df      = scatter_df[scatter_df['Outcome'] == 'True Positive']
tn_sample  = scatter_df[scatter_df['Outcome'] == 'True Negative'].sample(500, random_state=42)
plot_df    = pd.concat([tn_sample, tp_df, errors_df]).reset_index(drop=True)

OUTCOME_COLORS = {
    'True Negative':                        '#CCCCCC',
    'True Positive':                        '#00CC96',
    'False Positive (investigation cost)':  '#636EFA',
    'False Negative (fraud missed)':        '#EF553B',
}
fig4 = px.scatter(
    plot_df, x='Amount', y='idx',
    color='Outcome', color_discrete_map=OUTCOME_COLORS,
    title=f'{best_name} -- Errors by Transaction Amount (threshold={opt_t_best:.2f})',
    labels={'idx': 'Transaction index', 'Amount': 'Transaction Amount ($)'},
    opacity=0.7, height=560,
)
fig4.update_yaxes(showticklabels=False)
figs.append(('Errors by Transaction Amount', fig4))

# Write cost_results.html
with open(os.path.join(SCRIPT_DIR, 'cost_results.html'), 'w') as f:
    f.write("<html><head><title>Cost-Sensitive Fraud -- Charts</title></head><body>\n")
    first = True
    for title, fig in figs:
        f.write(f"<h2>{title}</h2>\n")
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False))
        f.write("\n<hr>\n")
        first = False
    f.write("</body></html>")
print("  Saved -> phase2_cost_analysis/cost_results.html")

# ============================================================
# STAKEHOLDER REPORT -> phase2_cost_analysis/cost_report.html
# ============================================================
today_str = date.today().strftime('%B %d, %Y')

def _tr(row):
    hl = ' style="background:#e8f5e9;"' if row['threshold_type'] == 'Cost-Optimal' else ''
    champ = ' ' if (row['model'] == best_name and row['threshold_type'] == 'Cost-Optimal') else ''
    return (
        f"<tr{hl}>"
        f"<td>{row['model']}{champ}</td>"
        f"<td>{row['threshold_type']}</td>"
        f"<td>{row['threshold']:.2f}</td>"
        f"<td>${row['total_cost']:,.0f}</td>"
        f"<td>${row['fp_cost']:,.0f}</td>"
        f"<td>${row['fn_cost']:,.0f}</td>"
        f"<td>{row['n_fp']}</td><td>{row['n_fn']}</td>"
        f"<td>{row['precision']:.3f}</td>"
        f"<td>{row['recall']:.3f}</td>"
        f"<td>{row['f1']:.3f}</td>"
        f"<td>{row['auc']:.4f}</td>"
        f"<td>${row['annual_cost_1M']:,.0f}</td>"
        f"</tr>"
    )

table_rows = '\n'.join(_tr(row) for _, row in test_df.iterrows())

sens_rows = '\n'.join(
    f"<tr><td>{lbl}</td><td>${d['fp_cost']}</td>"
    f"<td>{d['opt_threshold']:.2f}</td>"
    f"<td>${d['total_cost']:,.2f}</td></tr>"
    for lbl, d in sensitivity_results.items()
)

f1_thresh_best  = model_metadata[best_name]['f1_optimal_threshold']
cost_thresh_best = model_metadata[best_name]['cost_optimal_threshold']
dollar_diff_best = model_metadata[best_name]['dollar_cost_of_using_f1_threshold']

report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cost-Sensitive Fraud Detection -- Report</title>
<style>
  body      {{ font-family: Arial, sans-serif; max-width: 980px; margin: 40px auto; color: #222; line-height: 1.65; }}
  h1        {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2        {{ color: #283593; margin-top: 44px; }}
  .meta     {{ color: #666; font-size: 0.9em; margin-bottom: 28px; }}
  .box-warn {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  .box-good {{ background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 14px 20px; margin: 20px 0; border-radius: 4px; font-size: 1.05em; }}
  .box-info {{ background: #e3f2fd; border-left: 5px solid #1565c0; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  .box-risk {{ background: #fce4ec; border-left: 5px solid #c62828; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  table     {{ border-collapse: collapse; width: 100%; font-size: 0.83em; margin: 14px 0; }}
  th        {{ background: #1a237e; color: #fff; padding: 9px 10px; text-align: right; white-space: nowrap; }}
  th:first-child, th:nth-child(2) {{ text-align: left; }}
  td        {{ padding: 7px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  tr:hover  {{ background: #f5f5f5; }}
  small     {{ color: #888; }}
</style>
</head>
<body>

<h1>Cost-Sensitive Fraud Detection Analysis</h1>
<div class="meta">
  Generated: {today_str} &nbsp;|&nbsp;
  Dataset: European Credit Card Transactions (2013, anonymized) &nbsp;|&nbsp;
  Models: Logistic Regression &nbsp;&nbsp; Random Forest &nbsp;&nbsp; XGBoost
</div>

<div class="box-warn">
  <strong>Assumption Disclosure -- please read before interpreting figures</strong><br>
  These are <strong>illustrative cost estimates</strong> for a portfolio analysis conducted on
  an anonymized dataset. The dollar values below are not representative of any specific bank,
  card scheme, or geography. Real investigation and fraud recovery costs vary significantly by
  institution, transaction type, and contractual arrangements.
  <ul>
    <li>Base FP cost (investigation):  <strong>$10</strong> per flagged transaction &nbsp;<small>(industry estimate: $7-10)</small></li>
    <li>Friction penalty:              <strong>1%</strong> of Amount added to FP cost &nbsp;<small>(larger blocked txns create more friction)</small></li>
    <li>Full FP cost:                  <strong>$10 + (0.01 x Amount)</strong></li>
    <li>FN cost:                       <strong>full transaction Amount</strong> &nbsp;<small>(worst-case -- no partial recovery assumed)</small></li>
  </ul>
</div>

<h2>Key Findings</h2>
<div class="box-good">
  <strong>Best model:</strong> {best_name} at cost-optimal threshold <strong>{best_test_row['threshold']:.2f}</strong><br>
  <strong>Test set total cost:</strong> ${best_test_row['total_cost']:,.2f}
  &nbsp;|&nbsp; FP cost: ${best_test_row['fp_cost']:,.2f}
  &nbsp;|&nbsp; FN cost: ${best_test_row['fn_cost']:,.2f}<br>
  <strong>Recall:</strong> {best_test_row['recall']:.4f}
  &nbsp;|&nbsp; <strong>F1:</strong> {best_test_row['f1']:.4f}
  &nbsp;|&nbsp; <strong>ROC-AUC:</strong> {best_test_row['auc']:.4f}<br>
  <strong>Estimated annual cost at 1M transactions:</strong> ${best_test_row['annual_cost_1M']:,.0f}
</div>

<h2>Model Comparison -- Test Set</h2>
<p><small> = champion (lowest cost). Green rows = cost-optimal threshold results.</small></p>
<table>
  <tr>
    <th>Model</th><th>Threshold Type</th><th>Threshold</th>
    <th>Total Cost</th><th>FP Cost</th><th>FN Cost</th>
    <th>n FP</th><th>n FN</th>
    <th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th>
    <th>Est. Annual (1M txn)</th>
  </tr>
  {table_rows}
</table>

<h2>Operational Interpretation</h2>
<div class="box-info">
  At a cost-optimal threshold of <strong>{best_test_row['threshold']:.2f}</strong>,
  for every <strong>10,000 transactions</strong> this model flags approximately
  <strong>{flagged_10k:.0f}</strong> as fraud, of which approximately
  <strong>{genuine_10k:.0f}</strong> are genuine fraud and
  <strong>{false_alarm_10k:.0f}</strong> are false alarms.
  The estimated investigation and friction cost of false alarms is
  <strong>${fp_cost_10k:,.0f}</strong>, offset against approximately
  <strong>${fn_cost_10k:,.0f}</strong> in fraud prevented,
  for a net saving of approximately <strong>${net_saving_10k:,.0f}</strong>
  compared to no fraud detection.
</div>

<h2>Cost-Optimal vs F1-Optimal Threshold</h2>
<p>
  For <strong>{best_name}</strong>, the F1-optimal threshold is
  <strong>{f1_thresh_best:.2f}</strong> and the cost-optimal threshold is
  <strong>{cost_thresh_best:.2f}</strong>.
  Choosing the F1-optimal threshold instead of the cost-optimal threshold
  would increase total business cost by approximately
  <strong>${dollar_diff_best:,.2f}</strong> on the validation set.
</p>
<p>
  The F1 metric treats false positives and false negatives symmetrically --
  it has no awareness of the dollar asymmetry between a $10 investigation cost
  and a $500 fraud loss. The cost-optimal threshold internalises that asymmetry
  directly. A business with reliable estimates of investigation and fraud loss
  costs should use the cost-optimal threshold. F1-optimal is appropriate when
  costs are unknown or a balanced trade-off is preferred on principle.
</p>

<h2>Sensitivity Analysis -- Investigation Cost Scenarios</h2>
<p>
  How does the optimal threshold change as investigation costs rise?
  Higher FP costs make false alarms more expensive, pushing the optimal
  threshold <em>up</em> (flag fewer transactions, tolerate more missed fraud).
</p>
<table>
  <tr>
    <th style="text-align:left">Scenario</th>
    <th>FP Base Cost</th>
    <th>Optimal Threshold</th>
    <th>Total Cost (validation)</th>
  </tr>
  {sens_rows}
</table>

<h2>Limitations</h2>
<div class="box-risk">
<ul>
  <li><strong>Anonymized dataset:</strong> V1-V28 are PCA-transformed and cannot be interpreted
      as business features. Transaction amounts may not reflect real currency values.</li>
  <li><strong>Investigation cost is an estimate:</strong> $10 base cost is an industry
      estimate, not specific to any bank, card scheme, or internal workflow.</li>
  <li><strong>Concept drift:</strong> This dataset is from 2013. Fraud patterns evolve;
      model performance may degrade on future data without periodic retraining.</li>
  <li><strong>SMOTE generates synthetic samples:</strong> Oversampled minority examples are
      interpolated, not real transactions. Models may be partially fit to a synthetic
      distribution rather than real fraud patterns.</li>
  <li><strong>Random split, not time-ordered:</strong> Production models should be trained
      on past data and evaluated on future data to avoid temporal leakage.</li>
  <li><strong>Worst-case FN assumption:</strong> We assume 100% of the fraud amount is lost.
      In practice, chargebacks, insurance, or partial recovery reduce actual loss.</li>
</ul>
</div>

</body>
</html>"""

with open(os.path.join(SCRIPT_DIR, 'cost_report.html'), 'w') as f:
    f.write(report_html)
print("  Saved -> phase2_cost_analysis/cost_report.html")

# ============================================================
# WRITTEN SUMMARY -> phase2_cost_analysis/results_summary.md
# ============================================================
md_table_rows = []
for _, row in test_df.iterrows():
    champ = ' ****' if (row['model'] == best_name and row['threshold_type'] == 'Cost-Optimal') else ''
    md_table_rows.append(
        f"| {row['model']}{champ} | {row['threshold_type']} | {row['threshold']:.2f}"
        f" | ${row['total_cost']:,.0f} | ${row['fp_cost']:,.0f} | ${row['fn_cost']:,.0f}"
        f" | {row['recall']:.4f} | {row['f1']:.4f} | {row['auc']:.4f} |"
    )

sens_md_rows = '\n'.join(
    f"| {lbl} | ${d['fp_cost']} | {d['opt_threshold']:.2f} | ${d['total_cost']:,.2f} |"
    for lbl, d in sensitivity_results.items()
)

summary_md = f"""# Cost-Sensitive Fraud Detection -- Results Summary

## Assumptions

> **These are illustrative figures for a portfolio analysis on an anonymized dataset.
> Real investigation and fraud costs vary by institution.**

| Item | Value | Source |
|---|---|---|
| Base FP cost | $10 per transaction | Industry estimate ($7-10) |
| Friction penalty | 1% of Amount | Customer friction proxy |
| Full FP cost | $10 + (0.01 x Amount) | -- |
| FN cost | Full Amount lost | Worst-case assumption |

---

## Model Comparison -- Test Set

| Model | Threshold Type | Threshold | Total Cost | FP Cost | FN Cost | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(md_table_rows)}

---

## Operational Interpretation

At a cost-optimal threshold of **{best_test_row['threshold']:.2f}**, for every 10,000 transactions
this model flags approximately **{flagged_10k:.0f}** as fraud, of which approximately
**{genuine_10k:.0f}** are genuine fraud and **{false_alarm_10k:.0f}** are false alarms.
The estimated investigation and friction cost of false alarms is **${fp_cost_10k:,.0f}**,
offset against approximately **${fn_cost_10k:,.0f}** in fraud prevented,
for a net saving of approximately **${net_saving_10k:,.0f}** compared to no fraud detection.

---

## Cost-Optimal vs F1-Optimal Threshold

For **{best_name}**, the cost-optimal threshold is **{cost_thresh_best:.2f}** and
the F1-optimal threshold is **{f1_thresh_best:.2f}**.

- The **F1-optimal** threshold treats FP and FN symmetrically -- no dollar weighting.
- The **cost-optimal** threshold reflects the actual asymmetry: missed fraud typically
  costs far more than a false alarm investigation.
- Operating at the F1-optimal threshold instead of the cost-optimal threshold would
  cost an additional **${dollar_diff_best:,.2f}** on the validation set.

---

## Sensitivity Analysis -- Optimal Threshold by FP Cost Scenario

| Scenario | FP Base Cost | Optimal Threshold | Total Cost (validation) |
|---|---|---|---|
{sens_md_rows}

Higher investigation costs push the optimal threshold **up** -- the model becomes
more conservative, flagging fewer transactions and tolerating more missed fraud.

---

## Limitations

1. **Anonymized dataset**: V1-V28 are PCA-transformed. Amounts may not reflect real currency.
2. **Investigation cost is an estimate**: $10 base cost is industry-level, not bank-specific.
3. **Concept drift**: Dataset is from 2013. Performance may degrade on future fraud patterns.
4. **SMOTE is synthetic**: Oversampled minority class may not fully reflect real fraud.
5. **Random split**: Production evaluation should always be time-ordered.
6. **Worst-case FN**: 100% loss assumed. Chargebacks/insurance reduce actual FN cost.
"""

with open(os.path.join(SCRIPT_DIR, 'results_summary.md'), 'w') as f:
    f.write(summary_md)
print("  Saved -> phase2_cost_analysis/results_summary.md")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
print("  phase2_cost_analysis/cost_fraud_analysis.py  -- full reproducible code")
print("  phase2_cost_analysis/cost_results.html       -- interactive Plotly charts")
print("  phase2_cost_analysis/cost_report.html        -- stakeholder report")
print("  phase2_cost_analysis/results_summary.md      -- written summary")
print("  models/                 -- saved models + metadata JSON")
print("=" * 65)
