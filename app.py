# -*- coding: utf-8 -*-
"""
Streamlit dashboard: Credit Card Fraud Detection -- Business Recommendations
Run: streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Detection -- Business Dashboard",
    layout="wide",
)

# ============================================================
# DATA LOAD
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'models', 'dashboard_data.json')

@st.cache_data
def load_data():
    with open(DATA_PATH, encoding='utf-8') as f:
        return json.load(f)

data  = load_data()
sweep = data['validation_sweep']
test  = data['test_results']
meta  = data['metadata']

MODEL_NAMES  = list(sweep.keys())
BASE_FP_COST = meta['base_fp_cost']   # 10

MODEL_COLORS = {
    'Logistic Regression': '#636EFA',
    'Random Forest':       '#00CC96',
    'XGBoost':             '#EF553B',
}

# ============================================================
# HELPERS
# ============================================================
def sweep_arrays(model_name):
    """Return numpy arrays for a model's validation sweep."""
    m = sweep[model_name]
    return (
        np.array(m['threshold']),
        np.array(m['total_cost']),
        np.array(m['fp_cost']),
        np.array(m['fn_cost']),
        np.array(m['n_fp']),
        np.array(m['n_fn']),
        np.array(m['recall']),
    )


@st.cache_data
def precompute_sensitivity_curves(sweep_json_str, base_fp_cost):
    """
    For each model, compute minimum total cost at each FP base cost $1-$100.
    Cached so this runs once at load, not on every slider move.

    Math: fp_cost stored at base=$10 decomposes as:
        fp_cost = base * n_fp + friction
        friction = fp_cost - base * n_fp   (= 0.01 * sum of FP amounts, per threshold)

    For any new base:
        new_fp_cost = new_base * n_fp + friction
        new_total   = new_fp_cost + fn_cost
    """
    sweep_data = json.loads(sweep_json_str)
    fp_range   = np.arange(1, 101)
    result     = {}
    for model_name, m in sweep_data.items():
        t_arr    = np.array(m['threshold'])
        fp_c     = np.array(m['fp_cost'])
        fn_c     = np.array(m['fn_cost'])
        n_fp     = np.array(m['n_fp'])
        friction = fp_c - base_fp_cost * n_fp

        min_costs      = []
        opt_thresholds = []
        for base in fp_range:
            new_total = base * n_fp + friction + fn_c
            best_idx  = int(np.argmin(new_total))
            min_costs.append(float(new_total[best_idx]))
            opt_thresholds.append(float(t_arr[best_idx]))

        result[model_name] = {
            'fp_range':       fp_range.tolist(),
            'min_costs':      min_costs,
            'opt_thresholds': opt_thresholds,
        }
    return result


sens_curves = precompute_sensitivity_curves(json.dumps(sweep), BASE_FP_COST)

# ============================================================
# SIDEBAR -- all interactive controls
# ============================================================
st.sidebar.header("Controls")
st.sidebar.markdown("Settings that affect the Threshold Explorer, Sensitivity Analysis, and Operational Reality sections.")

selected_model = st.sidebar.selectbox(
    "Model",
    options=MODEL_NAMES,
    index=MODEL_NAMES.index('Random Forest'),
)

cost_opt_t = test[selected_model]['cost_optimal']['threshold']

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(round(cost_opt_t, 2)),
    step=0.01,
    format="%.2f",
)
st.sidebar.caption(f"Cost-optimal for {selected_model}: {cost_opt_t:.2f}")

st.sidebar.divider()

fp_base = st.sidebar.slider(
    "FP Investigation Base Cost ($ per flagged transaction)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    format="$%d",
)
st.sidebar.caption("Affects Sensitivity Analysis only. Base cost of investigating a flagged transaction.")

# ============================================================
# SECTION 1 -- Header
# ============================================================
st.title("Credit Card Fraud Detection")
st.markdown(
    "**Dataset:** 284,807 transactions | 492 confirmed fraud cases | 0.17% fraud rate "
    "(Kaggle / European cardholders, September 2013)"
)
st.info(
    "**Key finding:** Threshold selection matters more than model selection. "
    "A poorly-chosen threshold can triple the business cost of an otherwise "
    "competitive model."
)
st.divider()

# ============================================================
# SECTION 2 -- Bottom Line
# ============================================================
st.header("Bottom Line")

rf_opt  = test['Random Forest']['cost_optimal']
lr_def  = test['Logistic Regression']['default_0_50']
savings = lr_def['total_cost'] - rf_opt['total_cost']
sav_pct = savings / lr_def['total_cost'] * 100

c1, c2, c3 = st.columns(3)
c1.metric(
    label="RF -- Cost-Optimal (threshold 0.30)",
    value=f"${rf_opt['total_cost']:,.0f}",
    help="Random Forest with threshold tuned to minimise total business cost on the validation set",
)
c2.metric(
    label="LR -- Default Threshold (0.50)",
    value=f"${lr_def['total_cost']:,.0f}",
    help="Logistic Regression at the standard out-of-the-box threshold",
)
c3.metric(
    label="Savings from Threshold Optimisation",
    value=f"${savings:,.0f}",
    delta=f"-{sav_pct:.0f}% vs LR default",
    delta_color="inverse",
)
st.caption(
    "Costs are illustrative -- dataset currency is unknown. "
    "All figures are relative comparisons, not absolute dollar amounts. "
    f"Test set: {meta['n_test']:,} transactions."
)
st.divider()

# ============================================================
# SECTION 3 -- Threshold Explorer
# ============================================================
st.header("Threshold Explorer")
st.markdown(
    "Use this chart to confirm the cost-optimal threshold for the selected model. "
    "The recommended configuration, highlighted in the table below, was chosen because it minimizes total financial exposure. "
    "Select a model and threshold in the **Controls** panel to compare scenarios."
)

t_arr, total_c, fp_c, fn_c, n_fp_arr, n_fn_arr, recall_arr = sweep_arrays(selected_model)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_arr, y=total_c,
    name='Total Cost',
    line=dict(color=MODEL_COLORS[selected_model], width=2.5),
))
fig.add_trace(go.Scatter(
    x=t_arr, y=fp_c,
    name='FP Cost (investigation + friction)',
    line=dict(color='#636EFA', width=1.5, dash='dot'),
    opacity=0.75,
))
fig.add_trace(go.Scatter(
    x=t_arr, y=fn_c,
    name='FN Cost (missed fraud)',
    line=dict(color='#EF553B', width=1.5, dash='dot'),
    opacity=0.75,
))
fig.add_vline(
    x=cost_opt_t,
    line_dash='dash', line_color='green', line_width=1.5,
    annotation_text=f"Cost-optimal: {cost_opt_t:.2f}",
    annotation_position="top right",
    annotation_font_color="green",
)
if abs(threshold - cost_opt_t) > 0.015:
    fig.add_vline(
        x=threshold,
        line_dash='dash', line_color='grey', line_width=1.5,
        annotation_text=f"Selected: {threshold:.2f}",
        annotation_position="top left",
    )
fig.update_layout(
    xaxis_title='Threshold',
    yaxis_title='Cost ($)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    height=420,
    margin=dict(t=60, b=40),
)
st.plotly_chart(fig, width='stretch')

# Live metrics at selected threshold
idx      = int(np.argmin(np.abs(t_arr - threshold)))
live_tot = float(total_c[idx])
live_fp  = float(fp_c[idx])
live_fn  = float(fn_c[idx])
live_rec = float(recall_arr[idx])
live_nfp = int(n_fp_arr[idx])
live_nfn = int(n_fn_arr[idx])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Cost", f"${live_tot:,.0f}")
m2.metric("FP Cost",    f"${live_fp:,.0f}",  help=f"{live_nfp} false alarms flagged")
m3.metric("FN Cost",    f"${live_fn:,.0f}",  help=f"{live_nfn} fraud cases missed")
m4.metric("Recall",     f"{live_rec:.1%}",   help="Fraction of actual fraud caught")

st.caption(
    f"Costs computed on validation set ({meta['n_val']:,} transactions). "
    "Official test set results in the table below."
)
st.divider()

# ============================================================
# SECTION 4 -- Model Comparison Table
# ============================================================
st.header("Model Comparison -- Test Set")
st.markdown(
    "All six combinations (3 models x default / cost-optimal threshold). "
    "The recommended configuration is highlighted."
)

rows = []
for model_name in MODEL_NAMES:
    for ttype, tlabel in [('default_0_50', 'Default (0.50)'), ('cost_optimal', 'Cost-Optimal')]:
        r       = test[model_name][ttype]
        is_best = (model_name == 'Random Forest' and ttype == 'cost_optimal')
        rows.append({
            'Model':      ('* ' if is_best else '  ') + model_name,
            'Type':       tlabel,
            'Threshold':  f"{r['threshold']:.2f}",
            'Total Cost': f"${r['total_cost']:,.0f}",
            'FP Cost':    f"${r['fp_cost']:,.0f}",
            'FN Cost':    f"${r['fn_cost']:,.0f}",
            'FP Count':   r['n_fp'],
            'FN Count':   r['n_fn'],
            'Recall':     f"{r['recall']:.3f}",
            'F1':         f"{r['f1']:.3f}",
            '_best':      is_best,
        })

table_df   = pd.DataFrame(rows)
best_flags = table_df.pop('_best')


def highlight_best(row):
    if best_flags.iloc[row.name]:
        return ['background-color: rgba(0, 204, 150, 0.18); font-weight: bold'] * len(row)
    return [''] * len(row)


st.dataframe(
    table_df.style.apply(highlight_best, axis=1),
    width='stretch',
    hide_index=True,
)
st.caption("* Recommended configuration.")
st.divider()

# ============================================================
# SECTION 5 -- Sensitivity Analysis
# ============================================================
st.header("Sensitivity Analysis")
st.markdown(
    "A key concern before deploying any model is whether the recommendation depends heavily on cost assumptions we cannot verify precisely. "
    "It doesn't. Random Forest remains the preferred choice regardless of investigation cost. "
    "Adjust **FP Investigation Base Cost** in the Controls panel to confirm."
)

s_cols = st.columns(3)
for i, model_name in enumerate(MODEL_NAMES):
    base_idx = fp_base - 1   # fp_range starts at $1, list index at 0
    opt_cost = sens_curves[model_name]['min_costs'][base_idx]
    opt_t    = sens_curves[model_name]['opt_thresholds'][base_idx]
    s_cols[i].metric(label=model_name, value=f"${opt_cost:,.0f}")
    s_cols[i].caption(f"Optimal threshold: {opt_t:.2f}")

fig_s = go.Figure()
for model_name in MODEL_NAMES:
    fig_s.add_trace(go.Scatter(
        x=sens_curves[model_name]['fp_range'],
        y=sens_curves[model_name]['min_costs'],
        name=model_name,
        line=dict(color=MODEL_COLORS[model_name], width=2),
    ))
fig_s.add_vline(
    x=fp_base,
    line_dash='dash', line_color='grey', line_width=1,
    annotation_text=f"${fp_base}",
    annotation_position="top right",
)
fig_s.update_layout(
    xaxis_title='FP Investigation Base Cost ($ per transaction)',
    yaxis_title='Minimum Achievable Total Cost ($)',
    legend_title='Model',
    height=380,
    margin=dict(t=40, b=40),
)
st.plotly_chart(fig_s, width='stretch')
st.caption(
    "Random Forest delivers the lowest total cost across the full $1-$100 investigation cost range. "
    "The recommendation is robust: the optimal threshold shifts only when investigation costs exceed approximately $25, "
    "well outside the range of realistic operational assumptions."
)
st.divider()

# ============================================================
# SECTION 6 -- Operational Reality
# ============================================================
st.header("Operational Reality")
st.markdown(
    f"Using **{selected_model}** at threshold **{threshold:.2f}**, "
    "scaled to 1,000,000 transactions:"
)

n_val                 = meta['n_val']
n_fraud_val           = meta['n_fraud_val']
total_fraud_amt_val   = meta['total_fraud_amount_val']
scale_1m              = 1_000_000 / n_val

tp_count        = live_rec * n_fraud_val
flagged_1m      = (tp_count + live_nfp) * scale_1m
genuine_1m      = tp_count * scale_1m
false_alarm_1m  = live_nfp * scale_1m
fp_cost_1m      = live_fp * scale_1m
fn_cost_1m      = live_fn * scale_1m
no_det_cost_1m  = total_fraud_amt_val * scale_1m
fraud_prevented = no_det_cost_1m - fn_cost_1m
net_saving_1m   = no_det_cost_1m - live_tot * scale_1m

o1, o2, o3 = st.columns(3)
o1.metric("Transactions Flagged",  f"~{flagged_1m:,.0f}")
o2.metric("Genuine Fraud Caught",  f"~{genuine_1m:,.0f}")
o3.metric("False Alarms",          f"~{false_alarm_1m:,.0f}")

o4, o5, o6 = st.columns(3)
o4.metric("False Alarm Cost",           f"~${fp_cost_1m:,.0f}")
o5.metric("Fraud Loss Prevented",       f"~${fraud_prevented:,.0f}")
o6.metric("Net Saving vs No Detection", f"~${net_saving_1m:,.0f}")

st.caption(
    f"Estimates scaled from validation set ({n_val:,} transactions). "
    "Fraud loss prevented = total fraud amount minus undetected losses at this threshold."
)
st.divider()

# ============================================================
# SECTION 7 -- Limitations
# ============================================================
with st.expander("Limitations", expanded=True):
    st.markdown("""
**Transaction independence.** The model treats every transaction independently.
Real fraud often follows sequential patterns -- card testing uses small transactions
to verify a stolen card before executing larger fraud. The PCA anonymization eliminates
card identifiers needed to construct velocity features. Missing a \$2 probe transaction
is not a \$2 loss but potentially the sum of all subsequent fraud it enables.

**Anonymized amounts.** Dataset currency is unknown. All cost figures are illustrative
relative comparisons, not absolute dollar amounts.

**SMOTE synthetic samples.** Synthetic minority oversampling interpolates between real
fraud cases and may not reflect rare or novel fraud patterns.

**No temporal validation.** Data was split randomly rather than by time. Production
models should be trained on past data and evaluated on future data. Fraud patterns
drift as adversaries adapt; this model would degrade without retraining.

**Recovery rate not modeled.** Banks recover some fraud losses via chargebacks.
False negative cost as full transaction amount is a worst-case assumption.
""")
