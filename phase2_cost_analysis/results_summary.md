# Cost-Sensitive Fraud Detection -- Results Summary

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
| Logistic Regression | Cost-Optimal | 0.99 | $4,562 | $326 | $4,236 | 0.8378 | 0.7470 | 0.9646 |
| Logistic Regression | 0.5 Default | 0.50 | $14,103 | $10,779 | $3,325 | 0.8784 | 0.1162 | 0.9646 |
| Random Forest **** | Cost-Optimal | 0.30 | $4,389 | $153 | $4,236 | 0.8514 | 0.8344 | 0.9635 |
| Random Forest | 0.5 Default | 0.50 | $4,899 | $91 | $4,808 | 0.7973 | 0.8310 | 0.9635 |
| XGBoost | Cost-Optimal | 0.75 | $5,396 | $908 | $4,488 | 0.8108 | 0.5505 | 0.9722 |
| XGBoost | 0.5 Default | 0.50 | $6,164 | $1,677 | $4,487 | 0.8243 | 0.4207 | 0.9722 |

---

## Operational Interpretation

At a cost-optimal threshold of **0.30**, for every 10,000 transactions
this model flags approximately **18** as fraud, of which approximately
**15** are genuine fraud and **3** are false alarms.
The estimated investigation and friction cost of false alarms is **$36**,
offset against approximately **$992** in fraud prevented,
for a net saving of approximately **$958** compared to no fraud detection.

---

## Cost-Optimal vs F1-Optimal Threshold

For **Random Forest**, the cost-optimal threshold is **0.30** and
the F1-optimal threshold is **0.69**.

- The **F1-optimal** threshold treats FP and FN symmetrically -- no dollar weighting.
- The **cost-optimal** threshold reflects the actual asymmetry: missed fraud typically
  costs far more than a false alarm investigation.
- Operating at the F1-optimal threshold instead of the cost-optimal threshold would
  cost an additional **$2,165.51** on the validation set.

---

## Sensitivity Analysis -- Optimal Threshold by FP Cost Scenario

| Scenario | FP Base Cost | Optimal Threshold | Total Cost (validation) |
|---|---|---|---|
| Low ($5) | $5 | 0.30 | $2,406.07 |
| Base ($10) | $10 | 0.30 | $2,506.07 |
| High ($25) | $25 | 0.50 | $2,696.94 |
| Very High ($50) | $50 | 0.50 | $2,896.94 |

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
