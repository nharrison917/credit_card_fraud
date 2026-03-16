# Time & Amount Patterns in Fraudulent Transactions

## Dataset Context

The dataset covers two consecutive days of European card transactions (September 2013),
with `Time` recorded as seconds elapsed since the first transaction in the dataset.
`Amount` is the transaction value in euros. Both are among the few unmasked features —
the rest (V1–V28) are PCA-transformed and uninterpretable directly.

---

## Time Patterns

### The overnight anomaly

Transaction volume drops sharply between roughly midnight and 6am (hours 0–6 and 24–30),
falling from ~8,000–9,000 per hour during the day to ~1,000–2,000. Fraud does not
drop proportionally. The result is a dramatic spike in fraud *rate* during those windows:

```
Hour   Volume   Fraud   Fraud Rate
   2    1,576      21     1.332%     ← 8× baseline
   3    1,821      13     0.714%
   4    1,082       6     0.555%
   5    1,681      11     0.654%
   7    3,368      23     0.683%
  11    8,517      43     0.505%     ← high volume, still elevated
  26    1,752      36     2.055%     ← highest rate in the dataset (12× baseline)
  28    1,127      17     1.508%
```

The overall fraud rate is 0.173%. Hour 26 (2am on night 2) reaches 2.055% — roughly
**12× the baseline**. Hours 2–5 on night 1 mirror this pattern almost exactly.

### Why this happens

This is a well-documented fraud behaviour pattern. Fraudsters — particularly those
running automated card-testing operations — prefer low-activity windows for two reasons:

1. **Lower detection signal.** Real-time fraud scoring systems often rely partly on
   population-level anomaly detection. When transaction volume is low, an unusual
   cluster of similar transactions (same BIN, similar amount, short time window) is
   harder to distinguish from random noise.

2. **Slower human response.** Fraud operations teams are typically smaller overnight.
   Alert queues build up and are reviewed later, giving attackers more time before
   a card is blocked.

The pattern repeating on both nights, and at nearly identical hours, strongly suggests
systematic automated activity rather than opportunistic human fraud.

### Day 1 vs Day 2

```
Day 1 (hours  0–23):  281 fraud / 144,786 total = 0.1941%
Day 2 (hours 24–47):  211 fraud / 140,021 total = 0.1507%
```

Fraud is ~29% more prevalent on day 1 by rate, and higher in absolute count. One
plausible explanation: some stolen cards were detected and blocked during day 1,
reducing the pool available for day 2. This is consistent with fraud operations that
front-load activity before cards are cancelled.

### Modelling implication

Because the dataset was split randomly (not chronologically), this temporal structure
leaks into both train and test sets. A production model would be trained on past data
and scored on future data — it would never see day 2 fraud patterns while training on
day 1. The overnight rate spike suggests `Time` (or a derived hour-of-day feature)
would have real predictive power in a properly time-ordered deployment.

---

## Amount Patterns

### Distribution summary

```
               Legitimate       Fraud
Count           284,315           492
Mean             $88.29        $122.21
Median           $22.00          $9.25
75th pct         $77.05        $105.89
Max          $25,691.16      $2,125.87
```

The median fraud amount ($9.25) is less than half the median legitimate amount ($22.00),
despite the mean being higher for fraud. This is the signature of a bimodal fraud
distribution: a large cluster of very small probe transactions pulling the median down,
and a thinner tail of larger-value transactions pulling the mean up.

No fraud transaction exceeds $2,125.87. The maximum legitimate transaction is
$25,691.16. **No fraud occurs above $5,000.**

### The sub-$1 cluster

```
Bucket     Fraud Rate    Share of all fraud
$0–$1        0.594%           36.8%
$1–$5        0.106%            8.3%
$5–$10       0.086%            5.5%
```

Over **a third of all fraud transactions are under $1.** The fraud rate in this
bucket (0.594%) is 3.4× the overall baseline — and the absolute count (181 cases) is
the largest of any bucket. Zero-amount transactions are the most extreme case:

```
$0.00 transactions:  27 fraud (5.5% of all fraud)  vs  1,798 legit (0.63% of legit)
```

A $0.00 transaction is **8.7× more likely to be fraud than a legitimate transaction.**

This is the "card probing" pattern. When a stolen card number is obtained (e.g. from
a data breach or skimmer), the attacker needs to verify it is live and has not yet
been blocked. A $0.00 or $0.01 authorization request is the minimal-risk way to do
this — it produces no visible charge on the cardholder's statement but confirms the
card is valid. Successful probes are typically followed by larger cash-out attempts,
which is why the $500–$1,000 bucket also shows an elevated fraud rate (0.419%).

### The most common fraud amounts

```
Rank  Amount   Fraud Count   Notes
   1   $1.00       113       Classic probe / test amount
   2   $0.00        27       Zero-auth probe
   3  $99.99        27       Just-below-round, scripted tool default
   4   $0.76        17       Appears in both fraud and legit top-20
   5   $0.77        10
```

`$1.00` appears 113 times in fraud — by far the most common single value. For
comparison, it appears 13,575 times in legitimate transactions, but legitimate
transactions number 578× more, so the *per-transaction* rate of $1.00 being fraud
is still elevated. The presence of `$99.99` (27 fraud cases) alongside `$1.00` is
consistent with automated fraud tools that cycle through a small set of hardcoded
test amounts.

### Round numbers and cents endings

```
                         Fraud    Legit
Round dollar (no cents)  35.4%    24.9%
Ends in .99               6.3%     9.5%
Ends in .98               n/a      ~3.6%
```

Legitimate retail transactions are dominated by psychological pricing — amounts ending
in `.99`, `.98`, `.95`, `.49`. These endings reflect actual merchant pricing decisions.
Fraud tools do not replicate this distribution. They tend to use round numbers (35.4%
of fraud vs 24.9% of legit) or hardcoded probe values, producing a cents distribution
that is spiked at `.00` and otherwise fairly flat.

The *lower* rate of `.99`-ending amounts in fraud (6.3% vs 9.5%) is the inverse of
the same signal: fraud transactions systematically lack the retail pricing fingerprint.

### No large-value fraud

The complete absence of fraud above $5,000 is notable. Potential explanations:

- **Hard card limits:** Stolen cards may predominantly be lower-limit consumer cards
- **Velocity controls:** Large transactions trigger immediate manual review, making
  them impractical for automated fraud
- **Strategy:** The fraud observed here may be card-testing / small-probe operations
  rather than high-value cash-out fraud, which would appear on a different channel
  (wire transfer, crypto exchange, etc.)

### Modelling implication

`Amount` is one of the two unmasked features and carries real signal — particularly
the sub-$1 region and the round-number pattern. However, because it is skewed
(max legit is $25k vs $2.1k for fraud), a raw `Amount` feature is dominated by its
scale. The main analysis applies `StandardScaler` to both `Amount` and `Time`, which
normalises this. In a feature-engineering context, derived features like
`log1p(Amount)`, `is_probe` (Amount < 1), `is_round` (Amount % 1 == 0), and
`hour_of_day` would likely improve model performance meaningfully — none of these
were used in the current analysis.

---

## Summary

| Feature | Signal | Mechanism |
|---|---|---|
| Hour 2–5 (both nights) | 4–12× elevated fraud rate | Automated attacks during low-monitoring windows |
| Hour 26 | Highest fraud rate (2.05%) | Same pattern, more concentrated on night 2 |
| Amount < $1 | 37% of all fraud | Card probing to verify stolen credentials |
| Amount = $0 | 8.7× fraud rate vs legit | Zero-auth probe, no cardholder-visible charge |
| Round dollar amounts | 35% fraud vs 25% legit | Scripted tools lack retail pricing distributions |
| .99 endings | Under-represented in fraud | Same cause — fraud tools don't mimic retail pricing |
| Amount > $5,000 | Zero fraud cases | Velocity/limit controls or different fraud channel |

Both features exhibit patterns consistent with **systematic, automated fraud
operations** rather than individual opportunistic misuse — the temporal clustering,
the probe-amount signatures, and the scripted-amount fingerprints all point to
organised activity using tooling that has characteristic behavioural tells.
