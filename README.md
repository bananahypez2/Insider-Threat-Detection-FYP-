# Context-Aware Anomaly Detection for Suspicious User Behaviour in System Logs

Detects suspicious employee behaviour in system logs by learning **each individual's own baseline** rather than applying fixed thresholds, and explains every detection at the level of the single employee-day.

Final Year Project — BSc (Hons) Computer Science (Cyber Security), Asia Pacific University (APU).

---

## The problem

Four limitations motivated this project:

1. Manual log analysis does not scale to the volumes organisations now generate.
2. Rule-based detection is rigid and produces high false-positive rates.
3. Without context, normal and abnormal behaviour are hard to separate — an employee who routinely works late looks identical to one who never has.
4. Most detection systems are black boxes: they flag a row without saying why.

"Context-aware" here means three things combined:

- **Personalised deviation** — today's activity measured against that employee's own rolling 30-record baseline
- **Organisational context** — role, department, functional unit
- **Psychometric context** — Big Five (OCEAN) trait scores

---

## Results

Random Forest (200 trees, `max_depth=20`) over 32 features, SMOTE for class imbalance, evaluated on a temporal split — train on 2010, test on 2011.

**Model D, the deployed detector.** Measured on 84,029 test employee-days containing 244 malicious days, a prevalence of 0.3%:

```
Precision   0.93
Recall      0.68
F1          0.78

TP     165        FN      79
FP      13        TN  83,772
```

Thirteen false positives across 84,029 days.

**Model F, the generalisation check.** Trained and tested on disjoint sets of employees, so the 300 test employees were never seen during training: F1 0.78, precision 0.81, recall 0.76. Performance holds on unseen people, which indicates the model learned the behavioural pattern rather than the individuals.

### Does context actually help?

An ablation holding the model and the split constant, varying only the feature set:

```
C — behavioural counts only (16 features)
    precision 0.6821   recall 0.5451   F1 0.6059

D — plus contextual features (32 features)
    precision 0.9270   recall 0.6762   F1 0.7820
```

Adding context raised **F1 by 29%** and **precision by 36%**. This is the project's central claim, and it is measured rather than asserted.

### Against a rule-based baseline

Six hand-written rules over the same test period flagged **31,559 of 84,029 days**, or 37.6%, to reach recall 0.8074 — a precision of 0.0062. Model D reaches precision 0.93 with 13 false positives.

The rule-based approach achieves higher recall, but at a false-alarm rate no analyst could triage.

### Context sensitivity

A diagnostic replaced the contextual features on real employee-days while leaving behaviour untouched:

- Swapping **job only** (role, department, unit) moved the risk score by **0.4499** on average
- Swapping **job and personality together** moved it by **0.6461**
- Giving the most damaging organisational identity to **innocent** days pushed **0 of 10** above the decision threshold

Context genuinely influences the score, but it cannot by itself make an innocent day look malicious. The model requires suspicious behaviour as well as suspicious context.

---

## Explainability

Every prediction is decomposed with **SHAP TreeExplainer**, which is exact for tree models — the base value plus the 32 feature contributions reproduces `predict_proba` to the decimal, with zero error.

The employee detail page shows the top contributing features under plain-language names, a signed contribution chart, and the raw unscaled feature values, so an analyst can see which behaviour drove the score and in which direction.

---

## Architecture

Training is deliberately separated from scoring:

```
OFFLINE (run once)                      ONLINE (the application)
──────────────────────                  ────────────────────────
raw CERT logs                           upload: feature table CSV
   ↓ feature engineering                     or raw log ZIP
   ↓ 32-feature table                           ↓
   ↓ train / evaluate                      preprocess → score
   ↓                                            ↓
   └── model_D.pkl, scaler_D.pkl ──────→   SHAP explanation
       features_D.pkl, label_maps.pkl           ↓
                                           ranked dashboard
```

Three reasons for the split:

- **Speed** — training on 330,452 rows is slow; scoring has to be interactive.
- **Reproducibility** — one fixed model scores every upload, so the same file always returns the same answer.
- **Correctness** — the scaler and label encodings are never refit. Rebuilding encoders on a small upload would map roles to different numbers than training used, and every score would be wrong.

The application accepts two input paths that meet at the same 32-column contract: a prepared feature table, or raw logs it engineers itself. Both produce identical scores for the same employee-day, verified to six decimal places.

---

## Screenshots

![Upload page](App%20Screenshots/1.%20Upload%20Page%20.png)
*Upload page — feature table or raw log ZIP*

![Results dashboard](App%20Screenshots/2.%20Result%20Dashboard.png)
*Results ranked by risk score, with threshold and model selectors*

![Employee detail](App%20Screenshots/9.%20Detail%20Page%201.png)
*Employee detail — SHAP contributions behind one flagged day*

![Risk timeline](App%20Screenshots/4.%20Risk%20Timeline.png)
*Per-employee risk timeline, each point clickable*

Further screenshots — threshold selection, model switching, search and filters, error handling — are in [`App Screenshots/`](App%20Screenshots).

---

## Data

This project uses the **CERT Insider Threat Test Dataset r4.2**: 1,000 employees, 70 insiders, 330,452 employee-days, 966 malicious days.

**The dataset is not included in this repository.** It is Copyright © 2011 ExactData, LLC, All Rights Reserved, provided under agreement #12-00121-EXP-000 between Carnegie Mellon University and ExactData, LLC, and its End User Agreement prohibits redistribution over the internet. Obtain it from Carnegie Mellon's Software Engineering Institute:

<https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099>

To run the pipeline yourself, place the extracted dataset as follows:

```
r4.2/logon.csv, device.csv, file.csv, email.csv, http.csv, LDAP/, psychometric.csv
answers/ground-truth insider files
context_data/
  ├─ LDAP/<snapshot>.csv  earliest LDAP snapshot, used as inference-time context
  └─ psychometric.csv
```

`context_data/` is the fallback the application uses when an uploaded ZIP supplies no organisational context of its own. Without it, roles resolve to `Unknown` and scores shift downward — the application still runs, but the contextual features carry no signal.

---

## Setup

```bash
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

cd "Context-Aware Anomaly Detection System"
python "Context-Aware Anomaly Detection System.py"
```

Then open <http://127.0.0.1:5000>.

Python 3.11. The saved `.pkl` artefacts were produced with scikit-learn 1.9.0 — keep that version pinned, or retrain with `3. Model Training and Comparison.py`.

To rebuild everything from raw logs, run the scripts in order: `1. EDA` → `2. Feature Engineering` → `3. Model Training and Comparison` → `4. Ablation Study` → `5. Rule Based Baseline`.

---

## Repository structure

```
Context-Aware Anomaly Detection System/   the Flask application
├─ Context-Aware Anomaly Detection System.py   routes, scoring, SHAP
├─ Preprocess_Raw_Logs.py                      raw logs → 32-feature table
├─ models/                                     trained artefacts (.pkl)
├─ templates/                                  upload, results, detail, about
└─ static/style.css

Scripts Code/                             the offline pipeline
├─ 1. EDA.py                                   dataset exploration, ground truth
├─ 2. Feature Engineering.py                   builds the 32-feature table
├─ 3. Model Training and Comparison.py         trains Models D and F
├─ 4. Ablation Study.py                        six configurations, A–F
├─ 5. Rule Based Baseline.py                   hand-written rule comparison
├─ D1. Check Context Sensitivity.py            context sensitivity diagnostic
├─ E2. Generate 2026 Demo.py                   synthetic demonstration data
└─ Outputs/                                    console logs for every script

App Screenshots/                          the running system
```

Every reported figure can be traced to a log in `Scripts Code/Outputs/`.

---

## Limitations

Stated plainly, because they bound what the results mean:

- **The dataset is synthetic.** CERT r4.2 models a fictional organisation. Its regularity is almost certainly why AUC reaches 0.9996, and that figure should not be read as real-world performance. Validation on genuine organisational logs is the necessary next step.
- **The suspicious-URL keyword list is tuned to this dataset** and would need rebuilding elsewhere.
- **Hyperparameters were not optimised.** Tree count and depth were fixed, not searched.
- **The 30-record baseline window was never varied.** Forty or fifty records were not tested.
- **Supervised learning only**, so genuinely novel attack patterns absent from training can be missed. An Isolation Forest was evaluated as a semi-supervised alternative and performed far worse, at F1 0.0089.
- **No range validation on inputs.** Types are checked; plausibility is not. Absurd values are scored rather than rejected.
- **No locational context**, as the dataset provides none.

---

## Licence

Source code: MIT — see [LICENSE](LICENSE).
