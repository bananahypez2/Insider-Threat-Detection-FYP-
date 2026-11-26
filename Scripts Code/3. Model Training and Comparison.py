"""
Model Training for the prototype
(Contextual Random Forest (Temporal Split) and Contextual Random Forest (Held-out Users Split)
"""

# ------------------
# Imports
# ------------------
import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ---------------------
# Configurations
# ---------------------
BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
FEATURE_TABLE_LOCATION = os.path.join(BASE_FOLDER, "feature_table_final.csv")
OUTPUT_LOCATION = BASE_FOLDER
RESULTS_FOLDER = os.path.join(OUTPUT_LOCATION, "results", "production")
MODELS_FOLDER = os.path.join(OUTPUT_LOCATION, "models")
os.makedirs(RESULTS_FOLDER, exist_ok=True) 
os.makedirs(MODELS_FOLDER,  exist_ok=True)

RANDOM_STATE = 42

print("x" * 65)
print("CERT r4.2 - Model Training for the prototype Contextual Random Forest (Temporal Split) and Contextual Random Forest (Held-out Users Split)")
print("x" * 65)

# -----------------------------
# STEP 1 - Load Feature Table
# -----------------------------
print("\nStep 1 - Loading feature table")

df = pd.read_csv(FEATURE_TABLE_LOCATION, low_memory=False)
df["date_only"] = pd.to_datetime(df["date_only"])
print(f"  Shape : {df.shape}")
print(f"  Labels: {df['label'].value_counts().to_dict()}")

# -------------------------------
# STEP 2 - Define Feature Groups
# -------------------------------
print("\nStep 2 - Defining feature groups")

# Behavioural Context
BEHAVIOURAL_FEATURES = [
    "logon_count", 
    "logoff_count", 
    "after_hours_logon",
    "weekend_logon", 
    "unique_pcs",
    "usb_connect_count", 
    "usb_after_hours",
    "file_copy_count", 
    "file_copy_after_hours",
    "emails_total", 
    "emails_external",
    "emails_with_attachment", 
    "emails_after_hours",
    "http_count", 
    "suspicious_url_count", 
    "http_after_hours"
]

# Contextual additions - personalised deviations + psychometric + LDAP
CONTEXTUAL_ADDITIONS = [
    "logon_count_deviation", 
    "usb_connect_count_deviation",
    "file_copy_count_deviation", 
    "emails_total_deviation",
    "emails_external_deviation", 
    "http_count_deviation",
    "after_hours_logon_deviation", 
    "suspicious_url_count_deviation",
    "O", 
    "C", 
    "E", 
    "A", 
    "N",
    "role_encoded", 
    "department_encoded", 
    "functional_unit_encoded"
]

ALL_FEATURES = BEHAVIOURAL_FEATURES + CONTEXTUAL_ADDITIONS

missing = [f for f in ALL_FEATURES if f not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected feature columns in feature table: {missing}\n"
        f"Prototype models D and F must both train on the full {len(ALL_FEATURES)}-features\n"
        f"Fix the feature table or update ALL_FEATURES to match, then re-run"
    )

print(f"  All features (contextual) : {len(ALL_FEATURES)}")

# ---------------------------------------------------------------------------------
# Step 3 - Model D Temporal Train/Test Split (Contextual Random Forest Temporal Split)
# ---------------------------------------------------------------------------------
print("\nStep 3 - Temporal train/test split")

SPLIT_DATE = pd.Timestamp("2011-01-01")

train_df = df[df["date_only"] <  SPLIT_DATE].copy()
test_df = df[df["date_only"] >= SPLIT_DATE].copy()

print(f"  Train: {len(train_df):,} rows  "
      f"({train_df['label'].sum():,} malicious = {train_df['label'].mean()*100:.1f}%)")
print(f"  Test : {len(test_df):,} rows  "
      f"({test_df['label'].sum():,} malicious = {test_df['label'].mean()*100:.1f}%)")

y_train = train_df["label"].values.astype(int)
y_test = test_df["label"].values.astype(int)

# ---------------------------------------------------------------------------------------------------
# STEP 3B -  Model F Held-out User Train/Test Split (Contextual Random Forest Held-out User Split)
#            This means model is tested against users that never appeared in the training set before
#----------------------------------------------------------------------------------------------------
print("\nStep 3B - Held-out User Split")

user_labels = df.groupby("user")["label"].max()

# passing a plain numpy array avoids
# "TypeError: only integer scalar arrays can be converted to a scalar index".
train_users, test_users = train_test_split(
    user_labels.index.to_numpy(),
    test_size=0.3,
    stratify=user_labels.values,
    random_state=RANDOM_STATE
)

train_df_user = df[df["user"].isin(train_users)].copy()
test_df_user = df[df["user"].isin(test_users)].copy()

y_train_user = train_df_user["label"].values.astype(int)
y_test_user = test_df_user["label"].values.astype(int)

print(f"  Train users: {len(train_users)}  Test users: {len(test_users)}")
print(f"  Train rows : {len(train_df_user):,}  Test rows : {len(test_df_user):,}")

# --------------------------------------------------------------
# HELPER - train, evaluate, and saving function for each model
# --------------------------------------------------------------
def train_and_save(name, label, features, train_df, test_df, y_train, y_test):

    print(f"\n  [{label}] {name}")
    print(f"         Features: {len(features)} columns")

    X_train = train_df[features].values.astype(float)
    X_test = test_df[features].values.astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_scaled_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_scaled, y_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_scaled_SMOTE, y_train_SMOTE)

    y_pred = model.predict(X_test_scaled)
    y_predicted_prob = model.predict_proba(X_test_scaled)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_predicted_prob)

    print(f"         F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  AUC={auc:.4f}")

    # Count the four possible outcomes on the test set, so the saved report
    # shows actual day counts and not only the rates. The rates above are
    # worked out from these same four numbers.
    y_actual = np.array(y_test)
    true_positive = int(np.sum((y_actual == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_actual == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_actual == 1) & (y_pred == 0)))
    true_negative = int(np.sum((y_actual == 0) & (y_pred == 0)))

    print(f"         TP={true_positive}  FN={false_negative}  "
          f"FP={false_positive}  TN={true_negative}")

    # Save model artifacts - these are what the Flask prototype loads
    joblib.dump(model,    os.path.join(MODELS_FOLDER, f"model_{label}.pkl"))
    joblib.dump(scaler,   os.path.join(MODELS_FOLDER, f"scaler_{label}.pkl"))
    joblib.dump(features, os.path.join(MODELS_FOLDER, f"features_{label}.pkl"))
    print(f"         Saved model/scaler/features as '{label}' -> {MODELS_FOLDER}")

    # Minimal report - enough to confirm the deployed model's performance
    # without the full ablation comparison machinery
    report = classification_report(
        y_test, y_pred, target_names=["Normal", "Malicious"], zero_division=0
    )
    report_path = os.path.join(RESULTS_FOLDER, f"report_{label}.txt")
    with open(report_path, "w") as fh:
        fh.write(f"{name}\n{'='*60}\n")
        fh.write(f"Features : {len(features)}\n\n")
        fh.write(report)
        fh.write("\nCounts on the test set\n")
        fh.write(f"  Malicious days caught (TP)  : {true_positive}\n")
        fh.write(f"  Malicious days missed (FN)  : {false_negative}\n")
        fh.write(f"  False alarms (FP)           : {false_positive}\n")
        fh.write(f"  Normal days left alone (TN) : {true_negative}\n")
    print(f" Saved: {report_path}")

    return {
        "label": label, "name": name, "n_features": len(features),
        "f1": round(f1, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "auc_roc": round(auc, 4),
    }

# -------------------------------------------------------------------
# STEP 4 - Train Model D - Contextual Random Forest Temporal Split
# -------------------------------------------------------------------
print("\nStep 4: Training Model D - Contextual Random Forest Temporal Split")

result_d = train_and_save(
    name="Model D - Contextual Random Forest Temporal Split",
    label="D",
    features=ALL_FEATURES,
    train_df=train_df, test_df=test_df,
    y_train=y_train, y_test=y_test
)

# ----------------------------------------------------------------------
# STEP 5 - Train Model F - Contextual Random Forest Held-out User Split
# ----------------------------------------------------------------------
print("\nStep 5: Training Model F - Contextual Random Forest Held-out User Split")

result_f = train_and_save(
    name="Model F - Contextual Random Forest Held-out User Split",
    label="F",
    features=ALL_FEATURES,
    train_df=train_df_user, test_df=test_df_user,
    y_train=y_train_user, y_test=y_test_user
)


# --------------------------------
# STEP 6 - Save Metrics Summary
# --------------------------------
print("\nStep 6 Saving Metrics Summary for Models")

metrics_df = pd.DataFrame([result_d, result_f])
metrics_path = os.path.join(RESULTS_FOLDER, "production_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"  Saved: {metrics_path}")

print("\n" + "x" * 65)
print("PRODUCTION TRAINING COMPLETE")
print("x" * 65)
print(f"""
Model artifacts saved to: {MODELS_FOLDER}
  model_D.pkl, scaler_D.pkl, features_D.pkl  - main detector (F1={result_d['f1']}, AUC={result_d['auc_roc']})
  model_F.pkl, scaler_F.pkl, features_F.pkl  - explainability model (F1={result_f['f1']}, AUC={result_f['auc_roc']})

Reports saved to: {RESULTS_FOLDER}
  report_D.txt, report_F.txt, production_metrics.csv

""")
