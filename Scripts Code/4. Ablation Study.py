# 4. Ablation Study.py
# Runs the six ablation configurations (A-F) on the CERT r4.2 feature table
# and repeats the held-out-user pair across five seeds.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, classification_report)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# PATHS AND SETTINGS
# -------------------------------------------------------------
BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
FEATURE_TABLE_LOCATION = os.path.join(BASE_FOLDER, "feature_table_final.csv")
OUTPUT_LOCATION = BASE_FOLDER
ABLATION_FOLDER = os.path.join(OUTPUT_LOCATION, "results", "ablation")
os.makedirs(ABLATION_FOLDER, exist_ok=True)

RANDOM_STATE = 42
SPLIT_DATE = pd.Timestamp("2011-01-01")
ROBUSTNESS_SEEDS = [42, 7, 21, 99, 123]

print("x" * 65)
print("CERT r4.2 - Ablation Study")
print("x" * 65)

# -------------------------------------------------------------
# STEP 1 - LOAD THE FEATURE TABLE
# -------------------------------------------------------------
print("\nStep 1 - Loading feature table")
df = pd.read_csv(FEATURE_TABLE_LOCATION, low_memory=False)
df["date_only"] = pd.to_datetime(df["date_only"])
print(f"Shape : {df.shape}")
print(f"Labels : {df['label'].value_counts().to_dict()}")

# -------------------------------------------------------------
# STEP 2 - FEATURE GROUPS
# The only difference between an uncontextual and a contextual run
# is which of these two lists is handed to the model.
# -------------------------------------------------------------
print("\nStep 2 - Defining feature groups")
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
    "http_after_hours",
]
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
    "functional_unit_encoded",
]
ALL_FEATURES = BEHAVIOURAL_FEATURES + CONTEXTUAL_ADDITIONS
print(f"Behavioural only : {len(BEHAVIOURAL_FEATURES)}")
print(f"All features : {len(ALL_FEATURES)}")

# -------------------------------------------------------------
# STEP 3 - TEMPORAL SPLIT
# Train on 2010, test on 2011. Used by configurations A to D.
# -------------------------------------------------------------
print("\nStep 3 - Temporal split")
train_df = df[df["date_only"] < SPLIT_DATE].copy()
test_df = df[df["date_only"] >= SPLIT_DATE].copy()
y_train = train_df["label"].values.astype(int)
y_test = test_df["label"].values.astype(int)
print(f"Train : {len(train_df):,} rows ({y_train.sum():,} malicious)")
print(f"Test  : {len(test_df):,} rows ({y_test.sum():,} malicious)")

# -------------------------------------------------------------
# STEP 4 - HELD-OUT USER SPLIT
# 700 employees to train, 300 held out. No employee appears in both.
# Used by configurations E and F.
# -------------------------------------------------------------
print("\nStep 4 - Held-out user split")
user_labels = df.groupby("user")["label"].max() #differentiate between normal and malicious user (70 malicous, 930 normal)

# passing a plain numpy array avoids
# "TypeError: only integer scalar arrays can be converted to a scalar index".

train_users, test_users = train_test_split(
    user_labels.index.to_numpy(), # numpy array of user ids
    test_size=0.3, # test size 30%, train size 70%
    stratify=user_labels.values, # Stratify keeps the class ratio the same in the train and test sets. It does not make the classes equal, it preserves the original proportion
    random_state=RANDOM_STATE # fixed seed 
)

train_df_user = df[df["user"].isin(train_users)].copy() # users in train users
test_df_user = df[df["user"].isin(test_users)].copy() # users in test users

y_train_user = train_df_user["label"].values.astype(int)  # .values to turn into Numpy array
y_test_user = test_df_user["label"].values.astype(int)

print(f"Train users : {len(train_users)} \n Test users : {len(test_users)}")
print(f"Train rows : {len(train_df_user):,} ({y_train_user.sum():,} malicious)")
print(f"Test  rows : {len(test_df_user):,} ({y_test_user.sum():,} malicious)")

# -------------------------------------------------------------
# STEP 5 - ONE CONFIGURATION
# Every configuration goes through this same function, so nothing
# differs between them except the arguments passed in.
# -------------------------------------------------------------
def evaluate_config(name, label, model_type, features, train_df, test_df, y_train, y_test, quiet=False):

    X_train = train_df[features].values.astype(float) # selects column for training, turned into numpy array
    X_test = test_df[features].values.astype(float) # selects column for testing, turned into numpy array

    scaler = StandardScaler() # scaler applied for SMOTE 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "rf":
        smote = SMOTE(random_state=RANDOM_STATE) # fit (learn mean and standard deviation), transform scale data using learned values
        X_train_scaled_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_scaled, y_train) # no fitting prevent data leak 

        model = RandomForestClassifier(n_estimators=200, # more trees more stable
                                       max_depth=20, # limit depth prevent overfitting
                                       class_weight="balanced", # give more attention to minority class
                                       random_state=RANDOM_STATE, # fixed seed 
                                       n_jobs=-1 # use all cpus
                                       )
        
        model.fit(X_train_scaled_SMOTE, y_train_SMOTE) #start model training 

        y_pred = model.predict(X_test_scaled) # predict each test row 
        y_predicted_prob = model.predict_proba(X_test_scaled)[:, 1] #return the prediction probability, column 0 is normal probability, 1 is maliicous proabibilty

        auc = roc_auc_score(y_test, y_predicted_prob) # auc score

    else:
        # Isolation Forest is semi-supervised, it only sees normal rows
        X_train_scaled_normal = X_train_scaled[y_train == 0]

        model = IsolationForest(n_estimators=200, # more trees more stable
                                contamination=0.01, # under assumption of high class imbalance
                                random_state=RANDOM_STATE, # fixed seed
                                n_jobs=-1 # use all cpus
                                )
        
        model.fit(X_train_scaled_normal) # train the model

        y_pred = np.where(model.predict(X_test_scaled) == -1, 1, 0) # -1 is outlier, np.where(condition, 1, 0), if condition is True, put 1, otherwise put 0.

        # Isolation Forest gives lower scores to more abnormal rows, so multiply by -1
        # to make higher scores mean higher malicious/anomaly risk for AUC.
        auc = roc_auc_score(y_test, -model.score_samples(X_test_scaled))

    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    if not quiet:
        print(f"\n[{label}] {name}")
        print(f"Model : {'Random Forest' if model_type == 'rf' else 'Isolation Forest'}")
        print(f"Features : {len(features)}")
        print(f"F1={f1:.4f} Prec={precision:.4f} Rec={recall:.4f} AUC={auc:.4f}")

        report_path = os.path.join(ABLATION_FOLDER, f"report_{label}.txt")
        with open(report_path, "w") as file_handle:
            file_handle.write(f"{name}\n")
            file_handle.write("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
            file_handle.write(f"Features : {len(features)}\n\n")
            file_handle.write(classification_report(y_test, y_pred,target_names=["Normal", "Malicious"],zero_division=0))

    return {"label": label, "name": name, "model": "Random Forest" if model_type == "rf" else "Isolation Forest",
            "n_features": len(features), "f1": round(f1, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "auc_roc": round(auc, 4)}

# -------------------------------------------------------------
# STEP 6 - RUN THE SIX CONFIGURATIONS
# -------------------------------------------------------------
print("\nStep 6 - Running configurations (this takes a while)")
results = []

results.append(evaluate_config("Semi-Supervised + Uncontextual", "A", "if",
                               BEHAVIOURAL_FEATURES, train_df, test_df, y_train, y_test))
results.append(evaluate_config("Semi-Supervised + Contextual", "B", "if",
                               ALL_FEATURES, train_df, test_df, y_train, y_test))
results.append(evaluate_config("Supervised + Uncontextual", "C", "rf",
                               BEHAVIOURAL_FEATURES, train_df, test_df, y_train, y_test))
results.append(evaluate_config("Supervised + Contextual", "D", "rf",
                               ALL_FEATURES, train_df, test_df, y_train, y_test))
results.append(evaluate_config("Held-out Users + Uncontextual", "E", "rf",
                               BEHAVIOURAL_FEATURES, train_df_user, test_df_user,
                               y_train_user, y_test_user))
results.append(evaluate_config("Held-out Users + Contextual", "F", "rf",
                               ALL_FEATURES, train_df_user, test_df_user,
                               y_train_user, y_test_user))

result_a, result_b, result_c, result_d, result_e, result_f = results

# -------------------------------------------------------------
# STEP 7 - RESULTS TABLE
# -------------------------------------------------------------
print("\nStep 7 - Results")

table_border = "+-----+----------------------------------------+-------------------+------+--------+--------+--------+"
table_header = f"| {'Cfg':^3} | {'Configuration':<38} | {'Model':<17} | {'Feat':>4} | {'F1':>6} | {'Prec':>6} | {'AUC':>6} |"

print(table_border)
print(table_header)
print(table_border)
for row in results:
    print(f"| {row['label']:^3} | {row['name']:<38} | {row['model']:<17} | {row['n_features']:>4} | "
          f"{row['f1']:>6.4f} | {row['precision']:>6.4f} | {row['auc_roc']:>6.4f} |")
print(table_border)

print("\nCore comparisons")
print(f"Context effect, temporal split : F1 {result_c['f1']} -> {result_d['f1']}"
      f" (C vs D, delta = {result_d['f1'] - result_c['f1']:+.4f})")
print(f"Context effect, held-out users : F1 {result_e['f1']} -> {result_f['f1']}"
      f" (E vs F, delta = {result_f['f1'] - result_e['f1']:+.4f})")
print(f"Supervised vs Semi-supervised : F1 {result_b['f1']} -> {result_d['f1']}"
      f" (B vs D, delta = {result_d['f1'] - result_b['f1']:+.4f})")
print(f"Temporal vs held-out (contextual): F1 {result_d['f1']} -> {result_f['f1']}"
      f" (D vs F, delta = {result_f['f1'] - result_d['f1']:+.4f})")

results_df = pd.DataFrame(results)
results_path = os.path.join(ABLATION_FOLDER, "ablation_results.csv")
results_df.to_csv(results_path, index=False) # index = False prevent pandas from writing row numbers 
print(f"\nSaved: {results_path}")

# -------------------------------------------------------------
# STEP 8 - SEED ROBUSTNESS FOR E AND F
# Same two configurations repeated with different random seeds, to show
# the gap is not the result of one lucky split.
# -------------------------------------------------------------
print("\nStep 8 - Seed robustness (E and F)")
e_f1_scores = []
f_f1_scores = []

for seed in ROBUSTNESS_SEEDS:
    seed_train_users, seed_test_users = train_test_split(
        user_labels.index.to_numpy(), # numpy array of user ids 
        test_size=0.3, # test size 30%, train size 70%
        stratify=user_labels.values,  # Stratify keeps the class ratio the same in the train and test sets. It does not make the classes equal, it preserves the original proportion.
        random_state=seed # fixed seed 
        )

    seed_train_df = df[df["user"].isin(seed_train_users)].copy() # users in train users
    seed_test_df = df[df["user"].isin(seed_test_users)].copy() # users in test users

    seed_y_train = seed_train_df["label"].values.astype(int)  # .values to turn into Numpy array
    seed_y_test = seed_test_df["label"].values.astype(int)

    config_e_result = evaluate_config("E seed run", "E", "rf", BEHAVIOURAL_FEATURES,
                                      seed_train_df, seed_test_df, seed_y_train, seed_y_test,
                                      quiet=True)
    config_f_result = evaluate_config("F seed run", "F", "rf", ALL_FEATURES,
                                      seed_train_df, seed_test_df, seed_y_train, seed_y_test,
                                      quiet=True)

    e_f1_scores.append(config_e_result["f1"])
    f_f1_scores.append(config_f_result["f1"])

    print(f"seed {seed:>3} : E F1 = {config_e_result['f1']:.4f}   F F1 = {config_f_result['f1']:.4f}")

e_f1_array = np.array(e_f1_scores)
f_f1_array = np.array(f_f1_scores)
print(f"\nE (uncontextual) : mean {e_f1_array.mean():.4f}  std {e_f1_array.std():.4f}")
print(f"F (contextual)   : mean {f_f1_array.mean():.4f}  std {f_f1_array.std():.4f}")

robustness_df = pd.DataFrame({"seed": ROBUSTNESS_SEEDS,
                              "E_f1": e_f1_scores,
                              "F_f1": f_f1_scores})
robustness_path = os.path.join(ABLATION_FOLDER, "robustness_check_E_F.csv")
robustness_df.to_csv(robustness_path, index=False)
print(f"Saved: {robustness_path}")


