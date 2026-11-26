import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
import joblib

# ------------------------------------
# CONFIGURATION
# ------------------------------------
BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATASET_LOCATION = os.path.join(BASE_FOLDER, "r4.2", "r4.2")
ANSWERS_LOCATION = os.path.join(BASE_FOLDER, "answers", "answers")
OUTPUT_LOCATION = BASE_FOLDER

START_WORK_HOUR = 8
END_WORK_HOUR = 18

SUSPICIOUS_KEYWORDS = [
"career", 
"dropbox", 
"glassdoor", 
"indeed", 
"job",
"linkedin", 
"monster", 
"resume", 
"usajobs", 
"wikileaks"
]

print("x" * 60)
print("YEW FARN PING - FEATURE ENGINEERING on CERT r4.2 DATASET")
print("x" * 60)

# ------------------------------------
# Step 1 - Ground Truth Labels
# ------------------------------------
print("\nSTEP 1 - LOADING GROUND TRUTH LABELS")

ins = pd.read_csv(os.path.join(ANSWERS_LOCATION, "insiders.csv"))
r42 = ins[ins["dataset"] == 4.2]
malicious_users = set(r42["user"].unique())
print(f"Total Number of Malicious Users in CERT r4.2: {len(malicious_users)}")

# ---------------------------------------------
# STEP 1b - Day-level malicious ground truth
# ---------------------------------------------

print("\n STEP 1B - EXTRACTING DAY-LEVEL MALICIOUS GROUND TRUTH")

scenario_folders = [
    d for d in os.listdir(ANSWERS_LOCATION)
    if d.startswith("r4.2-") and os.path.isdir(os.path.join(ANSWERS_LOCATION, d))
]
print(f"Found {len(scenario_folders)} r4.2 scenario folders: {scenario_folders}")

malicious_days = defaultdict(set)
file_count, row_count, parse_errors = 0, 0, 0

for scenario_folder in scenario_folders:
    folder_path = os.path.join(ANSWERS_LOCATION, scenario_folder)
    detail_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    for fname in detail_files:
        file_count += 1
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                row_count += 1
                parts = line.strip().split(",")
                if len(parts) < 4:
                    parse_errors += 1
                    continue
                try:
                    date_val = pd.to_datetime(parts[2]).date()
                    user = parts[3]
                    if user in malicious_users:
                        malicious_days[user].add(date_val)
                except Exception:
                    parse_errors += 1
                    continue

n_users_with_days = len(malicious_days)
total_malicious_day_count = sum(len(v) for v in malicious_days.values())
users_missing_detail =  malicious_users - set(malicious_days.keys())

print(f"Detail files parsed: {file_count}")
print(f"Total rows read: {row_count} (parse_errors: {parse_errors})")
print(f"Malicious users with day-level ground truth: {n_users_with_days}")
if users_missing_detail:
    print(f"WARNING- THERE ARE MALICIOUS USERS WITHOUT ANY FLAGGED ANOMALY: {sorted(users_missing_detail)}")
    print(f"These users have no logs that are flagged anomaly despite being a malicious user, please check dataset integrity")
print(f"Total count of users and malicious days (user,date): {total_malicious_day_count}")
print(f"Avg malicious days per user: {total_malicious_day_count/max(n_users_with_days,1):.2f}")

#Save for Reference
day_rows = [(u,d) for u, dates in malicious_days.items() for d in dates]
day_df =  pd.DataFrame(day_rows, columns=["user", "malicious_date"])
day_df.to_csv(os.path.join(OUTPUT_LOCATION, "malicious_days_ground_truth_v3.csv"), index=False)
print(f"Saved: {os.path.join(OUTPUT_LOCATION, 'malicious_days_ground_truth_v3.csv')}")


# ------------------------------------
# Step 2 - Static Contextual Features
# ------------------------------------
print("\nSTEP 2 - LOADING STATIC CONTEXTUAL FEATURES")

psycho = pd.read_csv(os.path.join(DATASET_LOCATION, "psychometric.csv"))
psycho = psycho.rename(columns={"user_id": "user"})[["user","O","C","E","A","N"]]
print(f" No. of users in psychometric.csv: {len(psycho)} users")

ldap_location = os.path.join(DATASET_LOCATION, "LDAP")
ldap_files = sorted(os.listdir(ldap_location))
ldap = pd.read_csv(os.path.join(ldap_location, ldap_files[0]), low_memory=False)
ldap = ldap.rename(columns={"user_id": "user"})
# business_unit only has 1 value, drop it as it does not provide any information for the model
ldap = ldap[["user", "role", "department", "functional_unit"]].fillna("Unknown")
print (f"LDAP/{ldap_files[0]}: {len(ldap)} users")
print(f"Note: business_unit column dropped, this is because it has only a constant value and does not provide any benefit for model training whatsoever")
print("Only the first file is used from the LDAP because it contains the information of all 1000 users, it is confirmed that the 1000 user did not have any changes to their roles throughout the whole r4.2 dataset.")


# ------------------------
# HELPER
# ------------------------
def parse_dates(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["is_after_hours"] = (df["hour"] < START_WORK_HOUR) | (df["hour"] >= END_WORK_HOUR)
    return df


# -------------------------
# STEP 3 -  LOGON FEATURES
# -------------------------
print("\nSTEP 3 ENGINEERING LOGON FEATURES")

logon = pd.read_csv(os.path.join(DATASET_LOCATION, "logon.csv"), low_memory=False)
logon = parse_dates(logon)

logon_features = logon[logon["activity"] == "Logon"].groupby(["user", "date_only"]).agg(
    logon_count = ("id", "count"),
    after_hours_logon = ("is_after_hours", "sum"),
    weekend_logon = ("is_weekend", "sum"),
    unique_pcs = ("pc", "nunique"),
).reset_index()

logoff_features = logon[logon["activity"] == "Logoff"].groupby(["user", "date_only"]).agg(
    logoff_count = ("id", "count"),
).reset_index()

logon_daily = pd.merge(logon_features, logoff_features,
                       on=["user", "date_only"], how="outer").fillna(0)

print(f"Shape: {logon_daily.shape}")


# ---------------------------------------
# STEP 4 - Device (USB) features
# ---------------------------------------
print("\nStep 4 - Engineering Device (USB) Features")

device = pd.read_csv(os.path.join(DATASET_LOCATION, "device.csv"), low_memory=False)
device = parse_dates(device)

device_features = device[device["activity"] == "Connect"].groupby(["user", "date_only"]).agg(
    usb_connect_count = ("id", "count"),
    usb_after_hours = ("is_after_hours", "sum"),
).reset_index()
print(f"Shape: {device_features.shape}")

# --------------------------------------
# STEP 5 - File Features
# --------------------------------------
print("\nStep 5 Engineering File Features")

file = pd.read_csv(os.path.join(DATASET_LOCATION, "file.csv"), low_memory=False)
file = parse_dates(file)

file_features = file.groupby(["user", "date_only"]).agg(
    file_copy_count = ("id", "count"),
    file_copy_after_hours = ("is_after_hours", "sum"),
).reset_index()
print(f"Shape: {file_features.shape}")

# --------------------------------------
# STEP 6 - Email features (chunked)
# --------------------------------------
print("\nStep 6 Engineering Email Features")

email_chunks = []
chunk_size = 100000
email_location = os.path.join(DATASET_LOCATION, "email.csv")

for chunk in pd.read_csv(email_location, low_memory=False,
                            chunksize=chunk_size, on_bad_lines="skip"):
    chunk = parse_dates(chunk)

    if "to" in chunk.columns:
        chunk["have_external"] = chunk["to"].astype(str).apply(
            lambda x: int(any(
                addr.strip() and "@dtaa.com" not in addr.lower()
                for addr in x.replace(";", ",").split(",")
                if addr.strip() not in ("nan", "")
            ))
        )
    else:
        chunk["have_external"] = 0

    if "attachments" in chunk.columns:
        chunk["have_attachment"] = (
            pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        chunk["have_attachment"] = 0

    agg = chunk.groupby(["user", "date_only"]).agg(
        emails_total = ("id", "count"),
        emails_external = ("have_external", "sum"),
        emails_with_attachment = ("have_attachment", "sum"),
        emails_after_hours = ("is_after_hours", "sum"),
    ).reset_index()
    email_chunks.append(agg)

email_features = (
    pd.concat(email_chunks)
    .groupby(["user", "date_only"])
    .sum()
    .reset_index()
)

email_features["emails_external"] = pd.to_numeric(
    email_features["emails_external"], errors="coerce"
).fillna(0).astype(float)

print(f"Shape: {email_features.shape}")

# --------------------------------------
# STEP 7 - HTTP features (chunked)
# --------------------------------------
print("\nStep 7 - Engineering HTTP features")

http_chunks = []
http_location = os.path.join(DATASET_LOCATION, "http.csv")

for chunk in pd.read_csv(http_location, low_memory=False,
                          chunksize=chunk_size, on_bad_lines="skip"):
    chunk = parse_dates(chunk)

    if "url" in chunk.columns:
        chunk["is_suspicious_url"] = chunk["url"].astype(str).apply(
            lambda u: 1 if any(d in u.lower() for d in SUSPICIOUS_KEYWORDS) else 0
        )
    else:
        chunk["is_suspicious_url"] = 0

    agg = chunk.groupby(["user", "date_only"]).agg(
        http_count = ("id", "count"),
        suspicious_url_count = ("is_suspicious_url", "sum"),
        http_after_hours = ("is_after_hours", "sum"),
    ).reset_index()
    http_chunks.append(agg)

http_features = (
    pd.concat(http_chunks)
    .groupby(["user", "date_only"])
    .sum()
    .reset_index()
)
print(f"Shape: {http_features.shape}")

# ------------------------------------------------
# STEP 8 - Merging Previous Behavioural Features
# ------------------------------------------------
print("\nStep 8 - Merging Previously Engineered Behavioural Features")

feature_table = logon_daily.copy()
for name, df in [
    ("device", device_features),
    ("file",   file_features),
    ("email",  email_features),
    ("http",   http_features),
]:
    # OUTER join - previously a LEFT join onto logon days, which silently
    # dropped any day with device/file/email/http activity but no logon
    # event. Outer join keeps those days; missing counts become 0 below.
    feature_table = pd.merge(feature_table, df,
                              on=["user", "date_only"], how="outer")
    print(f"  After merging {name}: {feature_table.shape}")

behaviour_cols = [c for c in feature_table.columns
                  if c not in ["user", "date_only"]]
feature_table[behaviour_cols] = feature_table[behaviour_cols].fillna(0)
print(f"\nMerged table shape: {feature_table.shape}")

# ------------------------------------------------
# STEP 9 - Personalised Deviation Features
# ------------------------------------------------
print("\nStep 9 - Personalizing Deviation Features")

feature_table = feature_table.sort_values(
    ["user", "date_only"]).reset_index(drop=True)

source_deviation_cols = [
    "logon_count", "usb_connect_count", "file_copy_count",
    "emails_total", "emails_external", "http_count",
    "after_hours_logon", "suspicious_url_count"
]

for col in source_deviation_cols:
    if col not in feature_table.columns:
        continue
    rolling_mean = (
        feature_table.groupby("user")[col]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
    )
    # Where no baseline exists yet (fewer than 5 prior records), fall back to
    # the value itself so the deviation is 0
    feature_table[f"{col}_deviation"] = (
        feature_table[col] - rolling_mean.fillna(feature_table[col])
    )

dev_cols = [c for c in feature_table.columns if c.endswith("_deviation")]
print(f"Added {len(dev_cols)} deviation features")

for col in dev_cols:
    feature_table[col] = pd.to_numeric(
        feature_table[col], errors="coerce").fillna(0)

# ----------------------------------------------------------------------------
# STEP 10 - Merging the Static Contextual Features (Psychometrics and Roles)
# ----------------------------------------------------------------------------
print("\nStep 10 - Merging static contextual features")

feature_table = pd.merge(feature_table, psycho, on="user", how="left")
print(f"After psychometric merge: {feature_table.shape}")

feature_table = pd.merge(feature_table, ldap, on="user", how="left")
feature_table[["role", "department", "functional_unit"]] = (
    feature_table[["role", "department", "functional_unit"]].fillna("Unknown")
)
print(f"After LDAP merge: {feature_table.shape}")

# ----------------------------------------------------------------------------
# STEP 11 - Encode the Categorical Columns
# ----------------------------------------------------------------------------
print("\nStep 11 - Encoding Categorical Columns")


label_maps = {}
for col in ["role", "department", "functional_unit"]:
    if col in feature_table.columns:
        le = LabelEncoder()  # fresh encoder per column
        feature_table[f"{col}_encoded"] = le.fit_transform(
            feature_table[col].astype(str)
        )
        label_maps[col] = {int(i): c for i, c in enumerate(le.classes_)}
        print(f"Encoded '{col}': {feature_table[col].nunique()} unique values")

# Persist the encoded->text mappings so downstream consumers (e.g. the
# prototype app) can decode role/department/functional_unit codes without
# re-running feature engineering.
models_folder = os.path.join(OUTPUT_LOCATION, "models")
os.makedirs(models_folder, exist_ok=True)
joblib.dump(label_maps, os.path.join(models_folder, "label_maps.pkl"))
print(f"Saved encoder mappings: {os.path.join(models_folder, 'label_maps.pkl')}")

feature_table = feature_table.drop(
    columns=["role", "department", "functional_unit"], errors="ignore"
)

# -----------------------------------------------------------------------------------------------------------
# STEP 12 - Attach the rows that are of Malicious Dates as Malicious (e.g. Malicious = 1, Non-Malicious = 0)
# -----------------------------------------------------------------------------------------------------------
print("\nStep 12 - Labelling Malicious and Non-Malicious")

def is_malicious_day(row):
    user_days = malicious_days.get(row["user"])
    if not user_days:
        return 0
    return 1 if row["date_only"] in user_days else 0

feature_table["label"] = feature_table.apply(is_malicious_day, axis=1)

label_counts = feature_table["label"].value_counts()
print(f"Normal rows (0)   : {label_counts.get(0, 0):,}")
print(f"Malicious rows (1): {label_counts.get(1, 0):,}")
print(f"Class imbalance   : {label_counts.get(1,0)/len(feature_table)*100:.3f}% malicious")

# ------------------------
# STEP 13 - Final Checks
# ------------------------
print("\nStep 13 - Doing the Final Checks")

total_nulls = feature_table.isnull().sum().sum()
object_cols = [c for c in feature_table.columns
                if feature_table[c].dtype == object
                and c not in ["user", "date_only"]]

print(f"Total null values  : {total_nulls} ")
print(f"Object-dtype cols  : {object_cols} ")
print(f"Final table shape  : {feature_table.shape}")
print(f"\nAll columns:")
for col in feature_table.columns:
    dtype = feature_table[col].dtype
    nulls = feature_table[col].isnull().sum()
    unique = feature_table[col].nunique()
    print(f"    {col:<40} {str(dtype):<10} nulls:{nulls:<4} unique:{unique}")

# ------------------------
# STEP 14 - Save
# ------------------------
print("\nStep 14 - Saving Feature Table")

out_file = os.path.join(OUTPUT_LOCATION, "feature_table_final.csv")
feature_table.to_csv(out_file, index=False)
size_mb = os.path.getsize(out_file) / (1024 * 1024)
print(f"Saved : {out_file}")
print(f"Size  : {size_mb:.1f} MB")

print("\n" + "x" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("x" * 60)
print(f"""
Summary:
  Rows              : {len(feature_table):,}
  Columns           : {len(feature_table.columns)}
  Malicious rows    : {label_counts.get(1, 0):,}
  Normal rows       : {label_counts.get(0, 0):,}
  Output            : feature_table_final.csv
""")
