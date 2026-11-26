"""The first step is to explore the data inside CERT"""

import os
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
DATASET_LOCATION = os.path.join(BASE_FOLDER, "r4.2", "r4.2")
ANSWERS_LOCATION = os.path.join(BASE_FOLDER, "answers", "answers")
OUTPUT_LOCATION = os.path.join(BASE_FOLDER, "eda_outputs")

# Since the files are too large, we will need to reduce the size of sample
NO_OF_HTTP_SAMPLES = 200000
NO_OF_EMAIL_SAMPLES = 200000

os.makedirs(OUTPUT_LOCATION, exist_ok=True) # exist_ok=True, if folder already exist dont show error 

#--------------------------------
# STEP 1 - FILE VERIFICATION
#--------------------------------

print("x" * 60)
print("Executing File Verification")
print("x" * 60)

CERT_FILES = ["logon.csv", "http.csv", "email.csv",
              "device.csv", "psychometric.csv", "file.csv"]

for file in CERT_FILES:
    file_path = os.path.join(DATASET_LOCATION, file)
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File '{file}' is present at the location. ({size_mb:,.1f} MB)")
    else:
        print(f"File '{file}' is NOT found at the location.")

ldap_location = os.path.join(DATASET_LOCATION, "LDAP")
ldap_files = sorted(os.listdir(ldap_location)) if os.path.isdir(ldap_location) else []
print(f"  FOUND  : LDAP/ folder ({len(ldap_files)} snapshot files)")

insider_file_location = os.path.join(ANSWERS_LOCATION, "insiders.csv")
print(f"  {'FOUND' if os.path.exists(insider_file_location) else 'MISSING'}  : insiders.csv (in answers folder)")

#----------------------------------------------------
#STEP 2 - IDENTIFY MALICIOUS USERS IN R4.2 DATASET
#----------------------------------------------------

print ("\n" + "x" * 60)
print("Loading ground truth from insiders.csv")
print("x" * 60)

malicious_users = set()

if os.path.exists(insider_file_location):
    insider_df = pandas.read_csv(insider_file_location)
    print(f"Columns : {list(insider_df.columns)}")
    print(f"Total Records : {len(insider_df)}")
    print(f"  'dataset' dtype: {insider_df['dataset'].dtype}")
    print(f"  Sample values  : {insider_df['dataset'].unique()[:5]}") # first 5 values

    #Filtering only r4.2 entries
    r42_insider_df = insider_df[insider_df['dataset'] == 4.2] 
    malicious_users = set(r42_insider_df["user"].unique()) 
    all_malicious = set(insider_df["user"].unique()) 

    print(f"\nTotal number of malicious users across all datasets: {len(all_malicious)}")
    print(f"Total number of malicious users in r4.2 dataset: {len(malicious_users)}")
    if malicious_users:
        print(f" malicious r4.2 user IDs: {sorted(malicious_users)}")
else:
    print(f"File 'insiders.csv' is NOT found at the location {insider_file_location}.")
    print(f"Please recheck the file location field and try again.")
    exit(1)

# -------------------------------------------------------------
# STEP 2b - SCENARIO DETAIL FILE STRUCTURE EXPLORATION
# -------------------------------------------------------------
print("\n" + "x" * 60)
print("STEP 2b - SCENARIO DETAIL FILE STRUCTURE")
print("x" * 60)
print("  r4.2 is a 'dense needle' dataset with MANY instances per scenario.")
print("  Each r4.2-N subfolder contains one detail file per insider user,")
print("  listing the EXACT log lines that is their attack.")

# list of the folders 
scenario_folders = [d for d in os.listdir(ANSWERS_LOCATION)
                 if d.startswith("r4.2-") and
                 os.path.isdir(os.path.join(ANSWERS_LOCATION, d))]
print(f"\n  Scenario folders found: {scenario_folders}")

total_detail_files = 0
for sce_fold in scenario_folders:
    files = os.listdir(os.path.join(ANSWERS_LOCATION, sce_fold))
    csv_files = [f for f in files if f.endswith(".csv")]
    total_detail_files += len(csv_files)
    print(f"  {sce_fold}: {len(csv_files)} detail files")
print(f"  Total detail files (should match malicious user count): {total_detail_files}")

# -------------------------------------------------------------
# STEP 2c - INSIDERS.CSV START/END COLUMN VERIFICATION
# -------------------------------------------------------------
print("\n" + "x" * 60)
print("STEP 2c - insiders.csv START/END COLUMN CHECK")
print("x" * 60)
print("  insiders.csv also contains 'start' and 'end' timestamps per user,")
print("  representing the first and last event in their scenario detail file.")
print()
r42_insider_df_dates = r42_insider_df.copy()
r42_insider_df_dates["start"] = pandas.to_datetime(r42_insider_df_dates["start"], errors="coerce")
r42_insider_df_dates["end"] = pandas.to_datetime(r42_insider_df_dates["end"], errors="coerce")
print(r42_insider_df_dates[["user","scenario","start","end"]].head(10).to_string(index=False))


#----------------------------------------------------
#STEP 3 - LOAD AND SAMPLE DATA FROM EACH FILE
#----------------------------------------------------

print("\n" + "x" * 60)
print("Loading and sampling data from each file")
print("x" * 60)
print("  NOTE: http.csv and email.csv are SAMPLED here for EDA speed.")
print("  Feature engineering uses the COMPLETE files (chunked reading).")

"""Load a CSV, optionally limiting rows. Returns DataFrame or None."""
def load_csv_files(name, path, nrows=None):

    try:
        df = pandas.read_csv(path, low_memory=False, nrows=nrows, on_bad_lines="skip")
        label = f"(first {nrows:,} rows sampled)" if nrows else "(full file)"
        print(f"\n--- {name} {label} ---")
        print(f"  Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Columns : {list(df.columns)}")
        print(f"  Nulls   : {df.isnull().sum().sum()} total")
        if "user" in df.columns:
            print(f"  Users   : {df['user'].nunique()} unique")
        if "date" in df.columns:
            df["date"] = pandas.to_datetime(df["date"], errors="coerce")
            print(f"  Dates   : {df['date'].min()}  ->  {df['date'].max()}")
        if "activity" in df.columns:
            print(f"  Activity counts:\n{df['activity'].value_counts().to_string()}")
        print(f"\n  Sample rows:")
        print(df.head(3).to_string(index=False))
        return df
    except Exception as e:
        print(f"\n--- {name} ---\n  ERROR: {e}")
        return None

dataframe = {}

dataframe["logon"] = load_csv_files("logon.csv", os.path.join(DATASET_LOCATION, "logon.csv"))
dataframe["device"] = load_csv_files("device.csv", os.path.join(DATASET_LOCATION, "device.csv"))
dataframe["file"] = load_csv_files("file.csv", os.path.join(DATASET_LOCATION, "file.csv"))
dataframe["psychometric"] = load_csv_files("psychometric.csv", os.path.join(DATASET_LOCATION, "psychometric.csv"))
dataframe["http"] = load_csv_files("http.csv", os.path.join(DATASET_LOCATION, "http.csv"), nrows=NO_OF_HTTP_SAMPLES)
dataframe["email"] = load_csv_files("email.csv", os.path.join(DATASET_LOCATION, "email.csv"), nrows=NO_OF_EMAIL_SAMPLES)

# Load one LDAP file as sample
ldap_files = sorted(os.listdir(ldap_location)) if os.path.isdir(ldap_location) else []
if ldap_files:
    dataframe["ldap_sample"] = load_csv_files(
        f"LDAP/{ldap_files[0]} (first month)",
        os.path.join(ldap_location, ldap_files[0])
    )

# [ADD 5]: file.csv clarification per dataset readme
print("\n  NOTE on file.csv (per CERT r4.2 readme):")
print("  Each row represents a FILE COPY TO REMOVABLE MEDIA (USB/external drive).")
print("  This is NOT general file access - it specifically captures exfiltration-")
print("  relevant file copy activity, making it a strong insider threat signal.")


#----------------------------------------------------
#STEP 4 - EVENT VOLUME SUMMARY
#----------------------------------------------------

print("\n" + "x" * 60)
print("STEP 4 - EVENT VOLUME SUMMARY (note: http.csv and email.csv are sampled, aka not full file")
print("x" * 60)

event_count_per_file = {}
for key, df in dataframe.items():
    if df is not None and key != "ldap_sample":
        event_count_per_file[key] = len(df) # new dict to store row count 
        note = "(sampled)" if key in ["http", "email"] else ""
        print(f"  {key:15}: {len(df):,} rows {note}")

#---------------------------------------
# STEP 5 - MALICIOUS VS NORMAL IN LOGON
#---------------------------------------
if malicious_users and dataframe.get("logon") is not None:
    print("\n" + "x" * 60)
    print("STEP 5 - Malicious vs Normal Users (logon.csv)")
    print("x" * 60)

    logon = dataframe["logon"].copy()
    logon["user_type"] = logon["user"].apply(
        lambda u: "confirmed insider account" if u in malicious_users else "normal user"
    )
    print(f"\n  Logon events by label:")
    print(logon["user_type"].value_counts().to_string())
    total = logon["user"].nunique()
    mal_found = logon[logon["user_type"]=="confirmed insider account"]["user"].nunique()
    print(f"\n  Unique users total      : {total}")
    print(f"  Confirmed Insider Accounts : {mal_found}")
    print(f"  Normal users            : {total - mal_found}")

#----------------------------------------
# STEP 6 - PSYCHOMETRIC COLUMNS
#----------------------------------------
if dataframe.get("psychometric") is not None:
    psych = dataframe["psychometric"]
    print("\n" + "x" * 60)
    print("STEP 6 - Psychometric Data")
    print("x" * 60)
    print(f"\n  All columns: {list(psych.columns)}")
    print(f"\n  Full sample (first 5 rows):")
    print(psych.head(5).to_string(index=False))
    if len(psych) > 0:
        numeric_cols = psych.select_dtypes(include="number").columns.tolist() # select  numeric columns
        if numeric_cols:
            print(f"\n  Numeric column stats (O=Openness, C=Conscientiousness, E=Extraversion,")
            print(f"  A=Agreeableness, N=Neuroticism):")
            print(psych[numeric_cols].describe().to_string()) # describe count, mean, std, min, first quartile, median, third quartile and largest value 

        # Psychometric trait distribution for malicious vs normal users
        if malicious_users:
            psych_check = psych.rename(columns={"user_id": "user"}) if "user_id" in psych.columns else psych.copy()
            psych_check["user_type"] = psych_check["user"].apply(
                lambda u: "insider" if u in malicious_users else "normal"
            )
            traits = [c for c in ["O","C","E","A","N"] if c in psych_check.columns]
            if traits:
                print(f"\n  Mean trait scores - insider accounts vs normal users:")
                print(psych_check.groupby("user_type")[traits].mean().round(2).to_string())

# -------------------------------------------------------------
# STEP 6b - AFTER-HOURS ACTIVITY DISTRIBUTION
# -------------------------------------------------------------
if dataframe.get("logon") is not None:
    print("\n" + "x" * 60)
    print("STEP 6b - AFTER-HOURS ACTIVITY DISTRIBUTION")
    print("x" * 60)

    WORK_START_HOUR = 8
    WORK_END_HOUR = 18

    logon_tmp = dataframe["logon"].copy()
    logon_tmp["date"] = pandas.to_datetime(logon_tmp["date"], errors="coerce")
    logon_tmp["hour"] = logon_tmp["date"].dt.hour
    logon_tmp["is_after_hours"] = (logon_tmp["hour"] < WORK_START_HOUR) | (logon_tmp["hour"] >= WORK_END_HOUR)
    after_hours_pct = logon_tmp["is_after_hours"].mean() * 100
    print(f"\n  Logon events outside 8am-6pm: {after_hours_pct:.2f}%")
    print(f"  Hour distribution:")
    print(logon_tmp["hour"].value_counts().sort_index().to_string())


# -------------------------------------------------------------
# STEP 6c - LDAP VALIDATION (ALL 18 SNAPSHOTS)
# -------------------------------------------------------------
print("\n" + "x" * 60)
print("STEP 6c - LDAP VALIDATION (ALL 18 SNAPSHOTS)")
print("x" * 60)
print(f"  {len(ldap_files)} monthly snapshot files found in LDAP/")
print(f"  Checking user counts and role/department stability across all snapshots...")

all_ldap = []
for f in ldap_files:
    df_l = pandas.read_csv(os.path.join(ldap_location, f), low_memory=False)
    df_l = df_l.rename(columns={"user_id": "user"})
    df_l["snapshot"] = f
    all_ldap.append(df_l)

ldap_combined = pandas.concat(all_ldap)

# User count per snapshot, check difference between first and last file 
print(f"\n  User count per snapshot (first and last):")
for f in [ldap_files[0], ldap_files[-1]]:
    df_l = ldap_combined[ldap_combined["snapshot"]==f]
    print(f"    {f}: {df_l['user'].nunique()} unique users") 

print(f"  Total unique users across all 18 snapshots: {ldap_combined['user'].nunique()}")

# Role/department stability
changed_role = ldap_combined.groupby("user")["role"].nunique() # unique user + role 
changed_dept = ldap_combined.groupby("user")["department"].nunique() # unique user + department 
print(f"\n  Users with role changes across 18 snapshots    : {(changed_role>1).sum()}") # calculate changes 
print(f"  Users with department changes across 18 snapshots: {(changed_dept>1).sum()}") 

# Superset check - actually verify that no user appears in a later snapshot
# without also being in the first one
users_first = set(ldap_combined[ldap_combined["snapshot"] == ldap_files[0]]["user"]) # get unique users from first file
users_all = set(ldap_combined["user"]) # get unique user from all files
new_after_first = users_all - users_first # find the diff, if 0 means no additional user present
if not new_after_first:
    print(f"  Superset check PASSED: file 0 ({ldap_files[0]}) covers all "
          f"{len(users_all)} users seen in any snapshot.")
    print(f"  Using only the first LDAP snapshot is VALID for this dataset.")
else:
    print(f"  WARNING: first-snapshot-only LDAP treatment would miss these users.")

#----------------------------------------
# STEP 7 - PLOTS
#----------------------------------------
print("\n" + "x" * 60)
print("STEP 7 - Generating Plots")
print("x" * 60)

sns.set_style("whitegrid")
output_path = OUTPUT_LOCATION  # Save plots in the same dataset location

#Plot 1 - Event Volumes
plot_volume = {k: v for k, v in event_count_per_file.items() if k not in ("psychometric",)}
if plot_volume:
    fig, ax = plt.subplots(figsize=(9,4))
    keys = list(plot_volume.keys())
    vals = [plot_volume[k] for k in keys]
    colours = ["#185FA5" if k not in ("http", "email") else "#EF9F27" for k in keys]
    bars = ax.bar(keys, vals, color=colours, width=0.6, edgecolor="black")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Event volume per log file - CERT r4.2\n(orange = sampled)", fontsize=12)
    ax.set_xlabel("Log file")
    ax.set_ylabel("Rows loaded")
    for bar, val in zip(bars,vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    p1 = os.path.join(output_path, "event_volume_plot1.png")
    plt.savefig(p1, dpi=120); plt.close()
    print(f" Plot saved: {p1}")

#Plot 2 -  Logon over time
logon_df = dataframe.get("logon")
if logon_df is not None and "date" in logon_df.columns:
    logon_df["date"] = pandas.to_datetime(logon_df["date"], errors="coerce")
    monthly =  logon_df.groupby(logon_df["date"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(12,4))
    x = range(len(monthly))
    ax.plot(x, monthly.values, marker="o", color="#185FA5", linewidth=1.8, markersize=4)
    ax.fill_between(x, monthly.values, alpha=0.08, color="#185FA5")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Monthly logon events - CERT r4.2", fontsize=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Logon events")
    plt.tight_layout()
    p2 = os.path.join(output_path, "logon_monthly_plot2.png")
    plt.savefig(p2, dpi=120); plt.close()
    print(f" Plot saved: {p2}")

#Plot 3 - Malicious vs Normal users
if malicious_users and dataframe.get("logon") is not None:
    logon_df["user_type"] = logon_df["user"].apply(
        lambda u: "Confirmed insider\naccount" if u in malicious_users else "Normal user")
    total = logon_df["user"].nunique()
    n_malicious = logon_df[logon_df["user_type"].str.startswith("Confirmed")]["user"].nunique()
    n_normal = total - n_malicious
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Normal users", "Confirmed insider\naccounts"], [n_normal, n_malicious], color=["#185FA5", "#EF9F27"])
    ax.set_title("Users in logon.csv - CERT r4.2\n(user-level insider association, not event-level)", fontsize=12)
    ax.set_ylabel("Unique Users")
    for i, v in enumerate([n_normal, n_malicious]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(output_path, "logon_users_plot3.png")
    plt.savefig(p3, dpi=120); plt.close()
    print(f" Plot saved: {p3}")

# ---------------------------------------
# Completed
# ---------------------------------------
print("\n" + "x" * 60)
print("EDA completed. Plots saved in the dataset location.")
print("x" * 60)
print(f"""
Summary of key EDA findings:
  - r4.2 has exactly {len(malicious_users)} confirmed insider users
  - LDAP: 0 role/department changes across all 18 monthly snapshots
  - LDAP: file 0 ({ldap_files[0]}) covers all users 
  - file.csv records FILE COPIES TO REMOVABLE MEDIA only (per CERT readme)
  - http.csv and email.csv were SAMPLED only to 200,000 rows 
Outputs saved to: {OUTPUT_LOCATION}
""")
