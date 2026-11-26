# Context-Aware Anomaly Detection System.py
# http://127.0.0.1:5000


import os
import io
import re
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for, flash
from Preprocess_Raw_Logs import build_feature_table


# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

MODELS_FOLDER = os.path.join(BASE_FOLDER, "models")
CONTEXT_DATA_FOLDER = os.path.join(BASE_FOLDER, "context_data")
MAX_RAW_UPLOAD = 64 * 1024 * 1024

MAX_ROWS = 2000  # prevent big uploads freezing the demo procedure
ALLOWED_THRESHOLDS = (0.30, 0.50, 0.70)
DEFAULT_THRESHOLD = 0.50

# -------------------------------------------------------------
# LABEL MAPS AND FEATURE CATEGORIES
# -------------------------------------------------------------
def _load_label_maps():
    path = os.path.join(MODELS_FOLDER, "label_maps.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"label_maps.pkl not found at {path}.")
    maps = joblib.load(path)
    return maps["department"], maps["role"], maps["functional_unit"]

DEPT_MAP, ROLE_MAP, UNIT_MAP = _load_label_maps()

STATIC_FEATURES = {"O","C","E","A","N","role_encoded","department_encoded","functional_unit_encoded"}
DEVIATION_FEATURES = {f"{c}_deviation" for c in
                      ["logon_count","usb_connect_count","file_copy_count",
                        "emails_total","emails_external","http_count",
                        "after_hours_logon","suspicious_url_count"]}


def display_name(feature):
    return DISPLAY_NAMES_GRAPH.get(feature, feature)


def feature_category(feature):
    if feature in STATIC_FEATURES:
        return "Static contextual"
    if feature in DEVIATION_FEATURES:
        return "Personalised deviation"
    return "Behavioural raw"


# Plain-language phrasing replacing original column name for the auto-generated explanation sentence
FRIENDLY_NAMES_PARA = {
    "logon_count": "logon activity",
    "logoff_count": "logoff activity",
    "after_hours_logon": "after-hours logons",
    "weekend_logon": "weekend logons",
    "unique_pcs": "number of PCs used",
    "usb_connect_count": "USB device connections",
    "usb_after_hours": "after-hours USB connections",
    "file_copy_count": "file copies to removable media",
    "file_copy_after_hours": "after-hours file copies",
    "emails_total": "email volume",
    "emails_external": "emails to external addresses",
    "emails_with_attachment": "emails with attachments",
    "emails_after_hours": "after-hours emails",
    "http_count": "web browsing volume",
    "suspicious_url_count": "visits to suspicious sites",
    "http_after_hours": "after-hours web browsing",
    "O": "Openness score",
    "C": "Conscientiousness score",
    "E": "Extraversion score",
    "A": "Agreeableness score",
    "N": "Neuroticism score",
    "role_encoded": "job role",
    "department_encoded": "department",
    "functional_unit_encoded": "functional unit",
}
# Make http_count_deviation and http_count same key, for plain language explanation purpose
for _c in DEVIATION_FEATURES:
    FRIENDLY_NAMES_PARA[_c] = FRIENDLY_NAMES_PARA.get(_c.replace("_deviation", ""), _c)

# Display labels for tables/charts, must be different from friendly names
# because http_count_deviation and http_count cannot share the same display name
DISPLAY_NAMES_GRAPH = {}
for _f, _n in FRIENDLY_NAMES_PARA.items():
    if _f in STATIC_FEATURES or _f.endswith("_deviation"):
        continue
    DISPLAY_NAMES_GRAPH[_f] = _n[0].upper() + _n[1:] #capitalize first character
for _c in DEVIATION_FEATURES:
    _base = FRIENDLY_NAMES_PARA.get(_c.replace("_deviation", ""), _c)
    DISPLAY_NAMES_GRAPH[_c] = _base[0].upper() + _base[1:] + " deviation"
DISPLAY_NAMES_GRAPH.update({
    "O": "Openness", 
    "C": "Conscientiousness", 
    "E": "Extraversion",
    "A": "Agreeableness", 
    "N": "Neuroticism",
    "role_encoded": "Job role", 
    "department_encoded": "Department",
    "functional_unit_encoded": "Functional unit",
})

#Turns SHAP values into explanation, explains on top 3 contributors
def build_explanation(user, threshold):
    shap_vals = user["shap_values"]
    feats = user["features"]
    #sorting from highest contribution to lowest, keep contributions more than 0.005
    positive = sorted(((f, s) for f, s in shap_vals.items() if s > 0.005),
                      key=lambda x: x[1], reverse=True)[:3]

    # category key used for CSS colour + tag label
    def _cat(f):
        if f in DEVIATION_FEATURES:
            return "deviation", "Personalised deviation"
        if f in STATIC_FEATURES:
            return "static", "Static contextual"
        return "behavioural", "Behavioural raw"

    items = []
    for f, s in positive:
        val = feats.get(f, 0)
        name = FRIENDLY_NAMES_PARA.get(f, f)
        cat_key_CSS, label = _cat(f)
        if cat_key_CSS == "deviation":
            direction = "above" if val >= 0 else "below"
            text = f"{name} was {abs(val):.2f} {direction} this employee's personal baseline"
        elif cat_key_CSS == "static":
            text = f"their {name}"
        else:
            text = f"{name} was high on this day at {val:.0f}"
        items.append({"cat_key": cat_key_CSS, "cat_label": label, "text": text, "shap": s})

    # To calculate how much static features contributed to the increase
    total_pos = sum(s for s in shap_vals.values() if s > 0) or 1.0
    static_pos = sum(s for f, s in shap_vals.items() if s > 0 and f in STATIC_FEATURES)
    static_percent = 100.0 * static_pos / total_pos

    score = user["risk_score"]
    flagged = score >= threshold

    # A day scoring near zero is normal even if one feature nudged upward by a little
    # Contributors are only listed if its a big number.
    if score < 0.05 or not items:
        return {"is_normal_day": True,
                "text": (f"{user['user_id']} scored {score:.3f}. No feature showed a meaningful "
                         f"increase in risk, thus this day looks like normal behaviour.")}

    if flagged:
        head = (f"{user['user_id']} was flagged with risk score {score:.3f}. "
                f"This is primarily caused by the following features:")
    else:
        head = (f"{user['user_id']} was not flagged with risk score {score:.3f}, below the "
                f"{threshold:.2f} threshold. The strongest features that cause increased risk were:")

    noun = "alert" if flagged else "result"
    if static_percent < 25:
        tail = (f"Static context (role, department, personality profile) contributed "
                f"only {static_percent:.0f}% of the increased risk, meaning this {noun} is caused by "
                f"the employee's behaviour rather than their identity.")
    else:
        tail = (f"Static context (role, department, personality profile) "
                f"contributed {static_percent:.0f}% of the increased risk on this day.")

    return {"is_normal_day": False, "flagged": flagged, "score": score,
            "head": head, "items": items, "tail": tail}

# Unstick words
def clean_label(text):
    text = str(text or "").strip()
    if not text or text in ("nan", "Unknown"):
        return "Unknown"
    text = re.sub(r"^\d+\s*-\s*", "", text).strip()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text or "Unknown"

# labels for dept, role and unit
# case A: label already in plain text
# case B: label encoded need decoding using saved label mapping
# case C: label is nan, unknown or empty, display unknown instead
def decode_labels(row):
    dept = str(row.get("department", "")).strip()
    role = str(row.get("role", "")).strip()
    unit = str(row.get("functional_unit", "")).strip()
    if not dept or dept in ("nan", "Unknown", ""):
        dept = DEPT_MAP.get(int(float(row.get("department_encoded", 0))), "Unknown")
    if not role or role in ("nan", "Unknown", ""):
        role = ROLE_MAP.get(int(float(row.get("role_encoded", 0))), "Unknown")
    if not unit or unit in ("nan", "Unknown", ""):
        unit = UNIT_MAP.get(int(float(row.get("functional_unit_encoded", 0))), "Unknown")
    return clean_label(dept), clean_label(role), clean_label(unit)

# -------------------------------------------------------------
# FLASK SETUP AND STORED DATA
# -------------------------------------------------------------
app = Flask(__name__)

# Flask needs a secret key before flash() messages will work.
# Secret key is generated randomly each time app starts
app.secret_key = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = MAX_RAW_UPLOAD

_artifacts = {
    "D": {"model": None, "scaler": None, "features": None, "explainer": None,
          "split": "Temporal split (2010 -> 2011)", "f1": 0.7820, "auc": 0.9996},
    "F": {"model": None, "scaler": None, "features": None, "explainer": None,
          "split": "Held-out users (never seen in training)", "f1": 0.7841, "auc": 0.9990},
}
_active_model = "D"
_threshold = DEFAULT_THRESHOLD
_last_df = None  # holds the last uploaded data, kept for model switching
_selected_user = None  # which employee's timeline is shown, the selector for which employee to show, cleared on a new upload
results_store = {}  # holds the results rows {"users": {row_key: entry}, "ranked": [entry], "meta": {...}}

# Default is D unless explicitly stated as F
# Load and cache saved files and SHAP explainer for both model D or F
# lazy loading utilized
def load_models(model_key="D"):
    art = _artifacts[model_key]
    if art["model"] is None:
        print(f"  Loading Model {model_key} ............")
        art["model"] = joblib.load(os.path.join(MODELS_FOLDER, f"model_{model_key}.pkl"))
        art["scaler"] = joblib.load(os.path.join(MODELS_FOLDER, f"scaler_{model_key}.pkl"))
        art["features"] = list(joblib.load(os.path.join(MODELS_FOLDER, f"features_{model_key}.pkl")))
        art["explainer"] = shap.TreeExplainer(art["model"])
        print(f"  Model {model_key} loaded with {len(art['features'])} features")
    return art

# -------------------------------------------------------------
# SHAP HELPERS
# -------------------------------------------------------------
# Show value of how much each feature pushed the risk score up or down
def shap_class1(explainer, X_Feature_Cols_32_Scaled):
    vals = explainer.shap_values(X_Feature_Cols_32_Scaled)  # feature contribution for every row
    ev = explainer.expected_value  # the average score before any feature counts

    # Extract the base value. Some SHAP versions return one number per class,
    # others return a single number. Take the malicious one when there is a pair.
    # The second value of the pair is the malicious one, as designed non-malicious values comes first.
    if hasattr(ev, "__len__"):
        base = float(ev[1])
    else:
        base = float(ev)

    # Extract the contributions scores, which arrive in one of three shapes depending
    # on the SHAP version. Each shape needs the malicious class picking out.
    if isinstance(vals, list):  # a list holding one array per class (take second value(malicous))
        return np.array(vals[1]), base
    vals = np.array(vals)
    if vals.ndim == 3:  # one block holding both classes together (take sheet 1 (malicious))
        return vals[:, :, 1], base
    return vals, base  # single class, take the whole sheet directly

# Make a bar chart from SHAP, showing top 10
def make_shap_plot(user_id, features, shap_vals_all, feature_vals_all, base_value):

    COL_UP = "#c0392b"  # pushes risk UP   (positive SHAP)
    COL_DOWN = "#2e9e5b"  # pushes risk DOWN (negative SHAP)

    order = sorted(range(len(shap_vals_all)), key=lambda i: abs(shap_vals_all[i]), reverse=True) [:10][::-1] #get values, sort biggest to smallest, take top 10, reverse the list since matplotlib take bottom to top
    names = [display_name(features[i]) for i in order]
    shap_vals_plot = [float(shap_vals_all[i])    for i in order]
    feature_vals_plot = [float(feature_vals_all[i]) for i in order]
    colours = [COL_UP if v > 0 else COL_DOWN for v in shap_vals_plot]
    max_abs = max(abs(v) for v in shap_vals_plot) if shap_vals_plot else 1.0 # for padding size

    fig, axes = plt.subplots(figsize=(9.2, 5.2)) # set figuresize and axis
    fig.patch.set_facecolor("white") # rectangle background colour white
    axes.set_facecolor("#fffdf9") # colour behind area of axis
    axes.set_axisbelow(True) # gridlines drawn behind data

    axes.barh(range(len(names)), shap_vals_plot, color=colours, height=0.62, edgecolor="white", linewidth=0.7, alpha=0.93, zorder=2)

    for i, (v, a) in enumerate(zip(shap_vals_plot, feature_vals_plot)): # [(0, (-0.0204, 2.0)), (1, (0.0027, 11.0)), (2, (0.1626, 1.367)), ...]
        padding = max_abs * 0.03
        x_position = (v + padding) if v >= 0 else padding  # negative bars: label right of zero
        axes.text(x_position, i, f"{a:.2f}   ≈  {v:+.4f}", va="center", ha="left",
                fontsize=8.5, color=(COL_UP if v > 0 else COL_DOWN), fontweight="normal")

    # setting the left and right side tint colours
    x_limits = axes.get_xlim()
    axes.axvspan(0, x_limits[1], color=COL_UP,   alpha=0.035, zorder=0)
    axes.axvspan(x_limits[0], 0, color=COL_DOWN, alpha=0.045, zorder=0)
    axes.set_xlim(x_limits)

    # setting the title and the label for each bar
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(names, fontsize=9.5, color="#3a2c17")
    axes.axvline(0, color="#8a7a5f", linewidth=1.0, zorder=3)
    axes.set_xlabel("SHAP value: positive pushes risk up, negative pushes risk down",
                  fontsize=9, color="#6b7280")
    axes.set_title(f"SHAP feature contributions for {user_id} (base value {base_value:.2f})",
                 fontsize=12, color="#3a2c17", fontweight="bold", pad=12)
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.spines["bottom"].set_color("#ddceb4")
    axes.tick_params(left=False, colors="#6b7280")
    axes.grid(axis="x", color="#ece0cc", linewidth=0.6, zorder=1)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# -------------------------------------------------------------
# CORE SCORING
# -------------------------------------------------------------
# Score every row and compute SHAP values
# Chart are not generated for all results,
# on first visit to the detail page, it is only generated then cached
def process_dataframe(df, model_key="D"):

    # loads the saved artefacts, .pkl, scaler and feature
    art = load_models(model_key)
    model, scaler, features = art["model"], art["scaler"], art["features"]

    # checks if there is missing columns
    # done by checking against the saved feature file
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"These required columns are missing from your CSV: {missing}. "
            f"Check that you uploaded the correct feature table.")

    # Limit rows to only 2,000, to protect application from hang
    # This is because SHAP holds all computed results in memory
    # making a copy to remove pandas warning in editing partial view
    truncated = False
    if len(df) > MAX_ROWS:
        df, truncated = df.head(MAX_ROWS).copy(), True

    # Selects the features sorted in the same way it was trained, drop everything else
    X_Feature_Cols_32 = df[features].copy()

    # Missing psychometric scores will be set to 30, the middle of the 10-50 range the
    # traits actually use, same as the raw-log path. A 0 or a 50 here would be read by
    # the model as an extreme trait score.
    psy = [c for c in ("O", "C", "E", "A", "N") if c in X_Feature_Cols_32.columns]
    if psy:
        X_Feature_Cols_32[psy] = X_Feature_Cols_32[psy].fillna(30)
    # Any remaining blank is an activity count replace as 0
    X_Feature_Cols_32 = X_Feature_Cols_32.fillna(0)
    X_Feature_Cols_32_Scaled = scaler.transform(X_Feature_Cols_32.values) #use the same scaler trained, .values to remove sklearn warning

    # Returns the probability of malicious value, and computes the contributions for every row
    proba = model.predict_proba(X_Feature_Cols_32_Scaled)[:, 1]
    shap_vals, base = shap_class1(art["explainer"], X_Feature_Cols_32_Scaled)

    # user dict to store user info, ranked list to sort user dict, meta to store info of whole uploaded batch
    store = {"users": {}, "ranked": [], "meta": {
        "model_key": model_key, "truncated": truncated,
        "model_f1":  _artifacts[model_key]["f1"],
        "model_auc": _artifacts[model_key]["auc"],
    }}

    for index in range(len(df)):  # go through the uploaded rows line by line
        row = df.iloc[index] # take one row from the uploaded file
        user_id = str(row.get("user", f"ROW_{index:04d}")) # take user id from the user id column, if no invent one
        dept, role, unit = decode_labels(row) # take the dept, role and unit
        row_shap = shap_vals[index] # take the 32 contribution values from the computed batch
        top_feat = features[int(np.argmax(np.abs(row_shap)))] # find the max value ignoring +/- sign

        # dictionary to store info of single row
        entry = {
            "row_key": str(index), # row number as string
            "user_id": user_id, # user id
            "date": str(row.get("date_only", ""))[:10], # date but keep only first 10, to remove time
            "risk_score": round(float(proba[index]), 4), # risk score in 4 decimals
            "department": dept, # department
            "role": role, # role
            "unit": unit, # unit
            "top_factor": DISPLAY_NAMES_GRAPH.get(top_feat, top_feat), # get the top feature's plain language name
            "model_label": f"Model {model_key}", # Model version
            "features": {f: round(float(v), 3) for f, v in zip(features, X_Feature_Cols_32.iloc[index].values)}, # put feature name and its number of event happened side by side
            "shap_values": {f: round(float(v), 4) for f, v in zip(features, row_shap)}, # put feature name and its shap value side by side
            "shap_base": round(base, 4), # the starting score of shap
            "shap_img": None, # created only on first detail view
            "O": round(float(row.get("O", 0)), 2), # Openness Score
            "C": round(float(row.get("C", 0)), 2), # Conscientiousness Score
            "E": round(float(row.get("E", 0)), 2), # Extraversion Score
            "A": round(float(row.get("A", 0)), 2), # Agreeableness Score
            "N": round(float(row.get("N", 0)), 2), # Neuroticism Score
        }
        store["users"][entry["row_key"]] = entry # Store the row in the storage

    store["ranked"] = sorted(store["users"].values(), key=lambda e: e["risk_score"], reverse=True) # take all the stored rows out and sort
    return store

# the summary card on top of result page
def build_summary(ranked, meta, threshold):
    flagged = [u for u in ranked if u["risk_score"] >= threshold] # identify flagged rows which depends on threshold, >= threshold
    unique_users = len({u["user_id"] for u in ranked}) # {} is set, prevent duplicates
    flagged_depts = len({u["department"] for u in flagged}) # put into set to count the number of unique departments flagged
    return {
        "total": len(ranked), # Number of rows
        "unique_users": unique_users, # No of unique employees
        "is_single_user": unique_users == 1, # True/False
        "flagged": len(flagged), # Number of flagged rows
        "high_risk": sum(1 for u in flagged if u["risk_score"] >= 0.80), # Number of high risk if set at above 0.80
        "threshold": threshold, # threshold
        "flagged_depts": flagged_depts, # flagged depts hold the value of number of flagged unique departments
        "model_key": meta["model_key"], # D / F
        "model_label": f"Model {meta['model_key']}", # Model D / Model F
        "model_f1": meta["model_f1"], # F1 score
        "model_auc": meta["model_auc"], # AUC Score
        "truncated": meta["truncated"], # Whether file was cut at 2,000 rows
    }

# -------------------------------------------------------------
# TEMPLATE GLOBALS
# -------------------------------------------------------------
@app.context_processor
# inject required stuff into template without routing
def inject_template_globals():
    return {"DISPLAY_NAMES_GRAPH": DISPLAY_NAMES_GRAPH,
            "ALLOWED_THRESHOLDS": ALLOWED_THRESHOLDS}

# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
# default route to upload page
@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/analyze", methods=["POST"]) # accepts only POST request (submitting a form)
def analyze():
    global results_store, _last_df, _active_model, _selected_user # set and hold as global variables, so it exists outside of the function
    if "csv_file" not in request.files or request.files["csv_file"].filename == "": # Check if file was uploaded, if no file or empty filename, return error message
        flash("Please select a CSV file before uploading.", "error")
        return redirect(url_for("upload"))
    try:
        df = pd.read_csv(request.files["csv_file"]) # Read the uploaded file into a table
        _last_df = df.copy() # copy the uploaded file/table
        _active_model = "D"  # Default model is D
        _selected_user = None  # a new upload clears any timeline selection
        results_store = process_dataframe(df, _active_model) # run the function, to scale, score, explain and rank
        if results_store["meta"]["truncated"]: # display message if over 2,000 rows
            flash(f"File exceeded {MAX_ROWS} rows - analysis limited to the first {MAX_ROWS}.",
                  "error")
        return redirect(url_for("results")) #redirect to the results page
    except ValueError as e: # display column missing error message if happens
        flash(str(e), "error")
        return redirect(url_for("upload")) # redirect back to upload page
    except Exception as e: # Catch any other errors
        flash(f"Unexpected error: {e}", "error")
        return redirect(url_for("upload"))


# Raw log filenames to match CERT style
RAW_LOG_KINDS = ("logon", "device", "file", "email", "http")

# The web application accepts raw logs that resembles the CERT format, Psychometric and LDAP will need to be provided or it will fall back to defaults to prevent errors
@app.route("/analyze_raw", methods=["POST"])
def analyze_raw():
    global results_store, _last_df, _active_model, _selected_user  # set and hold as global variables, so it exists outside of the function
    import tempfile, shutil, zipfile # import done here because only used within this function, 'tempfile' makes temp folder, 'shutil' copies and delete folders, and 'zipfile' extracts zip files

    files = []
    for f in request.files.getlist("raw_files"):
        if f and f.filename:
            files.append(f) # collect uploaded files, only those with names

    if not files: # If no files, show the error message, then redirect back to upload page
        flash("Please select raw log CSVs (logon/device/file/email/http)", "error")
        return redirect(url_for("upload"))

    tmp = tempfile.mkdtemp(prefix="rawlogs_") # make a temporary folder to hold temp files

    try:
        # Save uploads - unzip archives, map recognised names onto <kind>.csv
        for f in files:
            name = os.path.basename(f.filename).lower() # rename file into lowercase
            if name.endswith(".zip"): # if its a zip file, save and unpack everything in the temp folder
                zip_path = os.path.join(tmp, "bundle.zip")
                f.save(zip_path)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp)
            else:
                kind = next((k for k in RAW_LOG_KINDS if k in name), None) #if not a zip file, check if there is any files that match the raw log list
                if kind is None: #if nothing, display error
                    flash(f"Skipped '{f.filename}' - name must contain one of {RAW_LOG_KINDS}.", "error") # maybe remove this error message
                    continue
                f.save(os.path.join(tmp, f"{kind}.csv")) # save those files if there is at least one usable

        # Normalise anything the zip extracted into nested folders
        for root, _dirs, fnames in os.walk(tmp): # os.walk is used to go through every folder and look for files
            for file_name in fnames: # for every filename
                low = file_name.lower() # lowercase it
                kind = next((k for k in RAW_LOG_KINDS if k in low and low.endswith(".csv")), None) # match with the raw file list
                if kind and root != tmp: # if there is a matching filename, and root != tmp
                    os.replace(os.path.join(root, file_name), os.path.join(tmp, f"{kind}.csv")) # move current file to tmp folder

        # all five CERT logs must be present
        missing_logs = [f"{k}.csv" for k in RAW_LOG_KINDS if not os.path.exists(os.path.join(tmp, f"{k}.csv"))] # collect any of the five that is not in the temp folder
        if missing_logs: # if the set is incomplete
            flash(f"These required log files are missing from your upload: {missing_logs}. The system only accepts a complete set of CERT r4.2 logs.", "error") # display error
            return redirect(url_for("upload")) # redirect back to upload page

        # Static context: prefer context files bundled IN the upload (self-
        # contained bundles like the 2026 scenario ship their own LDAP +
        # psychometric); fill any missing piece from the app's context_data.
        if os.path.isdir(CONTEXT_DATA_FOLDER): # If there is context data folder
            if not os.path.isdir(os.path.join(tmp, "LDAP")) and os.path.isdir(os.path.join(CONTEXT_DATA_FOLDER, "LDAP")): # if no LDAP provided, copy the LDAP from context data folder to temp folder
                shutil.copytree(os.path.join(CONTEXT_DATA_FOLDER, "LDAP"), os.path.join(tmp, "LDAP"))
            if not os.path.exists(os.path.join(tmp, "psychometric.csv")) and os.path.exists(os.path.join(CONTEXT_DATA_FOLDER, "psychometric.csv")): # if no psychometric provided, 
                                                                                                                                                    # copy the psychometric from context data folder to temp folder
                shutil.copy(os.path.join(CONTEXT_DATA_FOLDER, "psychometric.csv"), tmp)


        # Function to Build Feature Table, defined in preprocessing script, same as feature engineering
        feature_df = build_feature_table(
            tmp,
            maps_path=os.path.join(MODELS_FOLDER, "label_maps.pkl"),
            context_dir=tmp,
            log=lambda *a, **k: None, # lambda to hide all logs outputs in the preprocessing script
        )

        # If the preprocessed produced no rows, display error message, and redirect to upload page
        if len(feature_df) == 0:
            flash("Preprocessing produced no rows - check the log files' contents.", "error")
            return redirect(url_for("upload"))

        # Score exactly like a feature-table upload
        _last_df = feature_df.copy() # set as the latest data frame
        _active_model = "D" # the active model
        _selected_user = None # a new upload clears any timeline selection
        results_store = process_dataframe(feature_df, _active_model) # pass parameter and initiate scaling, scoring, explaining and ranking
        flash(f"Raw logs preprocessed into {len(feature_df):,} employee-day rows and scored.", "success") # Display success message
        return redirect(url_for("results")) # redirect to the results page
    except ValueError as e:
        flash(str(e), "error") # show message that came with the error
        return redirect(url_for("upload")) # redirect to upload page
    except Exception as e: # to handle unexpected errors
        flash(f"Unexpected error while preprocessing raw logs: {e}", "error") # display error message
        return redirect(url_for("upload")) #redirect to upload page
    finally:
        shutil.rmtree(tmp, ignore_errors=True) # Always runs no matter, delete the tmp folder along with its contents


@app.route("/switch/<model_key>") # model key reflects the current model selected 
# Re-score the last uploaded data with the selected model
def switch_model(model_key):
    global results_store, _active_model # variables used within the function reflects changes outside 
    if model_key not in ("D", "F"): # check if selected model is true
        return redirect(url_for("results")) # else return to the result page 
    if _last_df is None: # If there is no latest dataframe, redirect to upload page to upload 
        flash("Please upload a file first.", "error")
        return redirect(url_for("upload"))
    try:
        _active_model = model_key #the selected model is the active model
        results_store = process_dataframe(_last_df.copy(), _active_model) # make a copy of the latest df and score it again
        return redirect(url_for("results")) # redirect to the results page after scoring 
    except Exception as e: # Add error handling in case something unexpected happens 
        flash(f"Error switching model: {e}", "error")
        return redirect(url_for("results"))

# to change the threshold value
@app.route("/threshold/<value>")
def set_threshold(value): # values (0.3/0.5/0.7)
    global _threshold # allow changes to be applied globally
    try:
        Thres_Val = round(float(value), 2) #round to 2 decimals 
        if Thres_Val in ALLOWED_THRESHOLDS: # if Thres_Val is one of the preset values 
            _threshold = Thres_Val # set Thres_Val as the new threshold 
    except ValueError: # ignore if other than the 3 values provided 
        pass
    return redirect(url_for("results")) # redirect to results page


# The results page
@app.route("/results")
def results():
    if not results_store: #if there are nothing uploaded, redirect back to upload page
        return redirect(url_for("upload"))
    ranked = results_store["ranked"] # take the rows that are sorted from highest to lowest risk 
    for u in ranked:  # for every row in ranked
        u["flagged"] = u["risk_score"] >= _threshold # update the flagged status for all rows with the new threshold
    summary = build_summary(ranked, results_store["meta"], _threshold) # send info to update the summary block

    # Only employees with more than one day can have a timeline drawn
    day_counts = {}
    for u in ranked:
        day_counts[u["user_id"]] = day_counts.get(u["user_id"], 0) + 1 # go through the ranked list to see which employee has more than 1 day 
    timeline_users = sorted(uid for uid, n in day_counts.items() if n > 1) # a list of sorted employees in alphabatical order, where they have more than 1 day

    # Which employee to chart. A choice from the dropdown is remembered, so changing
    # the threshold or the model does not clear it. A single-employee upload selects
    # itself, which keeps the original behaviour unchanged.
    global _selected_user
    if "employee" in request.args: #if the employee id is in "results?employee=CCL0068" use that employee id
        _selected_user = request.args.get("employee") 
    if _selected_user not in timeline_users: # if its not in the list, set as none
        _selected_user = None
    if _selected_user is None and summary["is_single_user"] and timeline_users: # if its a single user, and more than 1 day use that employee 
        _selected_user = timeline_users[0]

    timeline = None #set timeline to none as default
    if _selected_user: # build the chart from that employee's days only
        timeline = sorted(
            ({"date": u["date"], 
              "score": u["risk_score"],  
              "flagged": u["flagged"],  
              "row_key": u["row_key"]} for u in ranked if u["user_id"] == _selected_user),
            key=lambda p: p["date"]) # create a small dictionary for plotting the timeline graph, row_key is used to make points clickable, key=lambda .. ["date"] is to sort dict by date

    return render_template("results.html",
                           summary=summary,
                           ranked=ranked,
                           timeline=timeline,
                           timeline_users=timeline_users,
                           selected_user=_selected_user,
                           active_model=_active_model,
                           active_threshold=_threshold) # Hand over information needed to load the results.html



# The user detail page
@app.route("/user/<row_key>")
def user_detail(row_key): 
    if not results_store or row_key not in results_store["users"]: # if there are no results, or entered a row that does not exist redirect back to results
        return redirect(url_for("results"))
    
    user = results_store["users"][row_key] # retrieve the row using its row key of selected row 

    # Lazy SHAP chart render, computed once per row, when it is clicked, then cached
    if user["shap_img"] is None: # if there are no shap images then produce it 
        art = load_models(results_store["meta"]["model_key"]) # load the selected model D/F
        features = art["features"] # load the feature file D/F
        user["shap_img"] = make_shap_plot(
            user["user_id"], # user id
            features, # features in order 
            np.array([user["shap_values"][f] for f in features]), # find the shap values for features in order
            np.array([user["features"][f]    for f in features]), # find the feature values for features in order
            user["shap_base"]) 

    # construct a table that displays the shap and feature value for each feature 
    shap_table = sorted(
        [{"feature": f, # f here is feature name 
          "shap": user["shap_values"][f], # shap value of a feature
          "value": user["features"][f], # feature value of a feature 
          "category": feature_category(f)} # category of a feature
         for f in user["shap_values"]], # for every 32 feature 
        key=lambda x: abs(x["shap"]), reverse=True)[:12] # for every row in this list, sort it by absolute value of shap, reverse=True because is lowest to highest by default, and keep top 12 

    #return everything that needs to be known to load user detail page
    return render_template("user_detail.html", 
                           user=user,
                           shap_table=shap_table,   
                           active_threshold=_threshold,  
                           explanation=build_explanation(user, _threshold))  

# About Page Route
# Return necessary info to load about page 
@app.route("/about") 
def about():
    return render_template("about.html",
                           model_d=_artifacts["D"], model_f=_artifacts["F"])

# -------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    print("x" * 60)
    print("Context-Aware Anomaly Detection Prototype (ㆆ_ㆆ)")
    print("x" * 60)
    load_models("D")
    print("\nAccess the Web Application at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
