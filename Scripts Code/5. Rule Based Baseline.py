"""
CERT r4.2 - Rule-Based Baseline
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
FEATURE_TABLE_LOCATION = os.path.join(BASE_FOLDER, "feature_table_final.csv") # feature table location
OUTPUT_LOCATION = os.path.join(BASE_FOLDER, "results", "rule_based") # where results are written
os.makedirs(OUTPUT_LOCATION, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2011-01-01") # same split date as configurations C and D

# The machine learning figures are read from the ablation output instead of being
# typed in here, so this table can never quote a stale number.
ABLATION_RESULTS_LOCATION = os.path.join(BASE_FOLDER, "results", "ablation", "ablation_results.csv")


AFTER_HOURS_LOGON_WITH_USB = 1 # after-hours logons before the USB rule applies
FILE_COPY_LIMIT = 10 # copies to removable media in one day
SUSPICIOUS_URL_LIMIT = 5 # suspicious sites visited in one day
USB_CONNECT_LIMIT = 3 # removable device connections in one day
AFTER_HOURS_VOLUME_LIMIT = 2 # after-hours logons on their own

            
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("CERT r4.2 - Rule-Based Baseline")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# -------------------------------------------------------------
# STEP 1 - Load the feature table and split it by date
# -------------------------------------------------------------
NEEDED_COLUMNS = ["user", "date_only", "label", "after_hours_logon", "weekend_logon",
                  "usb_connect_count", "file_copy_count", "suspicious_url_count"] # the needed columns to score
feature_table = pd.read_csv(FEATURE_TABLE_LOCATION, low_memory=False, usecols=NEEDED_COLUMNS) # read feature table and load it only with the needed columns
feature_table["date_only"] = pd.to_datetime(feature_table["date_only"]) # turn the text date into a real date

test_days = feature_table[feature_table["date_only"] >= SPLIT_DATE] # 2011, the same test period as C and D in ablation
y_test = test_days["label"].values.astype(int) # the true answer for each test day

print("Test period rows :", len(test_days)) # Test period rows: 84029
print("Malicious days :", int(y_test.sum())) # Malicious days : 244
print("Split date :", SPLIT_DATE.date()) # Split date : 2011-01-01


# -------------------------------------------------------------
# STEP 2 - Apply the six rules
# Each rule produces one True/False column. A day is flagged if ANY rule fires.
# -------------------------------------------------------------
print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("THE RULES")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Each rule becomes one True/False column covering every test day.
# The & in R1 and R5 means AND, so both conditions have to hold.
rule_columns = []

# R1 - after-hours working combined with a removable device
rule_columns.append(((test_days["after_hours_logon"] >= AFTER_HOURS_LOGON_WITH_USB) & (test_days["usb_connect_count"] >= 1)).values)

# R2 - heavy copying to removable media
rule_columns.append((test_days["file_copy_count"] >= FILE_COPY_LIMIT).values)

# R3 - visiting job sites and other suspicious destinations
rule_columns.append((test_days["suspicious_url_count"] >= SUSPICIOUS_URL_LIMIT).values)

# R4 - repeated removable device use
rule_columns.append((test_days["usb_connect_count"] >= USB_CONNECT_LIMIT).values)

# R5 - working at the weekend and copying files
rule_columns.append(((test_days["weekend_logon"] >= 1) & (test_days["file_copy_count"] >= 1)).values)

# R6 - a general rise in after-hours activity
rule_columns.append((test_days["after_hours_logon"] >= AFTER_HOURS_VOLUME_LIMIT).values)

print("R1 after-hours with USB :", "after_hours_logon >=", AFTER_HOURS_LOGON_WITH_USB, "AND usb_connect_count >= 1")
print("R2 file copies          :", "file_copy_count >=", FILE_COPY_LIMIT)
print("R3 suspicious sites     :", "suspicious_url_count >=", SUSPICIOUS_URL_LIMIT)
print("R4 USB connections      :", "usb_connect_count >=", USB_CONNECT_LIMIT)
print("R5 weekend copying      :", "weekend_logon >= 1 AND file_copy_count >= 1")
print("R6 after-hours volume   :", "after_hours_logon >=", AFTER_HOURS_VOLUME_LIMIT)
#print(rule_columns)


# go through every day and flag it if any of the six rules fired on that day
y_predicted = []
for row_number in range(len(test_days)): # for every row tested
    flagged = 0 # assume the day is normal
    for rule_column in rule_columns: #for every rule column of that row
        if rule_column[row_number]: # this rule fired on this day
            flagged = 1 #flagged = 1 if any of the rule is fired 
    y_predicted.append(flagged) # one answer per day

#print(y_predicted)

y_predicted = np.array(y_predicted) # transform into array, because sklearn uses array instead of list

#print(y_predicted)


# -------------------------------------------------------------
# STEP 3 - Compare what the rules said against the real answer
# -------------------------------------------------------------
rule_f1 = f1_score(y_test, y_predicted, zero_division=0) #zero deivision = 0 to allow to be divided by 0 and return no error 
rule_precision = precision_score(y_test, y_predicted, zero_division=0)
rule_recall = recall_score(y_test, y_predicted, zero_division=0)
days_flagged = int(y_predicted.sum()) # sum of days flagged by rule 

print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("ALL SIX RULES TOGETHER")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Days flagged :", days_flagged, "out of", len(test_days))
print("F1 :", round(rule_f1, 4))
print("Precision :", round(rule_precision, 4))
print("Recall :", round(rule_recall, 4))
print("")


# -------------------------------------------------------------
# STEP 4 - Compare against the machine learning configurations
# -------------------------------------------------------------
print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("COMPARISON WITH THE MACHINE LEARNING MODELS")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

if not os.path.exists(ABLATION_RESULTS_LOCATION):
    raise FileNotFoundError(
        "ablation_results.csv was not found. Run '4. Ablation Study.py' first, "
        "because the machine learning figures in this comparison are read from "
        "that file rather than typed into this one.")

ablation_results = pd.read_csv(ABLATION_RESULTS_LOCATION)

comparison_rows = []
comparison_rows.append({"label": "R0", "name": "Rule-based",
                        "f1": rule_f1, "precision": rule_precision, "recall": rule_recall})

for config_label, config_name in [("C", "Machine learning, no context"),
                                  ("D", "Machine learning, with context")]:
    matching_row = ablation_results[ablation_results["label"] == config_label]
    comparison_rows.append({"label": config_label, "name": config_name,
                            "f1": float(matching_row["f1"].iloc[0]),
                            "precision": float(matching_row["precision"].iloc[0]),
                            "recall": float(matching_row["recall"].iloc[0])})

for row in comparison_rows:
    print(row["label"], "-", row["name"])
    print("   F1        :", round(row["f1"], 4))
    print("   Precision :", round(row["precision"], 4))
    print("   Recall    :", round(row["recall"], 4))


# -------------------------------------------------------------
# STEP 5 - Save the results
# -------------------------------------------------------------
print("")
print("Saving results ...")

pd.DataFrame(comparison_rows).to_csv(os.path.join(OUTPUT_LOCATION, "rule_baseline_results.csv"), index=False)

report_file = open(os.path.join(OUTPUT_LOCATION, "rule_baseline_report.txt"), "w", encoding="utf-8")
report_file.write("Rule-Based Baseline - CERT r4.2, test period from " + str(SPLIT_DATE.date()) + "\n")
report_file.write("x" * 60 + "\n\n")
report_file.write("Thresholds used (operational assumptions, not taken from any publication):\n")
report_file.write("after-hours logons before the USB rule applies : " + str(AFTER_HOURS_LOGON_WITH_USB) + "\n")
report_file.write("file copies to removable media in one day : " + str(FILE_COPY_LIMIT) + "\n")
report_file.write("suspicious sites visited in one day : " + str(SUSPICIOUS_URL_LIMIT) + "\n")
report_file.write("removable device connections in one day : " + str(USB_CONNECT_LIMIT) + "\n")
report_file.write("after-hours logons on their own : " + str(AFTER_HOURS_VOLUME_LIMIT) + "\n\n")
report_file.write("Days flagged: " + str(days_flagged) + " out of " + str(len(test_days)) + "\n\n")
report_file.write("Results on the test period:\n")
for row in comparison_rows:
    report_file.write("  " + row["label"] + "  " + row["name"] + "\n")
    report_file.write("      F1 " + str(round(row["f1"], 4)) +
                      "   Precision " + str(round(row["precision"], 4)) +
                      "   Recall " + str(round(row["recall"], 4)) + "\n")
report_file.close()

print("Saved:", os.path.join(OUTPUT_LOCATION, "rule_baseline_results.csv"))
print("Saved:", os.path.join(OUTPUT_LOCATION, "rule_baseline_report.txt"))

print("")
print("xxxxxx")
print("DONE")
print("xxxxxx")
