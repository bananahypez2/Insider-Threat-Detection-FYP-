"""
Preprocess_Raw_Logs.py - Raw log -> feature-table converter
"""

import os # used for file paths and folder check 
import argparse # specifically used for debugging, allow arguments to be passed in cli
import warnings
warnings.filterwarnings("ignore") #remove pandas warning messages 

import numpy as np # to work with values
import pandas as pd # pandas act as dataframe, like excel in coding
import joblib # to load files

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__)) # full path to the folder of the script

# Same constants as "2. Feature Engineering.py"
START_WORK_HOUR = 8 
END_WORK_HOUR = 18

SUSPICIOUS_KEYWORDS = [
    "career", "dropbox", "glassdoor", "indeed", "job",
    "linkedin", "monster", "resume", "usajobs", "wikileaks"
]

CHUNK_SIZE = 100000

FEATURE_ORDER = [
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

DEVIATION_SOURCES = [
    "logon_count", 
    "usb_connect_count", 
    "file_copy_count",
    "emails_total", 
    "emails_external", 
    "http_count",
    "after_hours_logon", 
    "suspicious_url_count",
]

# clean date aspect
def parse_dates(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce") # convert into real date instead of text, 
                                                             #  error="coerce" to fill those that cant be data as Nat (not a time)
    df["date_only"] = df["date"].dt.date # drop time and keeps the date 
    df["hour"] = df["date"].dt.hour # keep only the hour 
    df["is_weekend"] = df["date"].dt.dayofweek >= 5 # idenitfy weekends 
    df["is_after_hours"] = (df["hour"] < START_WORK_HOUR) | (df["hour"] >= END_WORK_HOUR) # after hours before start, and after end
    return df # return cleaned data aspect 

# Read the raw log in set chunk, aggregate each chunk by user and day, then combine
def chunked_agg(path, agg_builder, label, log=print): # file to be read, agg file type, label name, and where to send current progress
    if not os.path.exists(path): # if file is not found 
        log(f"  [skip] {label}: file not found ({path})") # log that the file was not found
        return None # then return nothing
    parts = [] # a temp empty list to collect results for each chunck 
    for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE, on_bad_lines="skip"): # for every 100,000 rows, 
                                                                                                # skip bad lines, low memory = false forces pandas to look at 
                                                                                                # the whole chunk before 
                                                                                                # deciding the data type of the column, thus avoiding mix type
        chunk = parse_dates(chunk) # tidy the date aspect 
        if len(chunk) == 0: # skip the chunk if its empty 
            continue
        parts.append(agg_builder(chunk)) # put the rows that has their date aspect dealt with into parts list 
    if not parts: # if nothing is in parts list
        log(f"  [warn] {label}: no usable rows found") # meaning no rows with dates are found 
        return None
    out = pd.concat(parts).groupby(["user", "date_only"]).sum().reset_index() # combine the parts grouped by user then date, 
                                                                                #sum is to sum up the days due to seperation between chunks 
    log(f"  {label}: {out.shape[0]:,} user-days") # how many user days was concated 
    return out


# logon.csv 
def agg_logon(chunk):
    logon = chunk[chunk["activity"] == "Logon"].groupby(["user", "date_only"]).agg( # selects only logon activity, groupby user, and under it their dates 
        logon_count=("id", "count"), # id is something temp to count, id is row 
        after_hours_logon=("is_after_hours", "sum"), # add up how many is after hours 
        weekend_logon=("is_weekend", "sum"), # add up how many is log in at weekend 
        unique_pcs=("pc", "nunique"), # number of unique pc used/logged on 
    )
    logoff = chunk[chunk["activity"] == "Logoff"].groupby(["user", "date_only"]).agg( # same thing as above but with log off
        logoff_count=("id", "count"),
    )
    return logon.join(logoff, how="outer").fillna(0).reset_index() # .join used to combine both based on index (user and date), =outer to keep both days that are missing either log on or log off, meaning keep everything, .reset puts the index back as columns

# device.csv
def agg_device(chunk):
    return chunk[chunk["activity"] == "Connect"].groupby(["user", "date_only"]).agg( #select connect only
        usb_connect_count=("id", "count"), # count number of events 
        usb_after_hours=("is_after_hours", "sum"), # sum up if its after hours 
    ).reset_index() # reset index to column 

# file.csv
def agg_file(chunk):
    return chunk.groupby(["user", "date_only"]).agg( #group by user then date
        file_copy_count=("id", "count"), # count file copies
        file_copy_after_hours=("is_after_hours", "sum"), # sum of file copies after hours
    ).reset_index()

# email.csv
def agg_email(chunk):
    if "to" in chunk.columns: # check if the column has to field 
        chunk["have_external"] = chunk["to"].astype(str).apply( # blanks cells arrive as number so have to convert to string first 
            lambda x: int(any( # for every email address, if has any external email add then score return True else false, int () turns it into 1 and 0 
                addr.strip() and "@dtaa.com" not in addr.lower() #exclude those that have internall email add
                for addr in x.replace(";", ",").split(",") # split but , instead of ; 
                if addr.strip() not in ("nan", "") # skip the blank ones 
            ))
        )
    else:
        chunk["have_external"] = 0 # if no this column set to 0 (False)

    if "attachments" in chunk.columns: # if there is attachement columns 
        chunk["have_attachment"] = (
            pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0) > 0 # first coerce to hanlde errors by filling in Nan, while fill na then put as 0, and if result > 0
        ).astype(int) # convert type to int
    else:
        chunk["have_attachment"] = 0 # if no attachment colums at all then 0 
    return chunk.groupby(["user", "date_only"]).agg( #group by user then date and combine
        emails_total=("id", "count"), # count number of emails 
        emails_external=("have_external", "sum"), # sum of emails to external 
        emails_with_attachment=("have_attachment", "sum"), # sum of attachemnt
        emails_after_hours=("is_after_hours", "sum"), # sum of emails after hours 
    ).reset_index()

# http.csv 
def agg_http(chunk):
    if "url" in chunk.columns: # if ther is url column
        chunk["is_suspicious_url"] = chunk["url"].astype(str).apply( # turn into string first
            lambda u: 1 if any(d in u.lower() for d in SUSPICIOUS_KEYWORDS) else 0 # if any keyword matches the suspicious domains, flag as 1
        )
    else:
        chunk["is_suspicious_url"] = 0 # else of no url column set as 0 
    return chunk.groupby(["user", "date_only"]).agg( # group by user then date
        http_count=("id", "count"), # count number of http columns 
        suspicious_url_count=("is_suspicious_url", "sum"), # sum of suspicious url 
        http_after_hours=("is_after_hours", "sum"), # sum of after hours browsing 
    ).reset_index()

# Convert into 32 feature table 
def build_feature_table(input_dir, maps_path=None, context_dir=None, log=print):

    if maps_path is None: # if not supplied
        maps_path = os.path.join(BASE_FOLDER, "models", "label_maps.pkl") # use default path
    if context_dir is None: # if not supplied
        context_dir = input_dir # user cli supplied

    #Step 1: aggregate each raw log source per (user, day) 
    log("Step 1 - Aggregating raw logs (chunked)...")
    sources = [
        chunked_agg(os.path.join(input_dir, "logon.csv"),  agg_logon,  "logon",  log),
        chunked_agg(os.path.join(input_dir, "device.csv"), agg_device, "device", log),
        chunked_agg(os.path.join(input_dir, "file.csv"),   agg_file,   "file",   log),
        chunked_agg(os.path.join(input_dir, "email.csv"),  agg_email,  "email",  log),
        chunked_agg(os.path.join(input_dir, "http.csv"),   agg_http,   "http",   log),
    ] # run the chunck agg function for every file
    sources = [source for source in sources if source is not None] # remove those that are None
    if not sources: # if there is nothing at all raise an error 
        raise ValueError(
            "No usable log data found. Expected at least one of logon.csv, "
            "device.csv, file.csv, email.csv, http.csv with CERT-style columns.")

    # Step 2: outer-merge all sources (keeps non-logon activity days)
    log("Step 2 - Merging sources (outer join)...")
    table = sources[0] # first assign the logon table first 
    for source in sources[1:]: # for the preceeding tables 
        table = pd.merge(table, source, on=["user", "date_only"], how="outer") # join using outer join
    behaviour_cols = [column for column in table.columns if column not in ("user", "date_only")] # for every column except user and date only
    table[behaviour_cols] = table[behaviour_cols].fillna(0) # fill those with 0 if missing 
    for col in FEATURE_ORDER[:16]: # the first 16 columns (which are the behavioural columns)
        if col not in table.columns: # for those that do not exist
            table[col] = 0.0 # replace its values with 0 for all rows
    log(f"  Merged: {table.shape[0]:,} user-days") # log the output 

    # Step 3: personalised deviation features 
    log("Step 3 - Personalised deviation features...")
    table = table.sort_values(["user", "date_only"]).reset_index(drop=True) # sort by user and date, drop = true renumber the rows 
    for col in DEVIATION_SOURCES: # for every column that needs to calculate deviation from
        rolling_mean = (
            table.groupby("user")[col]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean()) # if less than 6 rows deviation is 0
        )
        # no baseline yet = deviation 0 
        table[f"{col}_deviation"] = table[col] - rolling_mean.fillna(table[col]) # calculate the deviation and insert in new column 

    # Step 4: static contextual features
    log("Step 4 - Static contextual features...")
    psy_path = os.path.join(context_dir, "psychometric.csv") # define psychometric path
    if os.path.exists(psy_path): # if it exists
        psy = pd.read_csv(psy_path).rename(columns={"user_id": "user"}) # if it exist, read file, rename user_id to user
        table = table.merge(psy[["user", "O", "C", "E", "A", "N"]], on="user", how="left") # user left join to keep every row of table even if no score for that column, merge in accordance to user
    for trait in ["O", "C", "E", "A", "N"]:
        if trait not in table.columns: # if nothing was inserted, meaning missing file, fill those columns and set to nan
            table[trait] = np.nan

    # Missing psychometric scores =  (30), 
    table[["O", "C", "E", "A", "N"]] = table[["O", "C", "E", "A", "N"]].fillna(30) # for those nana set to 30 which is the average for all peoples score

    ldap_folder = os.path.join(context_dir, "LDAP") # define the ldap folder
    if os.path.isdir(ldap_folder) and sorted(os.listdir(ldap_folder)): # if folder exists and contains items
        first = sorted(os.listdir(ldap_folder))[0] # use the first file
        ldap = pd.read_csv(os.path.join(ldap_folder, first), low_memory=False) # read the file, low memory means read it in one go 
        ldap = ldap.rename(columns={"user_id": "user"}) # rename user_id to user
        ldap = ldap[["user", "role", "department", "functional_unit"]].fillna("Unknown") # choose only necessary columns and fill rows with no value with unknown 
        table = table.merge(ldap, on="user", how="left") # merge the table
    for column in ["role", "department", "functional_unit"]:
        if column not in table.columns: # if no columns meaning no file, fill the columns and put every rrow as unknown 
            table[column] = "Unknown"
    table[["role", "department", "functional_unit"]] = (
        table[["role", "department", "functional_unit"]].fillna("Unknown")) # fill again because there are some employees that might not be in the LDAP at all 

    # Step 5: encode categoricals with the  mappings
    log("Step 5 - Encoding categoricals from label_maps.pkl ...")
    if not os.path.exists(maps_path): # check if the file exists else raise error 
        raise ValueError(
            f"label_maps.pkl not found at {maps_path}. Run '2. Feature Engineering.py' "
            f"first (it saves the mappings). Refitting encoders here would "
            f"misalign codes with the trained models.")
    maps = joblib.load(maps_path)
    for col in ["role", "department", "functional_unit"]:
        name_to_code = {v: k for k, v in maps[col].items()} # maps[col].items is a dict like this 0 : "Accountant", v: k flips them around so it becomes "Accountant : 0 " because its what it needs
        unknown_code = name_to_code.get("Unknown", 0) # if unknown is not in the map then use 0 
        table[f"{col}_encoded"] = (
            table[col].astype(str).map(name_to_code).fillna(unknown_code).astype(int)) # turn in to string for mapping, map, then fill unknwon with  what is given else fill with 0, then turn into integer
        n_unseen = (~table[col].astype(str).isin(name_to_code)).sum() # count items not found in name to code
        if n_unseen:
            log(f"  [note] {col}: {n_unseen} rows had values unseen in training placed as 'Unknown'")

    out_cols = ["user", "date_only"] + FEATURE_ORDER + ["role", "department", "functional_unit"] # return the columns needed for display
    return table[out_cols]

# used for CLI identify bugs
def main():
    default_input = os.path.normpath(os.path.join(BASE_FOLDER, "..", "r4.2", "r4.2")) # where the r4.2 files and folders resides
    arg_parser = argparse.ArgumentParser(description="Raw CERT-format logs -> 32-feature CSV for the prototype")
    arg_parser.add_argument("--input",  default=default_input, help="Folder containing logon.csv, device.csv, file.csv, email.csv, http.csv, LDAP/, psychometric.csv")
    arg_parser.add_argument("--output", default=os.path.join(BASE_FOLDER, "preprocessed_features.csv"))
    arg_parser.add_argument("--maps",   default=os.path.join(BASE_FOLDER, "models", "label_maps.pkl"),
                    help="label_maps.pkl saved by feature engineering (training-time encodings)")
    args = arg_parser.parse_args()

    print("x" * 60)
    print("RAW LOG PREPROCESSOR - general logs -> 32-feature table")
    print("x" * 60)
    print(f"  Input : {args.input}")

    table = build_feature_table(args.input, maps_path=args.maps)
    table.to_csv(args.output, index=False) # index false so pandas no add row number column
    print(f"\nSaved: {args.output}")
    print(f"  {table.shape[0]:,} rows x {table.shape[1]} columns "
          f"(32 model features + identifiers + display labels)")
    print("\nUpload this CSV to the web app (http://127.0.0.1:5000) to score it.")


if __name__ == "__main__":
    main()
