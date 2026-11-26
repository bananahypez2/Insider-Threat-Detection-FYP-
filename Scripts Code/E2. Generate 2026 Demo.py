"""
E2. Generate 2026 Demo v2.py
"""
import os, random, zipfile, string
import numpy as np, pandas as pd

# Fixed folders, worked out from where this script sits, so it just runs with no arguments
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))                              # Script Code
IMPLEMENTATION_FOLDER = os.path.normpath(os.path.join(SCRIPT_FOLDER, ".."))              # Cleaning
OUTPUT_FOLDER = SCRIPT_FOLDER  # where the demo bundle is written

SEED = 2026 #fixed seed
random_generator = np.random.default_rng(SEED)
random.seed(SEED)

FEATURE_TABLE_LOCATION = os.path.join(IMPLEMENTATION_FOLDER, "feature_table_final.csv")
LDAP_LOCATION = os.path.join(IMPLEMENTATION_FOLDER, "r4.2", "r4.2", "LDAP", "2009-12.csv")
PSYCHOMETRIC_LOCATION = os.path.join(IMPLEMENTATION_FOLDER, "r4.2", "r4.2", "psychometric.csv")

EMPLOYEES = ["EMP2601", "EMP2602", "EMP2603", "EMP2604", "EMP2605"]

INSIDER = "EMP2603"

NORMAL_EMPLOYEES = [employee for employee in EMPLOYEES if employee != INSIDER]

EMPLOYEE_NAMES = {"EMP2601":"Bao Jit",
                  "EMP2602":"Chen Xuan",
                  "EMP2603":"Julio Tjoe",
                  "EMP2604":"Kiwi Fruit",
                  "EMP2605":"Eugene Tan"}

START_DATE, END_DATE = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-07-17")

ATTACK_DATES = [pd.Timestamp(d) for d in ("2026-07-08","2026-07-10","2026-07-14","2026-07-16")]

BENIGN_URLS = ["cnn.com/world/story",
               "weather.com/today",
               "espn.com/scores",
               "wikipedia.org/wiki/Networking",
               "bbc.co.uk/news",
               "reuters.com/markets",
               "stackoverflow.com/questions",
               "nytimes.com/section/tech"]

SUSPICIOUS_URLS = ["www.indeed.com/viewjob",
                   "www.linkedin.com/jobs/view",
                   "www.monster.com/jobs",
                   "www.glassdoor.com/Reviews",
                   "careerbuilder.com/search",
                   "wikileaks.org/wiki/Leak"]

EXTERNAL_EMAIL_DOMAINS = ["gmail.com",
                          "yahoo.com",
                          "hotmail.com",
                          "outlook.com"]

# CERT style event identifier, for example {A1B2-C3D4E5F6-G7H8I9J0}
def random_event_id():
    letters_and_digits = lambda length: "".join(random.choices(string.ascii_uppercase + string.digits, k=length)) # lambda lenghth means def letters_and_digits(length), 
                                                                                                                  # combine all upcase of A-Z and digits of 0-9. length passed, combine randomly chosen of length into a string
    return "{" + f"{letters_and_digits(4)}-{letters_and_digits(8)}-{letters_and_digits(8)}" + "}" # generate the random id

# Random timestamp on the given date, between two hours of the day
def random_timestamp(date, from_hour, to_hour): #from hour and to hour passed as decimals eg 8.05
    seconds_into_day = random_generator.integers(int(from_hour*3600), int(to_hour*3600)) # turn the hours into seconds, pick a random number in between both 
    return (date + pd.Timedelta(seconds=int(seconds_into_day))).strftime("%m/%d/%Y %H:%M:%S") # turn decimals into integer, 
                                                                                              # timedelta turn seconds into length 8 hours, 20 minutes,2 seconds, 
                                                                                              # strftime turns it into the format month/day/year, hours:minutes:seconds

# Filler text so message and content fields are not empty
def filler_words(word_count): # return how many words as needed
    word_pool = ("report summary meeting quarterly review project client schedule update "
            "budget invoice draft notes agenda planning forecast analysis team").split() # wordlist, .split split the spaces turn into list of 18 words
    return " ".join(random.choices(word_pool, k=word_count)) #picks the defined count randomly from wordpull and join

print("Step 1 - Loading real CERT distributions for calibration...")
needed_columns = ["user","date_only","label","logon_count","after_hours_logon","weekend_logon",
                    "unique_pcs","logoff_count","usb_connect_count","usb_after_hours","file_copy_count",
                    "file_copy_after_hours","emails_total","emails_external","emails_with_attachment",
                    "emails_after_hours","http_count","suspicious_url_count","http_after_hours"] # 16 raw behaviour + user, date, label (who when and malicious or not)
feature_table = pd.read_csv(FEATURE_TABLE_LOCATION, usecols=needed_columns, parse_dates=["date_only"]) #usecols select those that are defined, 
                                                                                                       # parse dates tuen the date only column into Timestamp, Timestamp('2010-01-04 00:00:00')
feature_table["is_weekend"] = feature_table["date_only"].dt.dayofweek >= 5 #dayofweek Mon = 0, Sun = 6, locate Sat and Sun, True then turn is weekend column into 1

# The four ordinary employees are bootstrapped from active users with no
# malicious days. The INSIDER re-enacts ONE real CERT insider end-to-end:
# their normal days come from that insider's own pre-attack behaviour, their
# attack days from that insider's actual labelled malicious days. This
# preserves the real deviation geometry (attack vs personal baseline), which
# is the model's key signal, without any tuning against the model.
users_with_malicious_days = set(feature_table.loc[feature_table.label == 1, "user"]) # select those columns with label 1 meaning malicious, then .loc returns the user for that row, set strip the duplicate
days_per_user = feature_table[~feature_table.user.isin(users_with_malicious_days)].groupby("user").size() #keep only the clean users' row, group by user and return size, returns two halves, 
                                                                                                          # index is username, value is number of rows
eligible_normal_users = days_per_user[days_per_user >= 200].index.to_numpy() # take the index(username) of those >=200, 
                                                                             # .to_numpy() removes the dtype= and name= from Index(['AAE0190', 'AAF0791', 'AAL0706', 'AAN0823', 'AAV0450'], dtype='object', name='user')
normal_templates = random_generator.choice(eligible_normal_users, size=len(NORMAL_EMPLOYEES), replace=False) #draw 4(len(NORMAL_EMPLOYEES)) at random replace=Flase means no repeats
template_for_employee = dict(zip(NORMAL_EMPLOYEES, normal_templates)) # match two list position by position

# Pick the insider template: a strong exfiltration style attacker with a solid
# normal history, chosen on the data alone rather than on any model score.
malicious_rows = feature_table[feature_table.label == 1] # choose those rows that are labbelled malicious (dataframe)
exfiltration_intensity = (malicious_rows.usb_connect_count + malicious_rows.file_copy_count + malicious_rows.after_hours_logon + malicious_rows.suspicious_url_count) # return a Series that has all these values added up 
candidates = (malicious_rows.assign(intensity=exfiltration_intensity).groupby("user").agg(attack_day_count=("label","size"), highest_intensity=("intensity","max"))) # put a temp column (the sum of exfiltration intensity for each row), 
                                                                                                                                                                     # group by users, agg(label, size) size(count) how many label columns for that user,
                                                                                                                                                                     #  which means total rows, then another show the maximum intensity calculated for each user 
candidates["normal_day_count"] = feature_table[(feature_table.label==0) & feature_table.user.isin(candidates.index)].groupby("user").size() # add a new column called normal day count, 
                                                                                                                                            # where selecet label = 0  and is malicious insider, the total up how mnay rows
candidates = candidates[(candidates.attack_day_count >= 4) & (candidates.normal_day_count >= 120)] # select only those that has 4 attack days and 120 normal days
insider_template_user = candidates.sort_values("highest_intensity", ascending=False).index[0] # finally select the user, sort the candidates by highest intensity, acending is false, 
                                                                                              # means decending, high to low, take index[0] which is username
template_for_employee[INSIDER] = insider_template_user #set it as the insider in the new dataset
print("Templates:", template_for_employee)
print(f"Insider re-enacts real CERT insider: {insider_template_user}")

attack_day_pool = (feature_table[(feature_table.user == insider_template_user) & (feature_table.label == 1)]
                   .assign(intensity=lambda rows: rows.usb_connect_count + rows.file_copy_count
                    + rows.after_hours_logon + rows.suspicious_url_count) 
            .sort_values("intensity", ascending=False)) # keep the rows where it is the selected insider and is malicious day, 
                                                        # assign an intensity row, using lambda function for each row compute the sum of the 4 features, then sort by intensity highest to lowest)
print(f"  Attack pool: {len(attack_day_pool)} labelled malicious days of {insider_template_user}") 

print("Step 2 - Bootstrapping 2026 daily behaviour vectors...")
calendar_dates = pd.date_range(START_DATE, END_DATE, freq="D") #lay out the day freq"D" means daily day by day no skips, start and end date included 
rows_by_log_file = {name: [] for name in ("logon","device","file","email","http")} #dictionary with five keys - {'logon': [], 'device': [], 'file': [], 'email': [], 'http': []}
day_plan = []  # (employee, date, day_values, is_attack_day), ('EMP2601', Timestamp('2026-03-02'), <that day's counts>, False)

attack_day_counter = iter(range(len(ATTACK_DATES))) #len 4, range(4) [0,1,2,3], inter specifically to keep track of current number
for employee in EMPLOYEES: # for every employee in Employees
    template_days = feature_table[(feature_table.user == template_for_employee[employee]) & (feature_table.label == 0)] # filter out the normal days for each selected employee
    weekday_days = template_days[~template_days.is_weekend] # Filter out days that are weekends
    for current_date in calendar_dates: # For every date in range
        is_attack_day = (employee == INSIDER) and (current_date in ATTACK_DATES) # if employee insider and is an attack day then set as true
        if is_attack_day: # if is an attack day 
            day_values = attack_day_pool.iloc[next(attack_day_counter)] #next and iter works hand in hand, so if attack day point to the next unused number
        elif current_date.dayofweek >= 5: #if its sat and sun (skip)
            continue  # weekends are not part of the demonstration period
        else:
            day_values = weekday_days.sample(1, random_state=int(random_generator.integers(1e9))).iloc[0] # first generate a random number 1 to a biliion, 
                                                                                                         # sample 1 means how many whihc is one and random state means which, 
                                                                                                         # random state and sample is used to select same row with same seed, iloc is to unwrap the whole row 
        day_plan.append((employee, current_date, day_values, is_attack_day)) # a tuple of 500 

# count how many of the planned days ended up being attack days
attack_days_planned = 0
for employee, current_date, day_values, is_attack_day in day_plan:
    if is_attack_day:
        attack_days_planned += 1

print(f"  Planned {len(day_plan)} employee-days ({attack_days_planned} attack days)")

print("Step 3 - Emitting raw events...")
for employee, current_date, day_values, is_attack_day in day_plan: #for every item in day_plan
    pc_names = [f"PC-26{EMPLOYEES.index(employee)+1:02d}", f"PC-26{EMPLOYEES.index(employee)+1:02d}B", f"PC-26{EMPLOYEES.index(employee)+1:02d}C", f"PC-26{EMPLOYEES.index(employee)+1:02d}D",
                f"PC-26{EMPLOYEES.index(employee)+1:02d}E", f"PC-26{EMPLOYEES.index(employee)+1:02d}F", f"PC-26{EMPLOYEES.index(employee)+1:02d}G"]
                #PC is named in this format, the index of employee within the employee list + 1, turn into two digits, 2 -> 02, B is for somedays where employee use more than 2 pc
    pc_count = int(day_values.unique_pcs) # turn unique pc value from decimal 1.0 to int, because float is cannot used in index list

    # logons: after-hours ones at night, rest in the morning
    logon_count, after_hours_logon_count = int(day_values.logon_count), int(day_values.after_hours_logon) # assign logon count to logon count in day_values, and after hours 
                                                                                                          # logon count to after hours logon in day values
    for event_index in range(logon_count): #for every logon count 
        is_night = event_index < after_hours_logon_count # do the after logon hours first , 0 < 2, 1 < 2, 2 < 2
        rows_by_log_file["logon"].append([random_event_id(), random_timestamp(current_date, 0.5, 4.0) if is_night else random_timestamp(current_date, 8.05, 9.6),
                                      employee, pc_names[event_index % pc_count], "Logon"]) #start append the file, 1st event id, next timestamp, start time, end time, 
                                                                                            # employee, using % modulus to correctly put unique pcs
    for event_index in range(int(day_values.logoff_count)):
        rows_by_log_file["logon"].append([random_event_id(), random_timestamp(current_date, 17.0, 19.5), employee, pc_names[0], "Logoff"]) #same thing for log off

    # device
    usb_count = int(day_values.usb_connect_count) # take the usb count from the day values
    usb_after_hours_count = int(day_values.usb_after_hours) # take the usb after hours count from the day values
    for event_index in range(usb_count): # for every usb count
        is_night = event_index < usb_after_hours_count # do the after hours first
        connect_time = random_timestamp(current_date, 0.6, 3.8) if is_night else random_timestamp(current_date, 9.0, 17.0) # connection time for both before and after hours
        rows_by_log_file["device"].append([random_event_id(), connect_time, employee, pc_names[0], "Connect"]) # Connect log
        rows_by_log_file["device"].append([random_event_id(), connect_time, employee, pc_names[0], "Disconnect"]) # each connect will have a disconnect

    # file copies
    file_copy_count = int(day_values.file_copy_count) # take file copies count from day values
    file_copy_after_hours_count = int(day_values.file_copy_after_hours) # take file copies count after hours from day value
    for event_index in range(file_copy_count): # for every file copy count
        is_night = event_index < file_copy_after_hours_count # do after hours first
        file_name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8)) + random.choice([".doc",".pdf",".zip",".xlsx"]) #random choices of 8 asciss and digits, + file extension
        rows_by_log_file["file"].append([random_event_id(), random_timestamp(current_date, 0.7, 3.9) if is_night else random_timestamp(current_date, 9.0, 17.5), 
                                     employee, pc_names[0], file_name, "D0-CF-11-E0-A1-B1-1A-E1 " + filler_words(6)]) # same thinng append into the file 
        
    # emails
    email_count = int(day_values.emails_total)
    external_email_count = int(day_values.emails_external)
    attachment_email_count = int(day_values.emails_with_attachment)
    after_hours_email_count = int(day_values.emails_after_hours)
    first_name, last_name = EMPLOYEE_NAMES[employee].split()
    for event_index in range(email_count):
        is_external = event_index < external_email_count # do is external emails first 
        recipient = (f"Agent{random_generator.integers(1,99)}@{random.choice(EXTERNAL_EMAIL_DOMAINS)}" if is_external
                else f"{random.choice(['Ali','Abu','Bob','Taylor'])}.{random.choice(['Smith','Wong','Kong','Qing'])}@dtaa.com") #if external assign external email, else put as internal
        attachment_count = int(random_generator.integers(1, 20)) if event_index < attachment_email_count else 0 # do attachment email count 
        is_night = event_index < after_hours_email_count # do after hours email first
        rows_by_log_file["email"].append([random_event_id(), random_timestamp(current_date, 0.8, 3.5) if is_night else random_timestamp(current_date, 8.5, 17.5),
                                      employee, pc_names[0], recipient, "", "", f"{first_name}.{last_name}@dtaa.com",
                                      int(random_generator.integers(8000, 90000)), attachment_count, filler_words(8)]) # fill in the rows
        
    # http
    http_count = int(day_values.http_count)
    suspicious_url_count = int(day_values.suspicious_url_count)
    after_hours_http_count = int(day_values.http_after_hours)
    for event_index in range(http_count): # for every http count
        is_suspicious = event_index < suspicious_url_count #do suspicious url count first
        is_night = event_index < after_hours_http_count # do after hours first
        url = "http://" + (random.choice(SUSPICIOUS_URLS) if is_suspicious else random.choice(BENIGN_URLS)) + f"/{random_generator.integers(1000,999999)}" #if is suspiciouos then sus url else benign 
        rows_by_log_file["http"].append([random_event_id(), random_timestamp(current_date, 0.9, 4.2) if is_night else random_timestamp(current_date, 8.2, 17.9),
                                     employee, pc_names[0], url, filler_words(5)]) #fill in the rows 

print("Step 4 - Writing bundle files...")
BUILD_FOLDER = os.path.join(OUTPUT_FOLDER, "demo_2026_build") #to save the unzip version
os.makedirs(os.path.join(BUILD_FOLDER, "LDAP"), exist_ok=True) #make a LDAP folder to store LDAP file
FILE_HEADERS = {"logon": ["id","date","user","pc","activity"],
           "device": ["id","date","user","pc","activity"],
           "file":  ["id","date","user","pc","filename","content"],
           "email": ["id","date","user","pc","to","cc","bcc","from","size","attachments","content"],
           "http":  ["id","date","user","pc","url","content"]} # define the headears
for log_name, rows in rows_by_log_file.items(): # for every log names and its rows in the dictionary
    pd.DataFrame(rows, columns=FILE_HEADERS[log_name]).to_csv(os.path.join(BUILD_FOLDER, f"{log_name}.csv"), index=False) # index False is to prevent pandas from writing its own number as first column 
    print(f"{log_name}.csv: {len(rows):,} rows") #print log name and number of rows 

# Self-contained context: roles/departments drawn from REAL CERT categories
ldap_all_users = pd.read_csv(LDAP_LOCATION, low_memory=False) #read the original LDAP snapshot file, low memory false to tell pandas to read the whole file before deciding column type
ldap_context_options = ldap_all_users[["role","business_unit","functional_unit","department"]].dropna() #remove those that are blank in one of the four columns
chosen_context = ldap_context_options.sample(len(EMPLOYEES), random_state=SEED).reset_index(drop=True) #keep the sampled 5 rows and drop the row numbering, renumber with new index

# The insider inherits the REAL organisational context of the insider they
# re-enact - same behaviour in a different role would (correctly) score
# differently, to ensure context-awareness being demonstrated
insider_ldap_row = ldap_all_users[ldap_all_users.user_id == insider_template_user].iloc[0] #locate the insider ldap row 
insider_index = EMPLOYEES.index(INSIDER) #index of the insider within the EMPlOYEES list
for column in ("role", "business_unit", "functional_unit", "department"): # for these columns
    chosen_context.loc[insider_index, column] = insider_ldap_row[column] # replace the current list with the real LDAP insider profile
ldap_output = pd.DataFrame({
    "employee_name": [EMPLOYEE_NAMES[employee] for employee in EMPLOYEES], #write employee name
    "user_id": EMPLOYEES, #employee id
    "email": [EMPLOYEE_NAMES[employee].replace(" ", ".") + "@dtaa.com" for employee in EMPLOYEES], #email
    "role": chosen_context["role"], "business_unit": chosen_context["business_unit"], #role, unit
    "functional_unit": chosen_context["functional_unit"], "department": chosen_context["department"], #functional unit department
    "team": [f"2 - Team{index+1}" for index in range(len(EMPLOYEES))], #random team (not used)
    "supervisor": ["Kool Beanz"] * len(EMPLOYEES), # suprevissor name
})
ldap_output.to_csv(os.path.join(BUILD_FOLDER, "LDAP", "2026-01.csv"), index=False) # write into file with index as false prevent pandas from writing its own number as frist column

psychometric_all_users = pd.read_csv(PSYCHOMETRIC_LOCATION) # read the psychometric file
chosen_psychometric = psychometric_all_users.sample(len(EMPLOYEES), random_state=SEED).reset_index(drop=True) #select random 5, drop original index and renumber index 
insider_psychometric_row = psychometric_all_users[psychometric_all_users.user_id == insider_template_user].iloc[0] #identify insiders row and get its data
for trait in ("O", "C", "E", "A", "N"): #for every trait
    chosen_psychometric.loc[insider_index, trait] = insider_psychometric_row[trait] #replace with real insider trait
psychometric_output = pd.DataFrame({
    "employee_name": [EMPLOYEE_NAMES[employee] for employee in EMPLOYEES], #name
    "user_id": EMPLOYEES, #id
    "O": chosen_psychometric["O"],  
    "C": chosen_psychometric["C"], 
    "E": chosen_psychometric["E"],
    "A": chosen_psychometric["A"], 
    "N": chosen_psychometric["N"],
})
psychometric_output.to_csv(os.path.join(BUILD_FOLDER, "psychometric.csv"), index=False) # write into file with index as false prevent pandas from writing its own number as frist column
print("LDAP/2026-01.csv and psychometric.csv generated")

zip_location = os.path.join(OUTPUT_FOLDER, "demo_raw_logs_2026.zip") #zip file location
with zipfile.ZipFile(zip_location, "w", zipfile.ZIP_DEFLATED) as zip_file: #zip the file with write mode, with zip deflated meaning compressed 
    for log_name in FILE_HEADERS: 
        zip_file.write(os.path.join(BUILD_FOLDER, f"{log_name}.csv"), f"{log_name}.csv") #copy into zip
    zip_file.write(os.path.join(BUILD_FOLDER, "LDAP", "2026-01.csv"), "LDAP/2026-01.csv") #copy into zip
    zip_file.write(os.path.join(BUILD_FOLDER, "psychometric.csv"), "psychometric.csv") #copy into zip
print(f"\nSaved: {zip_location} ")
print(f"Insider: {INSIDER} ({EMPLOYEE_NAMES[INSIDER]}) - attack days: {[str(d.date()) for d in ATTACK_DATES]}") #convert date into this format '2026-07-08'
