"""
CERT r4.2 - Context-Sensitivity (Counterfactual) Test
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER = os.path.join(BASE_FOLDER, "models") # Model folder
FEATURE_TABLE_LOCATION = os.path.join(BASE_FOLDER, "feature_table_final.csv") # feature table location
RANDOM_STATE = 42 # fixed seed
NUMBER_OF_TEST_DAYS = 10 # how many days are examined in this test

STATIC_ORGANISATIONAL = ["role_encoded", "department_encoded", "functional_unit_encoded"] # static organisational column
STATIC_PSYCHOMETRIC = ["O", "C", "E", "A", "N"] # psychometric column
STATIC_ALL = STATIC_PSYCHOMETRIC + STATIC_ORGANISATIONAL # all static column


print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Context-Sensitivity (Counterfactual) Test - Model D")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

model = joblib.load(os.path.join(MODELS_FOLDER, "model_D.pkl")) # load model
scaler = joblib.load(os.path.join(MODELS_FOLDER, "scaler_D.pkl")) # load scaler
FEATURES = joblib.load(os.path.join(MODELS_FOLDER, "features_D.pkl")) # load feature
label_maps = joblib.load(os.path.join(MODELS_FOLDER, "label_maps.pkl")) # load label mapping
role_map = label_maps["role"] # turn encoded role into role
department_map = label_maps["department"] # turn encoded department into department
functional_unit_map = label_maps["functional_unit"] # turn encoded unit into unit

feature_table = pd.read_csv(FEATURE_TABLE_LOCATION, low_memory=False) # read feature table
print("Feature table rows :", len(feature_table)) # print the number of rows in feature table
print("Model used :", type(model).__name__) # print the model use which is Random Forest
print("Number of features :", len(FEATURES)) # print the number of features

# position of each static column inside the 32 features, needed because numpy
# arrays only have column numbers and not column names
ORG_COLUMN_INDEXES = []
for column in STATIC_ORGANISATIONAL:
    ORG_COLUMN_INDEXES.append(FEATURES.index(column)) # turn into list of indexes for numpy use

ALL_COLUMN_INDEXES = []
for column in STATIC_ALL:
    ALL_COLUMN_INDEXES.append(FEATURES.index(column)) # turn into list of indexes for numpy use


# takes rows of raw feature values and returns the probability of malicious for each row
def score_rows(feature_values):
    scaled_values = scaler.transform(feature_values.astype(float)) # scale with the saved training scaler, which except only one type, so float it is
    probabilities = model.predict_proba(scaled_values) # column 0 is normal, column 1 is malicious
    return probabilities[:, 1] # keep only the malicious probability


# every role/department/unit combination that actually exists, so no invented job is tested
org_contexts = feature_table[STATIC_ORGANISATIONAL].drop_duplicates() #drop duplicates
org_context_values = org_contexts.values.astype(float) #drop column name and row index for numpy

# every complete identity that actually exists, which is job plus the five personality scores
full_contexts = feature_table[STATIC_ALL].drop_duplicates() #drop duplicates
full_context_values = full_contexts.values.astype(float) #drop column name and row index for numpy

print("Real organisational contexts :", len(org_context_values)) # count number of rows
print("Real full identities :", len(full_context_values)) # count number of rows


# the days under test are drawn at random with a fixed seed before anything is
# scored, so the choice cannot have been influenced by the results
malicious_days = feature_table[feature_table["label"] == 1] # select malicious days with label 1
test_days = malicious_days.sample(NUMBER_OF_TEST_DAYS, random_state=RANDOM_STATE).reset_index(drop=True) # sample 10 rows from all malicious
test_day_values = test_days[FEATURES].values.astype(float) # drop column name and row index for numpy
real_scores = score_rows(test_day_values) # score of each day exactly as it really happened

print("Malicious days available :", len(malicious_days))# number of malicious days
print("Malicious days under test :", len(test_days))# nuber of test days


print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("TEST 1 - Swap the job only (role, department, functional unit)")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Each test day is scored again under every real organisational context.")
print("Behaviour and personality scores are not touched.")
print("")

test1_all_scores = [] # every day list of scores, one score per context
test1_differences = [] # highest minus lowest for each day

for day_number in range(len(test_days)): #for every day (10)
    original_day = test_day_values[day_number] # select first row as current day

    # build one copy of this day for every organisational context
    modified_rows = []
    for context in org_context_values: #runs until all new rows are appended
        new_row = original_day.copy() # make a copy of the row
        new_row[ORG_COLUMN_INDEXES] = context # only the job columns are replaced
        modified_rows.append(new_row) # appends

    scores_for_this_day = score_rows(np.array(modified_rows))
    lowest_score = scores_for_this_day.min() # lowest score for this day
    highest_score = scores_for_this_day.max() # highest score for this day
    difference = highest_score - lowest_score # difference

    test1_all_scores.append(scores_for_this_day) #append the score for the day
    test1_differences.append(difference) # append the difference

    print("Day", day_number + 1, "-", test_days["user"][day_number], "on", test_days["date_only"][day_number]) # Present data
    print("     score with the real job :", round(real_scores[day_number], 4)) # display rounded to 4 decimals
    print("     lowest score :", round(lowest_score, 4))
    print("     highest score :", round(highest_score, 4))
    print("     difference :", round(difference, 4))

test1_mean_difference = sum(test1_differences) / len(test1_differences) # mean difference among 10 days
test1_largest_difference = max(test1_differences) # max difference among 10 days
test1_worst_day = test1_differences.index(test1_largest_difference) #the worst day difference

print("")
print("Mean difference over", len(test_days), "days :", round(test1_mean_difference, 4))
print("Largest difference :", round(test1_largest_difference, 4))
print("Day with the largest difference  :", test_days["user"][test1_worst_day], "on", test_days["date_only"][test1_worst_day]) # display results


print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("TEST 2 - Swap the whole identity (job plus Big Five personality)")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("The same test days are scored again under every real full identity.")
print("Only the behaviour is kept the same this time.")
print("")

test2_differences = [] # highest minus lowest for each day

for day_number in range(len(test_days)):
    original_day = test_day_values[day_number] #select the current test day row

    # build one copy of this day for every full identity
    modified_rows = []
    for context in full_context_values:
        new_row = original_day.copy() # make copy of original
        new_row[ALL_COLUMN_INDEXES] = context # job columns and personality columns are replaced
        modified_rows.append(new_row) # append until finish

    scores_for_this_day = score_rows(np.array(modified_rows)) # scores for ths day
    lowest_score = scores_for_this_day.min()
    highest_score = scores_for_this_day.max()
    difference = highest_score - lowest_score

    test2_differences.append(difference)

    print("Day", day_number + 1, "-", test_days["user"][day_number], "on", test_days["date_only"][day_number])
    print("     score with the real identity :", round(real_scores[day_number], 4))
    print("     lowest score                 :", round(lowest_score, 4))
    print("     highest score                :", round(highest_score, 4))
    print("     difference                   :", round(difference, 4))

test2_mean_difference = sum(test2_differences) / len(test2_differences)
test2_largest_difference = max(test2_differences)
test2_worst_day = test2_differences.index(test2_largest_difference)

print("")
print("Mean difference over", len(test_days), "days :", round(test2_mean_difference, 4))
print("Largest difference               :", round(test2_largest_difference, 4))
print("Day with the largest difference  :", test_days["user"][test2_worst_day], "on", test_days["date_only"][test2_worst_day])


print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("COMPARISON - Test 1 against Test 2 on the same days")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("The extra movement in Test 2 is what the personality scores add on top")
print("of the job title.")
print("")

# calculate difference between job switch and full identity switch
for day_number in range(len(test_days)):
    added_by_personality = test2_differences[day_number] - test1_differences[day_number]
    print("Day", day_number + 1, "-", test_days["user"][day_number])
    print("     moved by job only :", round(test1_differences[day_number], 4))
    print("     moved by full identity :", round(test2_differences[day_number], 4))
    print("     added by personality :", round(added_by_personality, 4))

# calculate the mean difference between job switch and identitiy switch
print("")
print("Mean movement from the job only      :", round(test1_mean_difference, 4))
print("Mean movement from the full identity :", round(test2_mean_difference, 4))
print("Mean added by the personality scores :", round(test2_mean_difference - test1_mean_difference, 4))


print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("MOST DAMAGING ORGANISATIONAL CONTEXT")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("This is the job that produced the highest average score across all the")
print("test days, so it is measured over every day and not taken from one day.")
print("")

# average score each organisational context produced across all the test days
average_score_per_context = np.array(test1_all_scores).mean(axis=0) # axis 0 makes it so it averages the one that we want, which is compressing day instead of context, so the result will be a row of 107 context average
most_damaging_position = int(np.argmax(average_score_per_context)) # find the position of the worst context combination
most_damaging_context = org_context_values[most_damaging_position] # the role unit and department of the most damaging one

print("Role :", role_map.get(int(most_damaging_context[0]))) # return the actual label name using label map
print("Department :", department_map.get(int(most_damaging_context[1])))
print("Functional unit :", functional_unit_map.get(int(most_damaging_context[2])))
print("Average score it produced :", round(average_score_per_context[most_damaging_position], 4)) # the actual average score
print("Lowest average of any job :", round(average_score_per_context.min(), 4)) # the minimum average score


print("")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("TEST 3 - Reverse test, can a damaging job flag an innocent day")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("Ordinary non-malicious days are given the most damaging job found above.")
print("Their behaviour is not touched. If the job alone was enough to convict")
print("someone, these scores would cross the 0.50 flagging line.")
print("")

normal_days = feature_table[feature_table["label"] == 0] # select only non malicious days
print(len(normal_days))
normal_sample = normal_days.sample(NUMBER_OF_TEST_DAYS, random_state=RANDOM_STATE).reset_index(drop=True) # select 10 at random
normal_values = normal_sample[FEATURES].values.astype(float) # turn into float
scores_before = score_rows(normal_values) # score each row

# same rows again, but with the damaging job written over their real job
injected_values = normal_values.copy() #make a copy
injected_values[:, ORG_COLUMN_INDEXES] = most_damaging_context # replace all rows (:) the job with the most damaging job dept role setting
scores_after = score_rows(injected_values) #score again after injection

number_of_days_flagged = 0 # initialize
score_changes = [] # store differrence

for day_number in range(len(normal_sample)):
    change = scores_after[day_number] - scores_before[day_number] # changes in score for every day
    score_changes.append(change)

    if scores_after[day_number] >= 0.50:
        flagged = "yes"
        number_of_days_flagged = number_of_days_flagged + 1
    else:
        flagged = "no"

    print("Day", day_number + 1, "-", normal_sample["user"][day_number], "on", normal_sample["date_only"][day_number])
    print("     score with the real job :", round(scores_before[day_number], 4))
    print("     score with the damaging job :", round(scores_after[day_number], 4))
    print("     change :", round(change, 4))
    print("     flagged at 0.50 :", flagged)

mean_change = sum(score_changes) / len(score_changes)

print("")
print("Mean change from the damaging job :", round(mean_change, 4))
print("Normal days flagged at 0.50 :", number_of_days_flagged, "out of", len(normal_sample))


print("")
print("xxxxxxxxxxxxxxxx")
print("SUMMARY")
print("xxxxxxxxxxxxxxxx")
print("1. Context is used. Changing only the job moved the test days by",
      round(test1_mean_difference, 4), "on average.")
print("2. Personality is used as well. Changing the whole identity moved them by",
      round(test2_mean_difference, 4), "on average, which is",
      round(test2_mean_difference - test1_mean_difference, 4), "more.")
print("3. Behaviour still decides. Giving ordinary days the most damaging job changed")
print("   their score by", round(mean_change, 4), "on average and flagged", number_of_days_flagged, "of them.")
print("Context adjusts the score, but behaviour decides whether a day is flagged.")
