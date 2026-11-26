xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SCRIPTS CODE - README
Context-Aware Anomaly Detection for Suspicious User Behaviour in
System Logs Using Machine Learning
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

This folder is where the seven Python scripts that build the dataset lives,
train the two models, run the ablation study, compare against a
rule-based baseline, test whether the model actually uses
context, and generate the 2026 demonstration data.


--------------------------------------------------------------------
1. Expected Folder Layout
--------------------------------------------------------------------

The expected layout is:

    <project root>/
        Scripts Code/                 <- this folder
            1. EDA.py
            2. Feature Engineering.py
            3. Model Training and Comparison.py
            4. Ablation Study.py
            5. Rule Based Baseline.py
            D1. Check Context Sensitivity.py
            E2. Generate 2026 Demo.py
        r4.2/
            r4.2/                     <- the CERT r4.2 log files
                logon.csv
                device.csv
                file.csv
                email.csv
                http.csv
                psychometric.csv
                LDAP/
        answers/
            answers/                  <- the CERT r4.2 answer key
                insiders.csv
                r4.2-1/ r4.2-2/ r4.2-3/
        feature_table_final.csv       <- created by script 2
        models/                       <- created by scripts 2 and 3
        eda_outputs/                  <- created by script 1
        results/                      <- created by scripts 3, 4 and 5

The three output folders do not need to exist beforehand. Each script
creates the folders it writes into.

--------------------------------------------------------------------
3. Requirements
--------------------------------------------------------------------

Python 3.10 was used for these scripts.

Install the libraries with:

    pip install pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn

The saved model files in the application folder were produced with
scikit-learn 1.9.0.


--------------------------------------------------------------------
4. Order of Script Execution
--------------------------------------------------------------------

The scripts must be run in the order below, because each one refers to
what the previous one wrote.

    1. EDA.py
    2. Feature Engineering.py
    3. Model Training and Comparison.py
    4. Ablation Study.py
    5. Rule Based Baseline.py
    D1. Check Context Sensitivity.py
    E2. Generate 2026 Demo.py

--------------------------------------------------------------------
5. Script Description
--------------------------------------------------------------------

1. EDA.py
    Purpose  : Explores the raw CERT r4.2 dataset before any feature
               is built.
    Reads    : r4.2/r4.2/ and answers/answers/
    Writes   : eda_outputs/event_volume_plot1.png
               eda_outputs/logon_monthly_plot2.png
               eda_outputs/logon_users_plot3.png

2. Feature Engineering.py
    Purpose  : Turns the raw event logs into the day-level feature
               table that every later script and the prototype use.
    Reads    : r4.2/r4.2/ and answers/answers/
    Writes   : feature_table_final.csv
               malicious_days_ground_truth_v3.csv
               models/label_maps.pkl

3. Model Training and Comparison.py
    Purpose  : Trains the two models the prototype deploys, Model D
               on a temporal split and Model F on a held-out user
               split.
    Reads    : feature_table_final.csv
    Writes   : models/model_D.pkl, scaler_D.pkl, features_D.pkl
               models/model_F.pkl, scaler_F.pkl, features_F.pkl
               results/production/report_D.txt, report_F.txt
               results/production/production_metrics.csv

4. Ablation Study.py
    Purpose  : Runs the six comparison configurations to confirm core thesis, then repeats the held-out pair across
               five seeds.
    Reads    : feature_table_final.csv
    Writes   : results/ablation/report_A.txt through report_F.txt
               results/ablation/ablation_results.csv
               results/ablation/robustness_check_E_F.csv

5. Rule Based Baseline.py
    Purpose  : Tests score on six fixed threshold
               rules, and compare machine learning method with it
    Reads    : feature_table_final.csv
               results/ablation/ablation_results.csv
    Writes   : results/rule_based/rule_baseline_results.csv
               results/rule_based/rule_baseline_report.txt

D1. Check Context Sensitivity.py
    Purpose  : Tests whether the model is genuinely using the
               contextual features, by not changing behavioural related features and
               changing only static context.
    Reads    : models/ and feature_table_final.csv
    Writes   : console output only. The saved run is in the report.

E2. Generate 2026 Demo.py
    Purpose  : Builds the derived 2026 demonstration dataset from CERT,
               five fictional employees with one insider among them.
    Reads    : feature_table_final.csv
               r4.2/r4.2/LDAP/2009-12.csv
               r4.2/r4.2/psychometric.csv
    Writes   : demo_raw_logs_2026.zip, 
               demo_2026_build/, 

--------------------------------------------------------------------
6. Ouputs
--------------------------------------------------------------------

The outputs of the scirpts are provided within the outputs folder provided within this current folder. 

--------------------------------------------------------------------
6. CERT r4.2 dataset download
--------------------------------------------------------------------

Link: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
Download:
	- r4.2.tar.bz2
	- answers.tar.bz2