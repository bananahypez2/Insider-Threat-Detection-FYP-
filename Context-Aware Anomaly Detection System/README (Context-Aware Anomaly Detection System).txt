xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CONTEXT-AWARE ANOMALY DETECTION SYSTEM - README
Prototype web application
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
This folder holds the working prototype. 
It receives either feature table or CERT raw logs 
and scores each employee
day, ranks the results, and explains each
row in plain language with a SHAP contribution
chart.

The scripts that built the dataset and trained the models are in the
another folder named "Scripts Code".


--------------------------------------------------------------------
1. Folder and its Contents
--------------------------------------------------------------------

    Context-Aware Anomaly Detection System.py
        The Flask application. Routes, scoring, SHAP explanations,
        chart generation and the ranked results view.

    Preprocess_Raw_Logs.py
        Converts raw CERT-format log files into the 32-feature table
        the models expect.

    models/
        model_D.pkl, scaler_D.pkl, features_D.pkl
        model_F.pkl, scaler_F.pkl, features_F.pkl
        label_maps.pkl

    context_data/
        LDAP/2009-12.csv : Organizational snapshot
        psychometric.csv : Big Five personality scores
        Used to attach role, department, functional unit and
        personality to a raw-log upload that does not bring its own
        context files.

    templates/
        upload.html, results.html, user_detail.html, about.html

    static/style.css


--------------------------------------------------------------------
2. Requirements
--------------------------------------------------------------------

Python 3.10 or 3.11.

Install the libraries with:

    pip install flask pandas numpy scikit-learn shap joblib matplotlib


--------------------------------------------------------------------
3. How to Run
--------------------------------------------------------------------

From a terminal opened in this folder:

    python "Context-Aware Anomaly Detection System.py"

Then open a browser at:

    http://127.0.0.1:5000

Stop the server with Ctrl+C.

