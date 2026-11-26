# Trained model artefacts
## Download them

Get all seven files from the [Releases page](https://github.com/bananahypez2/Insider-Threat-Detection-FYP-/releases) and place them in
this folder:

```
models/
├─ model_D.pkl        Random Forest, temporal split (the deployed detector)
├─ scaler_D.pkl       StandardScaler fitted on the Model D training set
├─ features_D.pkl     the 32 feature names, in training order
├─ model_F.pkl        Random Forest, held-out users split
├─ scaler_F.pkl       StandardScaler fitted on the Model F training set
├─ features_F.pkl     the 32 feature names, in training order
└─ label_maps.pkl     role / department / functional unit → integer codes
```
label_maps.pkl is needed for the app to work
