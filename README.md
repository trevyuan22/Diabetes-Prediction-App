**train\_model.py**



This script:



* Loads data, handles zeros with median imputation.
* Applies Feature Engineering.
* Splits data.
* Uses SMOTE to balance the classes.
* Performs Hyperparameter Tuning using GridSearchCV on XGBoost.
* Trains the best model, evaluates it, and saves it along with the scaler and feature names.
* Generates SHAP summary plots for explainability.


Working URL for this app:
https://diabetespredictorlive.streamlit.app/





**app.py**



This app loads the saved model, scaler, feature names, and medians. It uses the same preprocessing pipeline to ensure consistency between training and prediction.

