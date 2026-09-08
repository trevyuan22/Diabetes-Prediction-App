# train_model.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import urllib.request
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap

# Import your feature engineering function
from utils import engineer_features

# ------------------------------
# 0. Download dataset if missing
# ------------------------------
if not os.path.exists('diabetes.csv'):
    print("Downloading diabetes.csv...")
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
    urllib.request.urlretrieve(url, 'diabetes.csv')
    print("Download complete.")

# ------------------------------
# 1. Load and clean data (no chained assignment)
# ------------------------------
df = pd.read_csv('diabetes.csv')

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for feat in zero_features:
    df[feat] = df[feat].replace(0, np.nan)
    median_val = df[feat].median()
    df[feat] = df[feat].fillna(median_val)   # Direct assignment, no inplace

# ------------------------------
# 2. Feature Engineering
# ------------------------------
df_eng = engineer_features(df)
X = df_eng.drop('Outcome', axis=1)
y = df_eng['Outcome']
feature_names = X.columns.tolist()

# ------------------------------
# 3. Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# 4. Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# ------------------------------
# 5. SMOTE + Hyperparameter Tuning
# ------------------------------
# Explicitly disable categorical splits to avoid SHAP error later
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss',
    enable_categorical=False   # <--- key fix
)

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb_model)
])

# Use n_jobs=1 to avoid Windows access violation issues
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc',
                           n_jobs=1, verbose=1)   # <--- n_jobs=1 for stability
grid_search.fit(X_train_scaled_df, y_train)

best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ------------------------------
# 6. Evaluation on Test Set
# ------------------------------
y_pred = best_pipeline.predict(X_test_scaled_df)
y_proba = best_pipeline.predict_proba(X_test_scaled_df)[:, 1]

print("\n" + "="*60)
print("TEST SET PERFORMANCE")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------
# 7. SHAP Explainability (fixed)
# ------------------------------
model = best_pipeline.named_steps['classifier']

# Use tree_path_dependent to avoid categorical split error
explainer = shap.TreeExplainer(
    model,
    X_train_scaled_df,
    feature_perturbation='tree_path_dependent'   # <--- key fix
)
shap_values = explainer.shap_values(X_test_scaled_df)

# Summary plot
shap.summary_plot(shap_values, X_test_scaled_df, feature_names=feature_names, show=False)
plt.title("SHAP Feature Importance - Diabetes Prediction")
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot
shap.summary_plot(shap_values, X_test_scaled_df, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSHAP plots saved as 'shap_summary.png' and 'shap_bar.png'")

# ------------------------------
# 8. Save Artifacts
# ------------------------------
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'feature_names.pkl')
medians = {feat: df[feat].median() for feat in zero_features}
joblib.dump(medians, 'medians.pkl')

print("\n✅ All artifacts saved successfully!")