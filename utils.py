# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Create new features from existing ones to capture non-linear relationships.
    Works on both DataFrames and single-row input (dict/list).
    """
    # Ensure we work on a copy
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif isinstance(df, list):
        df = pd.DataFrame([df], columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                         'SkinThickness', 'Insulin', 'BMI',
                                         'DiabetesPedigreeFunction', 'Age'])
    else:
        df = df.copy()

    # 1. BMI categories (WHO classification)
    df['BMI_Category'] = pd.cut(df['BMI'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # 2. Age groups
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 30, 45, 60, 120],
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])

    # 3. Glucose to insulin ratio (indicates insulin resistance)
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)  # +1 to avoid division by zero

    # 4. Pregnancy * Age (interaction term)
    df['Preg_Age'] = df['Pregnancies'] * df['Age']

    # Convert categorical to dummy variables (one-hot encoding)
    df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)

    return df

def preprocess_input(data, scaler=None):
    """
    Full preprocessing pipeline:
    - Handle zero/missing values (replace with median – we'll pass medians)
    - Feature engineering
    - Scaling
    """
    # Define medians from training set (computed later during training)
    # We'll hardcode them here for simplicity, but in production you'd load them
    medians = {
        'Glucose': 117.0, 'BloodPressure': 72.0, 'SkinThickness': 23.0,
        'Insulin': 30.5, 'BMI': 32.0
    }

    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

    # Replace zeros with median for biologically impossible values
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(medians[col], inplace=True)

    # Engineer features
    df_eng = engineer_features(df)

    # Ensure columns are in the exact order and count as during training
    # We'll define this after training. For now, we assume the scaler knows the columns.
    if scaler:
        # The scaler expects the exact feature set; we'll align later in app
        pass

    return df_eng