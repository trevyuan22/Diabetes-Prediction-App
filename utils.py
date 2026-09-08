import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Add new features: BMI categories, age groups, glucose/insulin ratio, and interaction terms.
    Works for both DataFrames and single-row dicts.
    """
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    elif isinstance(df, list):
        df = pd.DataFrame([df], columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                         'SkinThickness', 'Insulin', 'BMI',
                                         'DiabetesPedigreeFunction', 'Age'])
    else:
        df = df.copy()

    # BMI categories (WHO)
    df['BMI_Category'] = pd.cut(df['BMI'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Age groups
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 30, 45, 60, 120],
                             labels=['Young', 'Middle', 'Senior', 'Elderly'])

    # Glucose/Insulin ratio (add 1 to avoid division by zero)
    df['Glucose_Insulin_Ratio'] = df['Glucose'] / (df['Insulin'] + 1)

    # Interaction: pregnancies * age
    df['Preg_Age'] = df['Pregnancies'] * df['Age']

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['BMI_Category', 'Age_Group'], drop_first=True)

    return df