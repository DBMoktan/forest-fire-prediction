import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os

def load_data(filepath):
    """Load dataset from filepath."""
    return pd.read_csv(filepath, header=1)

def clean_data(df):
    """Perform data cleaning as per notebook 01."""
    # Drop rows that are completely empty
    df.dropna(how='all', inplace=True)
    
    # Add a 'Region' column (0 for Bejaia, 1 for Sidi-Bel Abbes)
    # The Bejaia region spans up to original index 122.
    df.loc[:122, 'Region'] = 0
    df.loc[122:, 'Region'] = 1
    df['Region'] = df['Region'].astype(int)
    
    # Remove the redundant header rows from the middle of the dataset
    df = df[df['day'] != 'day']
    df = df[df['day'] != 'Sidi-Bel Abbes Region Dataset']
    
    # Strip leading and trailing spaces from all column names
    df.columns = df.columns.str.strip()
    
    # Drop the row with missing values (corrupted data entry)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Convert chronological and discrete columns to integers
    int_columns = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws']
    for col in int_columns:
        df[col] = df[col].astype(int)
        
    # Convert indices to floats
    float_columns = ['Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop any newly introduced NaNs from coercion
    df.dropna(inplace=True)
    
    # Clean string inconsistencies in the Classes column and map to binary
    df['Classes'] = df['Classes'].astype(str).str.strip()
    df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)
    
    return df.reset_index(drop=True)

def preprocess_features(X, is_train=True, scaler_path='models/saved_models/scaler.pkl'):
    """Apply feature engineering steps: dropping columns, log1p transformation, and scaling."""
    # Copy to avoid modifying original
    X = X.copy()
    
    # Drop date columns if present
    cols_to_drop = ['day', 'month', 'year']
    X.drop(columns=[col for col in cols_to_drop if col in X.columns], inplace=True)
    
    # Drop highly correlated features identified in EDA
    correlated_cols = ['BUI', 'ISI', 'FWI', 'DC']
    X.drop(columns=[col for col in correlated_cols if col in X.columns], inplace=True)
    
    # Apply Log1p transformation to highly right-skewed features
    # Note: Only Rain and DMC are in the final 7 features that were skewed
    skewed_features = ['Rain', 'DMC']
    for feature in skewed_features:
        if feature in X.columns:
            X[feature] = np.log1p(X[feature])
            
    # Scaling
    if is_train:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        
    return pd.DataFrame(X_scaled, columns=X.columns)
