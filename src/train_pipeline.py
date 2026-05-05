import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Relative imports if running as a module, or direct imports if running as a script
try:
    from .data_preprocessing import load_data, clean_data, preprocess_features
    from .model_training import train_model, save_model, evaluate_model
except ImportError:
    from data_preprocessing import load_data, clean_data, preprocess_features
    from model_training import train_model, save_model, evaluate_model

def main():
    # Ensure paths are correct relative to project root
    # This assumes the script is run from the project root directory
    raw_data_path = os.path.join('data', 'raw', 'algerian_forest_fire_dataset.csv')
    model_save_path = os.path.join('models', 'saved_models', 'best_rf_model.pkl')
    scaler_save_path = os.path.join('models', 'saved_models', 'scaler.pkl')
    
    # 1. Load and Clean
    if not os.path.exists(raw_data_path):
        # Try one level up if run from src/
        raw_data_path = os.path.join('..', 'data', 'raw', 'algerian_forest_fire_dataset.csv')
        if not os.path.exists(raw_data_path):
            print(f"Raw data not found at {raw_data_path}")
            return
            
    df = load_data(raw_data_path)
    df_cleaned = clean_data(df)
    
    # 2. Split
    X = df_cleaned.drop(['Classes'], axis=1)
    y = df_cleaned['Classes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Preprocess and Save Scaler
    # preprocess_features logic handles scaler saving at 'models/saved_models/scaler.pkl'
    # We should make sure the path is correct relative to root
    X_train_processed = preprocess_features(X_train, is_train=True)
    X_test_processed = preprocess_features(X_test, is_train=False)
    
    # 4. Train best model (Random Forest)
    model = train_model(X_train_processed, y_train, model_type='rf')
    
    # 5. Evaluate
    metrics = evaluate_model(model, X_test_processed, y_test)
    print(f"Model Metrics: {metrics}")
    
    # 6. Save Model
    save_model(model, model_save_path)
    print("Model and Scaler saved successfully.")

if __name__ == "__main__":
    main()
