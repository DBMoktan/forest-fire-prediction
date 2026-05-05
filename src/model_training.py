import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def train_model(X_train, y_train, model_type='rf', params=None):
    """Train a classification model."""
    if params is None:
        params = {}
        
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, **params)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42, **params)
    else:
        raise ValueError("Invalid model_type. Choose 'logistic' or 'rf'.")
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model using various classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
    return metrics

def save_model(model, filepath):
    """Save the model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load the model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    return joblib.load(filepath)
