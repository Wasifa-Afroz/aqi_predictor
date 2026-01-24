import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def load_training_data():
    """Load prepared training data"""
    print("📊 Loading training data...")
    
    try:
        df = pd.read_csv('data/processed/training_data.csv')
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Features: {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def prepare_features_targets(df, target='target_aqi_24h'):
    """Prepare X (features) and y (target) for training"""
    print(f"\n🎯 Preparing features for target: {target}")
    
    # Remove non-feature columns
    exclude_cols = ['timestamp', 'city', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target].values
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    return X, y, feature_cols

def split_data(X, y, test_size=0.2):
    """Split data into train and test sets"""
    print(f"\n✂️  Splitting data (test size: {test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # Don't shuffle time series!
    )
    
    print(f"✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features to improve model performance"""
    print("\n📏 Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics (RMSE, MAE, R² as required in PDF)"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 {model_name} Performance:")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   R²: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def train_ridge_regression(X_train, X_test, y_train, y_test):
    """Train Ridge Regression model (Required in PDF)"""
    print("\n" + "=" * 70)
    print("🔵 Training Ridge Regression Model (REQUIRED)")
    print("=" * 70)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "Ridge (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "Ridge (Test)")
    
    return model, test_metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model (Required in PDF)"""
    print("\n" + "=" * 70)
    print("🌲 Training Random Forest Model (REQUIRED)")
    print("=" * 70)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Training (this may take a minute)...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "Random Forest (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "Random Forest (Test)")
    
    return model, test_metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model (Advanced gradient boosting)"""
    print("\n" + "=" * 70)
    print("⚡ Training XGBoost Model (ADVANCED)")
    print("=" * 70)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Training...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "XGBoost (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "XGBoost (Test)")
    
    return model, test_metrics

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model (Fast gradient boosting - replaces LSTM)"""
    print("\n" + "=" * 70)
    print("💡 Training LightGBM Model (DEEP LEARNING ALTERNATIVE)")
    print("=" * 70)
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=15,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("🔄 Training...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_metrics = evaluate_model(y_train, y_pred_train, "LightGBM (Train)")
    test_metrics = evaluate_model(y_test, y_pred_test, "LightGBM (Test)")
    
    return model, test_metrics

def save_models(ridge_model, rf_model, xgb_model, lgb_model, scaler, feature_cols, metrics):
    """Save all trained models"""
    print("\n💾 Saving models...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models
    joblib.dump(ridge_model, 'models/ridge_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    joblib.dump(lgb_model, 'models/lightgbm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save metrics
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ All models saved to models/ directory")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("🚀 TRAINING PIPELINE - ML MODELS")
    print("=" * 70)
    
    # Load data
    df = load_training_data()
    if df is None:
        return False
    
    # Prepare features and targets (24h prediction)
    X, y, feature_cols = prepare_features_targets(df, target='target_aqi_24h')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train models
    print("\n" + "=" * 70)
    print("🎯 TRAINING 4 MODELS")
    print("=" * 70)
    
    # 1. Ridge Regression (REQUIRED)
    ridge_model, ridge_metrics = train_ridge_regression(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 2. Random Forest (REQUIRED)
    rf_model, rf_metrics = train_random_forest(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 3. XGBoost (Advanced)
    xgb_model, xgb_metrics = train_xgboost(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 4. LightGBM (Replaces LSTM - Deep Learning Alternative)
    lgb_model, lgb_metrics = train_lightgbm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Compare models
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON (RMSE, MAE, R² as required)")
    print("=" * 70)
    
    metrics = {
        'Ridge Regression': ridge_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
        'LightGBM': lgb_metrics
    }
    
    comparison_df = pd.DataFrame(metrics).T
    print("\n" + comparison_df.to_string())
    
    # Find best model
    best_model = comparison_df['rmse'].idxmin()
    best_rmse = comparison_df.loc[best_model, 'rmse']
    print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f})")
    
    # Save all models
    save_models(ridge_model, rf_model, xgb_model, lgb_model, scaler, feature_cols, metrics)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📝 Note: LightGBM replaces TensorFlow LSTM due to Python 3.13 compatibility")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed!")
        sys.exit(1)

from src.mlflow_registry import register_all_models

# Register models in MLflow
print("\n📦 Registering models in MLflow Model Registry...")
register_all_models(
    models={'Random Forest': rf_model, 'XGBoost': xgb_model, 
            'LightGBM': lgb_model, 'Ridge': ridge_model},
    metrics=metrics,
    scaler=scaler
)