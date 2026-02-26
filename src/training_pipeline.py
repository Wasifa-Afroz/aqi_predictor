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
    """Load prepared training data - MongoDB first, CSV fallback"""
    print("📊 Loading training data...")
    
    # ═══ CHANGE 1: Try MongoDB first ═══════════════════════════════════════
    try:
        from src.utils.mongodb_feature_store import MongoDBFeatureStore
        print("   🔗 Trying MongoDB...")
        store = MongoDBFeatureStore()
        df = store.load_features('aqi_features')
        store.close()
        
        if df is not None and len(df) > 100:
            print(f"✅ Loaded {len(df)} samples from MongoDB")
            print(f"   Features: {len(df.columns)} columns")
            
            # Create targets if missing
            df = df.sort_values('timestamp').reset_index(drop=True)
            if 'target_aqi_24h' not in df.columns:
                print("   🔧 Creating target columns...")
                df['target_aqi_24h'] = df['aqi'].shift(-24)
                df['target_aqi_48h'] = df['aqi'].shift(-48)
                df['target_aqi_72h'] = df['aqi'].shift(-72)
                df = df[:-72]
            
            # Save CSV backup
            os.makedirs('data/processed', exist_ok=True)
            df.to_csv('data/processed/training_data.csv', index=False)
            print("   💾 Saved CSV backup")
            return df
    except Exception as e:
        print(f"   ⚠️  MongoDB unavailable: {e}")
    # ═══════════════════════════════════════════════════════════════════════
    
    # Original CSV loading (unchanged)
    try:
        df = pd.read_csv('data/processed/training_data.csv')
        print(f"✅ Loaded {len(df)} samples from CSV")
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
    
    # ═══ CHANGE 2: Remove NaN values ══════════════════════════════════════
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    # ═══════════════════════════════════════════════════════════════════════
    
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
    
    # ═══ CHANGE 3: Also save best model as model_24h.pkl ══════════════════
    # Find best model
    best_name = min(metrics, key=lambda k: metrics[k]['rmse'])
    if 'LightGBM' in best_name:
        best_model = lgb_model
    elif 'XGBoost' in best_name:
        best_model = xgb_model
    elif 'Random Forest' in best_name:
        best_model = rf_model
    else:
        best_model = ridge_model
    
    # Save as model_24h.pkl (for app.py)
    joblib.dump(best_model, 'models/model_24h.pkl')
    print(f"✅ Best model ({best_name}) saved as model_24h.pkl")
    # ═══════════════════════════════════════════════════════════════════════
    
    # Save all models (unchanged)
    joblib.dump(ridge_model, 'models/ridge_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    joblib.dump(lgb_model, 'models/lightgbm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save metrics with metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'best_model': best_name,
        'metrics': metrics
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
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
    else:
        print("\n✅ Training successful!")
        sys.exit(0)

# ═══ CHANGE 4: MLflow import COMMENTED OUT (prevents crashes) ═══════════
# Uncomment these lines if you want MLflow model registry tracking:
#
# from src.mlflow_registry import register_all_models
# print("\n📦 Registering models in MLflow Model Registry...")
# register_all_models(
#     models={'Random Forest': rf_model, 'XGBoost': xgb_model, 
#             'LightGBM': lgb_model, 'Ridge': ridge_model},
#     metrics=metrics,
#     scaler=scaler
# )
#
# To enable MLflow:
# 1. Ensure mlflow_registry.py is in src/ folder
# 2. Uncomment the lines above
# 3. Run: pip install mlflow
# ═══════════════════════════════════════════════════════════════════════════
