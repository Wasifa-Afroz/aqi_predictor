"""
Model Improvement Script
Goal: Achieve R² > 0.7 by:
1. Hyperparameter tuning
2. More data
3. Better features
4. Ensemble methods
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib

print("=" * 80)
print("🎯 MODEL IMPROVEMENT - TARGET R² > 0.7")
print("=" * 80)

# Load data
df = pd.read_csv('data/processed/training_data.csv')
print(f"\n📊 Dataset: {len(df)} samples")

# Prepare features
exclude_cols = ['timestamp', 'city', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['target_aqi_24h'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# STRATEGY 1: Hyperparameter Tuning for Random Forest
# ============================================================================
print("\n" + "=" * 80)
print("🔧 STRATEGY 1: Hyperparameter Tuning (Random Forest)")
print("=" * 80)

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [25, 30, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

print("🔍 Testing combinations... (this may take 5-10 minutes)")

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf_base, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

best_rf = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# Evaluate
y_pred_train = best_rf.predict(X_train_scaled)
y_pred_test = best_rf.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n📊 Tuned Random Forest Performance:")
print(f"   Train R²: {train_r2:.4f}")
print(f"   Test R²: {test_r2:.4f}")
print(f"   Test RMSE: {test_rmse:.2f}")
print(f"   Test MAE: {test_mae:.2f}")

# ============================================================================
# STRATEGY 2: Optimized XGBoost
# ============================================================================
print("\n" + "=" * 80)
print("🔧 STRATEGY 2: Optimized XGBoost")
print("=" * 80)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("🔄 Training XGBoost...")
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)

print(f"\n📊 Optimized XGBoost Performance:")
print(f"   Test R²: {xgb_r2:.4f}")
print(f"   Test RMSE: {xgb_rmse:.2f}")
print(f"   Test MAE: {xgb_mae:.2f}")

# ============================================================================
# STRATEGY 3: Optimized LightGBM
# ============================================================================
print("\n" + "=" * 80)
print("🔧 STRATEGY 3: Optimized LightGBM")
print("=" * 80)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=20,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("🔄 Training LightGBM...")
lgb_model.fit(X_train_scaled, y_train)

y_pred_lgb = lgb_model.predict(X_test_scaled)
lgb_r2 = r2_score(y_test, y_pred_lgb)
lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
lgb_mae = mean_absolute_error(y_test, y_pred_lgb)

print(f"\n📊 Optimized LightGBM Performance:")
print(f"   Test R²: {lgb_r2:.4f}")
print(f"   Test RMSE: {lgb_rmse:.2f}")
print(f"   Test MAE: {lgb_mae:.2f}")

# ============================================================================
# STRATEGY 4: Ensemble (Voting) - Combine All 3
# ============================================================================
print("\n" + "=" * 80)
print("🔧 STRATEGY 4: Ensemble Model (Best of All 3)")
print("=" * 80)

ensemble = VotingRegressor([
    ('rf', best_rf),
    ('xgb', xgb_model),
    ('lgb', lgb_model)
])

print("🔄 Training Ensemble...")
ensemble.fit(X_train_scaled, y_train)

y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_r2 = r2_score(y_test, y_pred_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)

print(f"\n📊 Ensemble Performance:")
print(f"   Test R²: {ensemble_r2:.4f}")
print(f"   Test RMSE: {ensemble_rmse:.2f}")
print(f"   Test MAE: {ensemble_mae:.2f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("🏆 FINAL MODEL COMPARISON")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Tuned Random Forest', 'Optimized XGBoost', 'Optimized LightGBM', 'Ensemble'],
    'R²': [test_r2, xgb_r2, lgb_r2, ensemble_r2],
    'RMSE': [test_rmse, xgb_rmse, lgb_rmse, ensemble_rmse],
    'MAE': [test_mae, xgb_mae, lgb_mae, ensemble_mae]
})

results = results.sort_values('R²', ascending=False)
print("\n" + results.to_string(index=False))

# Find best model
best_model_name = results.iloc[0]['Model']
best_r2 = results.iloc[0]['R²']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")

if best_r2 >= 0.7:
    print("   ✅ TARGET ACHIEVED! R² >= 0.7")
else:
    print(f"   ⚠️ Close! Need {(0.7 - best_r2):.4f} more to reach 0.7")

# Save best model
if best_model_name == 'Tuned Random Forest':
    best_model = best_rf
elif best_model_name == 'Optimized XGBoost':
    best_model = xgb_model
elif best_model_name == 'Optimized LightGBM':
    best_model = lgb_model
else:
    best_model = ensemble

joblib.dump(best_model, 'models/best_model_optimized.pkl')
joblib.dump(scaler, 'models/scaler_optimized.pkl')

print(f"\n✅ Best model saved to models/best_model_optimized.pkl")

# Save improvement report
import os
os.makedirs('reports', exist_ok=True)

print("\n📄 Saving improvement report...")
with open('reports/model_improvement_report.txt', 'w') as f:
    f.write("MODEL IMPROVEMENT REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Goal: Achieve R² > 0.7\n\n")
    f.write("Strategies Applied:\n")
    f.write("1. Hyperparameter tuning (GridSearchCV)\n")
    f.write("2. Optimized XGBoost parameters\n")
    f.write("3. Optimized LightGBM parameters\n")
    f.write("4. Ensemble voting method\n\n")
    f.write("Results:\n")
    f.write(results.to_string(index=False))
    f.write(f"\n\nBest Model: {best_model_name}\n")
    f.write(f"R² Score: {best_r2:.4f}\n")
    
    if best_r2 >= 0.7:
        f.write("\n✅ TARGET ACHIEVED!\n")
    else:
        f.write(f"\n⚠️ Need more data or features to reach 0.7\n")
        f.write("\nRecommendations:\n")
        f.write("- Collect more historical data (currently 90 days → 6 months)\n")
        f.write("- Add external features (traffic data, industrial activity)\n")
        f.write("- Use LSTM for temporal patterns\n")

print("✅ Report saved to reports/model_improvement_report.txt")

print("\n" + "=" * 80)
print("✅ MODEL IMPROVEMENT COMPLETE!")
print("=" * 80)