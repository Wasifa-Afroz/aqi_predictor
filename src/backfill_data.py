import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.mongodb_feature_store import MongoDBFeatureStore

def fetch_current_aqi_for_baseline():
    """Fetch current AQI as baseline"""
    from src.utils.data_fetcher import fetch_current_aqi
    
    print("\n📡 Fetching current AQI as baseline...")
    current_data = fetch_current_aqi(city='karachi')
    
    if current_data:
        print(f"✅ Baseline AQI: {current_data['aqi']}")
        return current_data
    else:
        print("⚠️  Using default baseline values")
        return {
            'aqi': 150,
            'pm25': 65,
            'temperature': 25,
            'humidity': 50,
            'pressure': 1013,
            'wind_speed': 5
        }

def generate_synthetic_historical_data(baseline_data, days=90):
    """Generate synthetic historical data"""
    print(f"\n🔧 Generating {days} days of synthetic historical data...")
    historical_data = []
    
    start_date = datetime.now() - timedelta(days=days)
    
    aqi_base = baseline_data.get('aqi', 150)
    pm25_base = baseline_data.get('pm25', 65)
    temp_base = baseline_data.get('temperature', 25)
    humidity_base = baseline_data.get('humidity', 50)
    pressure_base = baseline_data.get('pressure', 1013)
    wind_base = baseline_data.get('wind_speed', 5)
    
    for hour in range(days * 24):
        timestamp = start_date + timedelta(hours=hour)
        hour_of_day = timestamp.hour
        
        if 7 <= hour_of_day <= 9 or 18 <= hour_of_day <= 20:
            pollution_factor = 1.2
        elif 0 <= hour_of_day <= 5:
            pollution_factor = 0.7
        else:
            pollution_factor = 1.0
        
        noise = np.random.normal(1.0, 0.15)
        
        row = {
            'timestamp': timestamp,
            'city': 'karachi',
            'aqi': max(10, int(aqi_base * pollution_factor * noise)),
            'pm25': max(5, int(pm25_base * pollution_factor * noise)),
            'pm10': max(10, int((pm25_base * 1.5) * pollution_factor * noise)),
            'o3': np.random.randint(20, 80),
            'no2': np.random.randint(10, 50),
            'so2': np.random.randint(5, 30),
            'co': round(np.random.uniform(0.3, 2.0), 2),
            'temperature': round(temp_base + np.random.normal(0, 3), 1),
            'humidity': round(max(20, min(90, humidity_base + np.random.normal(0, 10))), 1),
            'pressure': round(pressure_base + np.random.normal(0, 5), 1),
            'wind_speed': round(max(0, wind_base + np.random.normal(0, 3)), 1),
        }
        historical_data.append(row)
    
    df = pd.DataFrame(historical_data)
    print(f"✅ Generated {len(df)} hourly records")
    return df

def add_time_features(df):
    """Add time-based features"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    return df

def add_lag_features(df, columns=['aqi', 'pm25', 'temperature'], lags=[1, 2, 3, 24]):
    """Add lag features"""
    df = df.sort_values('timestamp')
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def add_rolling_features(df, columns=['aqi', 'pm25', 'temperature'], windows=[3, 6, 12, 24]):
    """Add rolling features"""
    df = df.sort_values('timestamp')
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    return df

def add_derived_features(df):
    """Add derived features"""
    df = df.sort_values('timestamp')
    df['aqi_change_rate'] = df['aqi'].diff().fillna(0)
    df['aqi_change_rate_pct'] = df['aqi'].pct_change().fillna(0) * 100
    
    if 'pm25' in df.columns:
        df['pm25_change_rate'] = df['pm25'].diff().fillna(0)
    if 'temperature' in df.columns:
        df['temp_change_rate'] = df['temperature'].diff().fillna(0)
    
    return df

def engineer_all_features(df):
    """Apply all feature engineering"""
    print("\n🔧 Engineering features...")
    
    df = add_time_features(df)
    print("   ✅ Time features")
    
    df = add_derived_features(df)
    print("   ✅ Derived features")
    
    df = add_lag_features(df)
    print("   ✅ Lag features")
    
    df = add_rolling_features(df)
    print("   ✅ Rolling features")
    
    df = df.fillna(0)
    print("   ✅ Filled missing values")
    
    print(f"✅ Total features: {len(df.columns)}")
    return df

def create_targets(df):
    """Create prediction targets"""
    print("\n🎯 Creating prediction targets...")
    df = df.sort_values('timestamp').copy()
    
    df['target_aqi_24h'] = df['aqi'].shift(-24)
    df['target_aqi_48h'] = df['aqi'].shift(-48)
    df['target_aqi_72h'] = df['aqi'].shift(-72)
    
    df = df[:-72]
    
    print(f"✅ Created targets for {len(df)} samples")
    return df

def save_local_backup(df):
    """Save local backup"""
    print("\n💾 Saving local backup...")
    
    os.makedirs('data/raw', exist_ok=True)
    raw_cols = ['timestamp', 'city', 'aqi', 'pm25', 'pm10', 'temperature', 
                'humidity', 'pressure', 'wind_speed', 'o3', 'no2', 'so2', 'co']
    df[raw_cols].to_csv('data/raw/aqi_raw.csv', index=False)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/training_data.csv', index=False)
    
    print("✅ Local backups saved")

def main():
    """Main backfill process with MongoDB"""
    print("=" * 70)
    print("🚀 BACKFILL PIPELINE - MONGODB FEATURE STORE")
    print("=" * 70)
    
    # Step 1: Connect to MongoDB
    mongo_store = MongoDBFeatureStore()
    
    # Step 2: Get baseline
    baseline_data = fetch_current_aqi_for_baseline()
    
    # Step 3: Generate historical data
    df = generate_synthetic_historical_data(baseline_data, days=90)
    
    # Step 4: Engineer features
    df = engineer_all_features(df)
    
    # Step 5: Create targets
    df = create_targets(df)
    
    # Step 6: Store in MongoDB
    print("\n💾 Storing data in MongoDB Feature Store...")
    success = mongo_store.store_features(df, collection_name='aqi_features')
    
    # Step 7: Save local backup
    save_local_backup(df)
    
    # Step 8: Close MongoDB connection
    mongo_store.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ BACKFILL COMPLETE WITH MONGODB!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Training samples: {len(df)}")
    print(f"\n📁 Storage:")
    if success:
        print(f"   ✅ MongoDB Feature Store (primary)")
    print(f"   ✅ Local CSV backup (data/processed/)")
    
    print("\n🎯 Next steps:")
    print("   1. Run training pipeline (loads from MongoDB)")
    print("   2. Build dashboard")
    print("   3. Setup automation")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Backfill failed!")
        sys.exit(1)