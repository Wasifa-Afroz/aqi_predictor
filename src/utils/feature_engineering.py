import pandas as pd
from datetime import datetime

def add_time_features(df):
    """
    Add time-based features (as required in PDF)
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    
    return df

def add_lag_features(df, columns=['aqi', 'pm25', 'temperature'], lags=[1, 2, 3, 24]):
    """
    Add lag features (previous values)
    """
    df = df.sort_values('timestamp')
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def add_rolling_features(df, columns=['aqi', 'pm25', 'temperature'], windows=[3, 6, 12, 24]):
    """
    Add rolling average features
    """
    df = df.sort_values('timestamp')
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    
    return df

def add_derived_features(df):
    """
    Add derived features like AQI change rate (as required in PDF)
    """
    df = df.sort_values('timestamp')
    
    # AQI change rate (difference from previous hour)
    df['aqi_change_rate'] = df['aqi'].diff()
    df['aqi_change_rate_pct'] = df['aqi'].pct_change() * 100
    
    # PM2.5 change rate
    if 'pm25' in df.columns:
        df['pm25_change_rate'] = df['pm25'].diff()
    
    # Temperature change rate
    if 'temperature' in df.columns:
        df['temp_change_rate'] = df['temperature'].diff()
    
    return df

def engineer_all_features(df):
    """
    Apply all feature engineering steps
    """
    print("🔧 Engineering features...")
    
    # Add time features
    df = add_time_features(df)
    print(f"   ✅ Time features added")
    
    # Add derived features (AQI change rate, etc.)
    df = add_derived_features(df)
    print(f"   ✅ Derived features added")
    
    # Add lag features
    df = add_lag_features(df)
    print(f"   ✅ Lag features added")
    
    # Add rolling features
    df = add_rolling_features(df)
    print(f"   ✅ Rolling features added")
    
    # Drop rows with NaN values (from lag/rolling calculations)
    initial_rows = len(df)
    df = df.dropna()
    print(f"   ℹ️  Dropped {initial_rows - len(df)} rows with missing values")
    
    return df