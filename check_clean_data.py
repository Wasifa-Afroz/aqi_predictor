"""
Check and Clean Duplicate Data in MongoDB
This ensures no redundant data affects model training
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from src.utils.mongodb_feature_store import MongoDBFeatureStore

print("=" * 70)
print("🔍 DATA QUALITY CHECK - MongoDB Feature Store")
print("=" * 70)

# Connect to MongoDB
store = MongoDBFeatureStore()

# Load all data
print("\n📊 Loading all data from MongoDB...")
df = store.load_features('aqi_features')

if df is None or len(df) == 0:
    print("❌ No data found in MongoDB!")
    store.close()
    sys.exit(1)

print(f"✅ Loaded {len(df)} total records")

# Check for duplicates
print("\n🔍 Checking for duplicate timestamps...")

if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Find duplicates
    duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
    
    if len(duplicates) > 0:
        print(f"⚠️  Found {len(duplicates)} duplicate records!")
        print(f"\nExample duplicates:")
        print(duplicates[['timestamp', 'aqi', 'pm25']].head(10))
        
        # Ask user if they want to remove duplicates
        print("\n" + "=" * 70)
        response = input("Do you want to remove duplicates? (yes/no): ")
        
        if response.lower() == 'yes':
            print("\n🧹 Removing duplicates...")
            
            # Keep first occurrence, remove rest
            df_clean = df.drop_duplicates(subset=['timestamp'], keep='first')
            
            print(f"✅ Removed {len(df) - len(df_clean)} duplicate records")
            print(f"📊 Clean data: {len(df_clean)} unique records")
            
            # Clear old data
            print("\n🗑️  Clearing old data from MongoDB...")
            store.clear_collection('aqi_features')
            
            # Store clean data
            print("💾 Storing clean data...")
            store.store_features(df_clean, 'aqi_features')
            
            print("\n✅ Data cleaned successfully!")
        else:
            print("\n❌ Cancelled - no changes made")
    else:
        print("✅ No duplicates found - data is clean!")
else:
    print("⚠️  No timestamp column found")

# Show data statistics
print("\n" + "=" * 70)
print("📊 DATA STATISTICS")
print("=" * 70)

if 'timestamp' in df.columns:
    df = df.sort_values('timestamp')
    print(f"\n📅 Date Range:")
    print(f"   Earliest: {df['timestamp'].min()}")
    print(f"   Latest: {df['timestamp'].max()}")
    print(f"   Total Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

print(f"\n📊 Records:")
print(f"   Total: {len(df)}")
print(f"   Features: {len(df.columns)}")

if 'aqi' in df.columns:
    print(f"\n🌫️  AQI Statistics:")
    print(f"   Mean: {df['aqi'].mean():.1f}")
    print(f"   Min: {df['aqi'].min():.1f}")
    print(f"   Max: {df['aqi'].max():.1f}")

# Close connection
store.close()

print("\n" + "=" * 70)
print("✅ DATA QUALITY CHECK COMPLETE!")
print("=" * 70)
