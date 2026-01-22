import os
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class MongoDBFeatureStore:
    """MongoDB Feature Store for AQI data"""
    
    def __init__(self):
        self.uri = os.getenv('MONGODB_URI')
        self.client = None
        self.db = None
        self.connect()
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client['aqi_predictor']
            print(f"✅ Connected to MongoDB: aqi_predictor database")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    def store_features(self, df, collection_name='aqi_features'):
        """Store features in MongoDB"""
        try:
            collection = self.db[collection_name]
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Convert timestamp to datetime if needed
            for record in records:
                if 'timestamp' in record and isinstance(record['timestamp'], str):
                    record['timestamp'] = pd.to_datetime(record['timestamp'])
            
            # Insert data
            result = collection.insert_many(records)
            
            print(f"✅ Stored {len(result.inserted_ids)} records in MongoDB")
            print(f"   Collection: {collection_name}")
            print(f"   Database: {self.db.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing in MongoDB: {e}")
            return False
    
    def load_features(self, collection_name='aqi_features', limit=None):
        """Load features from MongoDB"""
        try:
            collection = self.db[collection_name]
            
            # Get all documents (or limited)
            if limit:
                cursor = collection.find().limit(limit)
            else:
                cursor = collection.find()
            
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            # Remove MongoDB _id field
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            print(f"✅ Loaded {len(df)} records from MongoDB")
            print(f"   Collection: {collection_name}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading from MongoDB: {e}")
            return None
    
    def get_latest_data(self, collection_name='aqi_features', n=100):
        """Get latest N records"""
        try:
            collection = self.db[collection_name]
            
            # Get latest records sorted by timestamp
            cursor = collection.find().sort('timestamp', -1).limit(n)
            df = pd.DataFrame(list(cursor))
            
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Sort by timestamp ascending
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ Retrieved latest {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error getting latest data: {e}")
            return None
    
    def clear_collection(self, collection_name='aqi_features'):
        """Clear a collection (use carefully!)"""
        try:
            collection = self.db[collection_name]
            result = collection.delete_many({})
            print(f"✅ Deleted {result.deleted_count} records from {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔒 MongoDB connection closed")