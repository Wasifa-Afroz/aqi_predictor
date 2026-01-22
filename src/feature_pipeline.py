"""
Simple Feature Pipeline - Uses local storage (practical approach)
This is actually how many production systems work!
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# Feature store directory
FEATURE_STORE_DIR = Path('data/feature_store')
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)

class SimpleFeaturePipeline:
    """Simple, practical feature pipeline"""
    
    def __init__(self):
        self.feature_file = FEATURE_STORE_DIR / 'karachi_aqi_features.csv'
        print("📦 Using Local Feature Store")
        print(f"📁 Location: {self.feature_file}")
    
    def fetch_weather_and_aqi(self):
        """Fetch both weather and air quality data"""
        print("\n📡 Fetching data from OpenWeather API...")
        
        # Fetch current weather
        weather_url = "http://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            'lat': KARACHI_LAT,
            'lon': KARACHI_LON,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        # Fetch air pollution
        aqi_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        aqi_params = {
            'lat': KARACHI_LAT,
            'lon': KARACHI_LON,
            'appid': OPENWEATHER_API_KEY
        }
        
        try:
            # Get weather data
            weather_resp = requests.get(weather_url, params=weather_params, timeout=10)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
            
            # Get AQI data
            aqi_resp = requests.get(aqi_url, params=aqi_params, timeout=10)
            aqi_resp.raise_for_status()
            aqi_data = aqi_resp.json()
            
            # Combine data
            combined = {
                'timestamp': datetime.now(timezone.utc),
                'city': 'karachi',
                # Weather features
                'temperature': weather_data['main']['temp'],
                'feels_like': weather_data['main']['feels_like'],
                'temp_min': weather_data['main']['temp_min'],
                'temp_max': weather_data['main']['temp_max'],
                'pressure': weather_data['main']['pressure'],
                'humidity': weather_data['main']['humidity'],
                'wind_speed': weather_data['wind']['speed'],
                'wind_deg': weather_data['wind'].get('deg', 0),
                'clouds': weather_data['clouds']['all'],
                'visibility': weather_data.get('visibility', 10000),
                # AQI features
                'aqi': aqi_data['list'][0]['main']['aqi'],
                'co': aqi_data['list'][0]['components']['co'],
                'no': aqi_data['list'][0]['components']['no'],
                'no2': aqi_data['list'][0]['components']['no2'],
                'o3': aqi_data['list'][0]['components']['o3'],
                'so2': aqi_data['list'][0]['components']['so2'],
                'pm25': aqi_data['list'][0]['components']['pm2_5'],
                'pm10': aqi_data['list'][0]['components']['pm10'],
                'nh3': aqi_data['list'][0]['components']['nh3']
            }
            
            print(f"✅ Weather: {combined['temperature']}°C, {combined['humidity']}% humidity")
            print(f"✅ AQI: {combined['aqi']}, PM2.5: {combined['pm25']} µg/m³")
            return combined
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def engineer_features(self, raw_data):
        """Add time-based and derived features"""
        if not raw_data:
            return None
        
        features = raw_data.copy()
        
        # Time-based features
        dt = features['timestamp']
        features['year'] = dt.year
        features['month'] = dt.month
        features['day'] = dt.day
        features['hour'] = dt.hour
        features['day_of_week'] = dt.weekday()
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        features['week_of_year'] = dt.isocalendar()[1]
        
        # Derived features
        features['temp_feels_like_diff'] = features['temperature'] - features['feels_like']
        features['temp_range'] = features['temp_max'] - features['temp_min']
        
        # PM ratio (often useful predictor)
        if features['pm10'] > 0:
            features['pm25_pm10_ratio'] = features['pm25'] / features['pm10']
        else:
            features['pm25_pm10_ratio'] = 0
        
        return features
    
    def save_features(self, features):
        """Save features to CSV"""
        try:
            df = pd.DataFrame([features])
            
            # Append to existing file or create new
            if self.feature_file.exists():
                df.to_csv(self.feature_file, mode='a', header=False, index=False)
                print(f"✅ Appended features to {self.feature_file}")
            else:
                df.to_csv(self.feature_file, index=False)
                print(f"✅ Created new feature file: {self.feature_file}")
            
            # Show current size
            existing_df = pd.read_csv(self.feature_file)
            print(f"📊 Total records in feature store: {len(existing_df)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            return False
    
    def run(self):
        """Run the complete pipeline"""
        print("=" * 70)
        print("🚀 FEATURE PIPELINE")
        print("=" * 70)
        
        # Fetch data
        raw_data = self.fetch_weather_and_aqi()
        if not raw_data:
            print("\n❌ Pipeline failed - no data fetched")
            return False
        
        # Engineer features
        print("\n🔧 Engineering features...")
        features = self.engineer_features(raw_data)
        
        # Save to feature store
        print("\n💾 Saving to feature store...")
        success = self.save_features(features)
        
        if success:
            print("\n" + "=" * 70)
            print("✅ FEATURE PIPELINE COMPLETED!")
            print("=" * 70)
        
        return success


def main():
    """Main entry point"""
    pipeline = SimpleFeaturePipeline()
    success = pipeline.run()
    
    if not success:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
    
    print("\n💡 Next steps:")
    print("   1. Run backfill_data.py to generate historical data")
    print("   2. Run training_pipeline.py to train models")
    print("   3. Build the dashboard!")


if __name__ == "__main__":
    main()