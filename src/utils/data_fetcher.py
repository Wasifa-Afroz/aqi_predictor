import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

def fetch_current_aqi(city='karachi'):
    """
    Fetch current AQI and weather data from OpenWeather API
    """
    # Weather data endpoint
    weather_url = "http://api.openweathermap.org/data/2.5/weather"
    weather_params = {
        'lat': KARACHI_LAT,
        'lon': KARACHI_LON,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    
    # Air pollution endpoint
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
        clean_data = {
            'timestamp': datetime.now(),
            'city': city,
            # AQI (OpenWeather uses 1-5 scale, convert to US EPA scale ~0-500)
            'aqi': aqi_data['list'][0]['main']['aqi'] * 50,  # Rough conversion
            # Pollutants
            'pm25': aqi_data['list'][0]['components']['pm2_5'],
            'pm10': aqi_data['list'][0]['components']['pm10'],
            'o3': aqi_data['list'][0]['components']['o3'],
            'no2': aqi_data['list'][0]['components']['no2'],
            'so2': aqi_data['list'][0]['components']['so2'],
            'co': aqi_data['list'][0]['components']['co'],
            # Weather
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'wind_speed': weather_data['wind']['speed'],
        }
        
        return clean_data
        
    except Exception as e:
        print(f"❌ Error fetching data from OpenWeather: {e}")
        return None

def save_raw_data(data, filename='data/raw/aqi_raw.csv'):
    """
    Save raw data to CSV
    """
    if data is None:
        return
    
    df = pd.DataFrame([data])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Append or create
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    
    print(f"💾 Raw data saved to {filename}")