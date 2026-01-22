import hopsworks
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME')

print("=" * 50)
print("🔗 Testing Hopsworks Connection")
print("=" * 50)

try:
    # Connect to Hopsworks
    print(f"📡 Connecting to project: {PROJECT_NAME}...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=PROJECT_NAME
    )
    
    print(f"✅ Successfully connected to Hopsworks!")
    print(f"   Project: {project.name}")
    
    # Get feature store
    fs = project.get_feature_store()
    print(f"✅ Feature Store accessed!")
    print(f"   Feature Store: {fs.name}")
    
    print("\n" + "=" * 50)
    print("✅ HOPSWORKS CONNECTION SUCCESSFUL!")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n💡 Troubleshooting:")
    print("1. Check your HOPSWORKS_API_KEY in .env")
    print("2. Check your HOPSWORKS_PROJECT_NAME in .env")
    print("3. Make sure you created the project in Hopsworks")