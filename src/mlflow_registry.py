"""
MLflow Model Registry - Professional Model Versioning
Tracks models, metrics, and parameters
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Set MLflow tracking directory
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AQI_Predictor_Models")

class ModelRegistry:
    """Manage model versions and metadata"""
    
    def __init__(self):
        self.tracking_uri = "file:./mlruns"
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f"📁 MLflow tracking URI: {self.tracking_uri}")
    
    def register_model(self, model, model_name, metrics, params=None, model_type="sklearn"):
        """
        Register a model with MLflow
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., "Random_Forest")
            metrics: Dict of metrics (rmse, mae, r2)
            params: Dict of hyperparameters
            model_type: Type of model ("sklearn", "xgboost", "lightgbm")
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log additional metadata
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            mlflow.set_tag("data_version", "v1.0")
            
            # Log the model (use sklearn logger for all - it's compatible)
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=model_name
            )
            
            print(f"✅ Registered {model_name} in MLflow Model Registry")
            print(f"   RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            print(f"   MAE: {metrics.get('mae', 'N/A'):.2f}")
            print(f"   R²: {metrics.get('r2', 'N/A'):.4f}")
    
    def load_model(self, model_name, version="latest"):
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the registered model
            version: Version to load ("latest" or specific version number)
        
        Returns:
            Loaded model
        """
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Loaded {model_name} (version: {version})")
        return model
    
    def get_best_model(self, metric="rmse"):
        """
        Get the best performing model based on a metric
        
        Args:
            metric: Metric to optimize ("rmse", "mae", "r2")
        
        Returns:
            Best model name and its metrics
        """
        experiment = mlflow.get_experiment_by_name("AQI_Predictor_Models")
        if experiment is None:
            print("⚠️ No experiments found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if metric != 'r2' else 'DESC'}"]
        )
        
        if runs.empty:
            print("⚠️ No runs found")
            return None
        
        best_run = runs.iloc[0]
        
        print(f"\n🏆 Best Model (by {metric}):")
        print(f"   Run ID: {best_run['run_id']}")
        print(f"   Model: {best_run['tags.mlflow.runName']}")
        print(f"   RMSE: {best_run['metrics.rmse']:.2f}")
        print(f"   MAE: {best_run['metrics.mae']:.2f}")
        print(f"   R²: {best_run['metrics.r2']:.4f}")
        
        return {
            'run_id': best_run['run_id'],
            'model_name': best_run['tags.mlflow.runName'],
            'metrics': {
                'rmse': best_run['metrics.rmse'],
                'mae': best_run['metrics.mae'],
                'r2': best_run['metrics.r2']
            }
        }
    
    def list_models(self):
        """List all registered models"""
        client = mlflow.tracking.MlflowClient()
        
        try:
            registered_models = client.search_registered_models()
            
            if not registered_models:
                print("⚠️ No models registered yet")
                return []
            
            print("\n📦 Registered Models:")
            print("-" * 70)
            
            for rm in registered_models:
                print(f"   Model: {rm.name}")
                print(f"   Latest Version: {rm.latest_versions[0].version if rm.latest_versions else 'N/A'}")
                print("-" * 70)
            
            return registered_models
            
        except Exception as e:
            print(f"⚠️ No models found: {e}")
            return []
    
    def compare_models(self):
        """Compare all models and create a comparison report"""
        experiment = mlflow.get_experiment_by_name("AQI_Predictor_Models")
        if experiment is None:
            print("⚠️ No experiments found")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            print("⚠️ No runs found")
            return None
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': runs['tags.mlflow.runName'],
            'RMSE': runs['metrics.rmse'],
            'MAE': runs['metrics.mae'],
            'R²': runs['metrics.r2'],
            'Date': runs['start_time']
        })
        
        comparison = comparison.sort_values('RMSE')
        
        print("\n📊 Model Comparison:")
        print(comparison.to_string(index=False))
        
        # Save comparison
        Path('reports').mkdir(exist_ok=True)
        comparison.to_csv('reports/model_comparison.csv', index=False)
        print("\n✅ Saved comparison to reports/model_comparison.csv")
        
        return comparison


def register_all_models(models, metrics, scaler):
    """
    Register all trained models in MLflow
    
    Args:
        models: Dict of trained models
        metrics: Dict of model metrics
        scaler: Trained scaler
    """
    registry = ModelRegistry()
    
    print("\n" + "=" * 70)
    print("📦 REGISTERING MODELS IN MLFLOW")
    print("=" * 70)
    
    # Register each model
    for model_name, model in models.items():
        model_type = "sklearn"
        if "XGBoost" in model_name:
            model_type = "xgboost"
        elif "LightGBM" in model_name:
            model_type = "lightgbm"
        
        # Handle different metric key names
        metric_key = model_name
        if model_name == "Ridge" and "Ridge Regression" in metrics:
            metric_key = "Ridge Regression"
        elif model_name == "Random Forest" and "Random Forest" in metrics:
            metric_key = "Random Forest"
        
        registry.register_model(
            model=model,
            model_name=model_name.replace(" ", "_"),
            metrics=metrics[metric_key],
            model_type=model_type
        )
    
    # Register scaler as an artifact
    with mlflow.start_run(run_name="Feature_Scaler"):
        mlflow.sklearn.log_model(scaler, "scaler")
        mlflow.set_tag("model_type", "preprocessor")
        print("✅ Registered Feature Scaler")
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS REGISTERED IN MLFLOW!")
    print("=" * 70)
    
    # Show best model
    registry.get_best_model(metric="rmse")
    
    # Show comparison
    registry.compare_models()
    
    return registry


# Example usage
if __name__ == "__main__":
    import joblib
    
    # Load existing models
    print("Loading models from disk...")
    models = {
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'XGBoost': joblib.load('models/xgboost.pkl'),
        'LightGBM': joblib.load('models/lightgbm.pkl'),
        'Ridge Regression': joblib.load('models/ridge_regression.pkl')  # Match the metrics key
    }
    scaler = joblib.load('models/scaler.pkl')
    
    # Load metrics
    with open('models/model_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Register all models
    registry = register_all_models(models, metrics, scaler)
    
    print("\n🎯 To view MLflow UI, run:")
    print("   mlflow ui")
    print("\n   Then open: http://localhost:5000")