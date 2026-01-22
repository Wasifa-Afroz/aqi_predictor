"""
Model Explainability using SHAP
Required by PDF: "Use SHAP or LIME for feature importance explanations"
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load trained model and test data"""
    print("📦 Loading model and data...")
    
    # Load best model (Random Forest)
    model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Load training data
    df = pd.read_csv('data/processed/training_data.csv')
    
    # Prepare features
    exclude_cols = ['timestamp', 'city', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    print(f"✅ Loaded Random Forest model")
    print(f"✅ Loaded {len(df)} samples with {len(feature_cols)} features")
    
    return model, X_scaled, feature_names, df

def calculate_shap_values(model, X, feature_names, sample_size=100):
    """Calculate SHAP values for feature importance"""
    print(f"\n🔍 Calculating SHAP values (using {sample_size} samples)...")
    
    # Use a sample for faster computation
    X_sample = X[:sample_size]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"✅ SHAP values calculated")
    
    return explainer, shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, feature_names):
    """Create SHAP summary plot"""
    print("\n📊 Creating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/shap_summary.png")
    plt.close()

def plot_shap_bar(shap_values, feature_names):
    """Create SHAP feature importance bar plot"""
    print("\n📊 Creating SHAP feature importance bar plot...")
    
    # Calculate mean absolute SHAP values
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP value| (Average Impact on Model Output)')
    plt.title('Top 20 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('visualizations/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/shap_feature_importance.png")
    plt.close()
    
    return importance_df

def plot_shap_waterfall(explainer, shap_values, X_sample, feature_names, idx=0):
    """Create SHAP waterfall plot for a single prediction"""
    print(f"\n📊 Creating SHAP waterfall plot for sample {idx}...")
    
    plt.figure(figsize=(10, 8))
    
    # Create explanation object for waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_sample[idx],
            feature_names=feature_names
        ),
        show=False
    )
    
    plt.tight_layout()
    plt.savefig('visualizations/shap_waterfall.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/shap_waterfall.png")
    plt.close()

def plot_shap_dependence(shap_values, X_sample, feature_names, top_n=3):
    """Create SHAP dependence plots for top features"""
    print(f"\n📊 Creating SHAP dependence plots for top {top_n} features...")
    
    # Get top features by mean absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-top_n:][::-1]
    
    fig, axes = plt.subplots(1, top_n, figsize=(15, 4))
    
    for i, idx in enumerate(top_indices):
        feature_name = feature_names[idx]
        ax = axes[i] if top_n > 1 else axes
        
        shap.dependence_plot(
            idx, shap_values, X_sample,
            feature_names=feature_names,
            show=False,
            ax=ax
        )
        ax.set_title(f'SHAP Dependence: {feature_name}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/shap_dependence.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: visualizations/shap_dependence.png")
    plt.close()

def save_feature_importance_report(importance_df):
    """Save feature importance report"""
    print("\n💾 Saving feature importance report...")
    
    # Save top 30 features
    os.makedirs('reports', exist_ok=True)
    importance_df.head(30).to_csv('reports/feature_importance.csv', index=False)
    
    # Create text report
    with open('reports/feature_importance_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS (SHAP VALUES)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Top 20 Most Important Features:\n")
        f.write("-" * 70 + "\n\n")
        
        for idx, row in importance_df.head(20).iterrows():
            f.write(f"{idx+1:2d}. {row['feature']:30s} - Importance: {row['importance']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nKey Insights:\n")
        f.write("-" * 70 + "\n")
        f.write(f"• Most important feature: {importance_df.iloc[0]['feature']}\n")
        f.write(f"• Top 3 features account for {importance_df.head(3)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}% of total importance\n")
        f.write(f"• Total features analyzed: {len(importance_df)}\n")
    
    print("✅ Saved: reports/feature_importance.csv")
    print("✅ Saved: reports/feature_importance_report.txt")

def main():
    """Main explainability pipeline"""
    print("\n" + "=" * 70)
    print("🔍 MODEL EXPLAINABILITY - SHAP ANALYSIS")
    print("=" * 70)
    
    # Load model and data
    model, X_scaled, feature_names, df = load_model_and_data()
    
    # Calculate SHAP values (using 100 samples for speed)
    explainer, shap_values, X_sample = calculate_shap_values(
        model, X_scaled, feature_names, sample_size=100
    )
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("📊 CREATING SHAP VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Summary plot (beeswarm)
    plot_shap_summary(shap_values, X_sample, feature_names)
    
    # 2. Feature importance bar plot
    importance_df = plot_shap_bar(shap_values, feature_names)
    
    # 3. Waterfall plot (single prediction)
    plot_shap_waterfall(explainer, shap_values, X_sample, feature_names, idx=0)
    
    # 4. Dependence plots (top 3 features)
    plot_shap_dependence(shap_values, X_sample, feature_names, top_n=3)
    
    # Save reports
    save_feature_importance_report(importance_df)
    
    # Display top 10 features
    print("\n" + "=" * 70)
    print("🏆 TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 70)
    print("\n" + importance_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   📊 visualizations/shap_summary.png")
    print("   📊 visualizations/shap_feature_importance.png")
    print("   📊 visualizations/shap_waterfall.png")
    print("   📊 visualizations/shap_dependence.png")
    print("   📄 reports/feature_importance.csv")
    print("   📄 reports/feature_importance_report.txt")
    
    print("\n💡 These visualizations show:")
    print("   • Which features most impact AQI predictions")
    print("   • How feature values affect predictions")
    print("   • Individual prediction explanations")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Explainability analysis failed!")
        sys.exit(1)