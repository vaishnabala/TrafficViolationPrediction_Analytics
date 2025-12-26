"""
Explainable AI for Traffic Violation Prediction

Research Question:
    How can we provide interpretable explanations for traffic 
    violation predictions to support evidence-based policy making?

Methodology:
    - SHAP: Global and local feature importance
    - LIME: Local interpretable explanations
    - Compare both methods using rank correlation

References:
    - Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
    - Ribeiro et al. (2016). "Why Should I Trust You?: Explaining the Predictions"

Author: VaishnaBala
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle
import json
import os

import shap # type: ignore
import lime # type: ignore
import lime.lime_tabular # type: ignore

from scipy.stats import spearmanr # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


# ============ PATH SETUP ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

print(f"📁 Project Root: {PROJECT_ROOT}")
# ====================================


# Feature names for interpretability
FEATURE_COLS = [
    "coords_long_scaled", "coords_lat_scaled",
    "hour", "day_of_week",
    "is_weekend", "is_peak_hour", "is_night",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "junctionName_encoded", "vehicleType_encoded",
    "junction_risk_score", "vehicle_risk_score"
]

# Human-readable feature names for plots
FEATURE_NAMES_READABLE = [
    "Longitude", "Latitude",
    "Hour", "Day of Week",
    "Is Weekend", "Is Peak Hour", "Is Night",
    "Hour (sin)", "Hour (cos)", "Day (sin)", "Day (cos)",
    "Junction", "Vehicle Type",
    "Junction Risk", "Vehicle Risk"
]

CLASS_NAMES = ["RED_LIGHT", "SPEED", "WRONG_WAY"]


def load_model_and_data():
    """Load trained model and processed data."""
    
    # Load model
    model_path = os.path.join(MODELS_DIR, "best_baseline_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model not found: {model_path}\n"
            f"   Run baseline_model.py first"
        )
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Loaded model: {type(model).__name__}")
    
    # Load data
    data_path = os.path.join(DATA_DIR, "processed", "processed_data.csv")
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_COLS]
    y = df["alertType_encoded"]
    
    print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    return model, X, y


def run_shap_analysis(model, X, sample_size=1000):
    """
    Run SHAP analysis for global and local explanations.
    
    Args:
        model: Trained model
        X: Feature matrix
        sample_size: Number of samples for SHAP analysis
    
    Returns:
        dict: SHAP importance values
    """
    print("\n" + "=" * 50)
    print("SHAP ANALYSIS")
    print("=" * 50)
    
    # Sample data for efficiency
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()
    
    print(f"   Using {len(X_sample)} samples for SHAP analysis...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate global importance (mean |SHAP|)
    # Handle both multi-class (list) and binary (array) cases
    if isinstance(shap_values, list):
        # Multi-class: shap_values is a list of arrays [class0, class1, class2, ...]
        # Each array has shape (n_samples, n_features)
        # Average absolute values across all classes and samples
        shap_abs = np.abs(np.array(shap_values))  # Shape: (n_classes, n_samples, n_features)
        shap_importance = shap_abs.mean(axis=0).mean(axis=0)  # Shape: (n_features,)
    else:
        # Binary or regression: shap_values is a single array
        shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Ensure shap_importance is 1D array
    shap_importance = np.array(shap_importance).flatten()
    
    # Create importance dictionary
    importance_dict = {}
    for feat, imp in zip(FEATURE_COLS, shap_importance):
        importance_dict[feat] = float(imp)  # Ensure it's a float
    
    # Sort by importance
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    print("\n📊 Top 5 Important Features (SHAP):")
    for i, (feat, imp) in enumerate(list(importance_dict.items())[:5]):
        print(f"   {i+1}. {feat}: {imp:.4f}")
    
    # Generate plots
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Plot 1: Summary plot (beeswarm)
    print("\n   Generating SHAP summary plot...")
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=FEATURE_NAMES_READABLE,
            show=False,
            max_display=15
        )
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: shap_summary.png")
    except Exception as e:
        print(f"   ⚠️ Could not generate summary plot: {e}")
    
    # Plot 2: Bar plot (importance)
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X_sample,
            feature_names=FEATURE_NAMES_READABLE,
            plot_type="bar",
            show=False,
            max_display=15
        )
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_importance.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: shap_importance.png")
    except Exception as e:
        print(f"   ⚠️ Could not generate importance plot: {e}")
    
    # Plot 3: Custom bar plot (guaranteed to work)
    try:
        plt.figure(figsize=(10, 6))
        sorted_features = list(importance_dict.keys())
        sorted_values = list(importance_dict.values())
        
        plt.barh(range(len(sorted_features)), sorted_values, color="steelblue")
        plt.yticks(range(len(sorted_features)), sorted_features, fontsize=9)
        plt.xlabel("Mean |SHAP Value|", fontsize=12)
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_importance_custom.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: shap_importance_custom.png")
    except Exception as e:
        print(f"   ⚠️ Could not generate custom plot: {e}")
    
    return {
        "importance": importance_dict,
        "sample_size": len(X_sample)
    }


def run_lime_analysis(model, X, num_explanations=100):
    """
    Run LIME analysis for local explanations.
    
    LIME (Local Interpretable Model-agnostic Explanations):
    - Creates local linear approximations
    - Model-agnostic (works with any classifier)
    - Provides instance-level explanations
    
    Args:
        model: Trained model
        X: Feature matrix
        num_explanations: Number of instances to explain
    
    Returns:
        dict: LIME importance values
    """
    print("\n" + "=" * 50)
    print("LIME ANALYSIS")
    print("=" * 50)
    
    print(f"   Generating {num_explanations} LIME explanations...")
    
    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=FEATURE_COLS,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=42
    )
    
    # Sample instances to explain
    sample_indices = np.random.choice(len(X), size=min(num_explanations, len(X)), replace=False)
    
    # Collect feature importances across explanations
    feature_importances = {feat: [] for feat in FEATURE_COLS}
    
    for i, idx in enumerate(sample_indices):
        if (i + 1) % 20 == 0:
            print(f"   Processing explanation {i+1}/{num_explanations}...")
        
        exp = lime_explainer.explain_instance(
            X.iloc[idx].values,
            model.predict_proba,
            num_features=len(FEATURE_COLS),
            top_labels=1
        )
        
        # Extract feature importance
        exp_list = exp.as_list(label=exp.available_labels()[0])
        
        for feat_cond, importance in exp_list:
            # Match feature name (LIME adds conditions like "hour <= 5")
            for feat in FEATURE_COLS:
                if feat in feat_cond:
                    feature_importances[feat].append(abs(importance))
                    break
    
    # Average importance per feature
    lime_importance = {
        feat: np.mean(vals) if vals else 0 
        for feat, vals in feature_importances.items()
    }
    lime_importance = dict(sorted(lime_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\n📊 Top 5 Important Features (LIME):")
    for i, (feat, imp) in enumerate(list(lime_importance.items())[:5]):
        print(f"   {i+1}. {feat}: {imp:.4f}")
    
    # Generate LIME importance plot
    plt.figure(figsize=(10, 6))
    features = list(lime_importance.keys())
    values = list(lime_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(values)[::-1]
    
    plt.barh(
        range(len(features)),
        [values[i] for i in sorted_idx],
        color="coral"
    )
    plt.yticks(range(len(features)), [features[i] for i in sorted_idx], fontsize=9)
    plt.xlabel("Mean |Importance|", fontsize=12)
    plt.title("LIME Feature Importance (Averaged)", fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "lime_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: lime_importance.png")
    
    return {
        "importance": lime_importance,
        "num_explanations": num_explanations
    }


def compare_shap_lime(shap_results, lime_results):
    """
    Compare SHAP and LIME feature rankings using Spearman correlation.
    """
    print("\n" + "=" * 50)
    print("SHAP vs LIME COMPARISON")
    print("=" * 50)
    
    shap_imp = shap_results["importance"]
    lime_imp = lime_results["importance"]
    
    # Get rankings
    shap_ranking = list(shap_imp.keys())
    lime_ranking = list(lime_imp.keys())
    
    # Calculate ranks for each feature
    shap_ranks = {feat: rank for rank, feat in enumerate(shap_ranking)}
    lime_ranks = {feat: rank for rank, feat in enumerate(lime_ranking)}
    
    # Create rank arrays (same order for both)
    features = FEATURE_COLS
    shap_rank_arr = [shap_ranks.get(f, len(features)) for f in features]
    lime_rank_arr = [lime_ranks.get(f, len(features)) for f in features]
    
    # Spearman correlation
    correlation, p_value = spearmanr(shap_rank_arr, lime_rank_arr)
    
    # Handle NaN correlation
    if np.isnan(correlation):
        correlation = 0.0
        p_value = 1.0
    
    print(f"\n📊 Spearman Rank Correlation:")
    print(f"   ρ (rho) = {correlation:.4f}")
    print(f"   p-value = {p_value:.4f}")
    
    # Interpretation
    if abs(correlation) > 0.7:
        interpretation = "Strong agreement"
    elif abs(correlation) > 0.4:
        interpretation = "Moderate agreement"
    else:
        interpretation = "Weak agreement"
    
    print(f"   Interpretation: {interpretation}")
    
    # Generate comparison plot
    _plot_comparison(shap_imp, lime_imp)
    
    return {
        "spearman_correlation": float(correlation),
        "p_value": float(p_value),
        "interpretation": interpretation,
        "shap_top5": list(shap_imp.keys())[:5],
        "lime_top5": list(lime_imp.keys())[:5]
    }

def _plot_comparison(shap_imp, lime_imp):
    """Generate SHAP vs LIME comparison plot."""
    
    features = FEATURE_COLS
    
    # Normalize values for comparison
    shap_vals = np.array([shap_imp.get(f, 0) for f in features])
    lime_vals = np.array([lime_imp.get(f, 0) for f in features])
    
    if shap_vals.max() > 0:
        shap_norm = shap_vals / shap_vals.max()
    else:
        shap_norm = shap_vals
        
    if lime_vals.max() > 0:
        lime_norm = lime_vals / lime_vals.max()
    else:
        lime_norm = lime_vals
    
    # Sort by SHAP importance
    sorted_idx = np.argsort(shap_norm)[::-1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(features))
    width = 0.35
    
    ax.barh(x - width/2, [shap_norm[i] for i in sorted_idx], width, label="SHAP", color="steelblue")
    ax.barh(x + width/2, [lime_norm[i] for i in sorted_idx], width, label="LIME", color="coral")
    
    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title("SHAP vs LIME Feature Importance Comparison", fontsize=14)
    ax.set_yticks(x)
    ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "xai_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Saved: xai_comparison.png")


def save_results(shap_results, lime_results, comparison_results):
    """Save all explainability results to JSON."""
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        "shap": shap_results,
        "lime": lime_results,
        "comparison": comparison_results
    }
    
    output_path = os.path.join(RESULTS_DIR, "explainability_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n✅ Saved results: {output_path}")
    
    return results


def run_explainability_analysis():
    """
    Main function to run complete XAI analysis.
    
    Pipeline:
        1. Load model and data
        2. Run SHAP analysis
        3. Run LIME analysis
        4. Compare both methods
        5. Save results
    """
    print("\n" + "=" * 60)
    print(" RESEARCH EXPERIMENT 2: EXPLAINABLE AI ANALYSIS")
    print("=" * 60)
    
    # 1. Load model and data
    model, X, y = load_model_and_data()
    
    # 2. SHAP analysis
    shap_results = run_shap_analysis(model, X, sample_size=1000)
    
    # 3. LIME analysis
    lime_results = run_lime_analysis(model, X, num_explanations=100)
    
    # 4. Compare methods
    comparison_results = compare_shap_lime(shap_results, lime_results)
    
    # 5. Save results
    results = save_results(shap_results, lime_results, comparison_results)
    
    print("\n" + "=" * 60)
    print(" EXPERIMENT 2 COMPLETE")
    print("=" * 60)
    
    print("\n📁 Generated files:")
    print(f"   - {FIGURES_DIR}/shap_summary.png")
    print(f"   - {FIGURES_DIR}/shap_importance.png")
    print(f"   - {FIGURES_DIR}/shap_force_plot.png")
    print(f"   - {FIGURES_DIR}/lime_importance.png")
    print(f"   - {FIGURES_DIR}/xai_comparison.png")
    print(f"   - {RESULTS_DIR}/explainability_results.json")
    
    return results


if __name__ == "__main__":
    results = run_explainability_analysis()