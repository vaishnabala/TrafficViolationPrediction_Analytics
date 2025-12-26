"""
Baseline Model Comparison for Traffic Violation Prediction

Research Question:
    How do traditional ML models compare for spatio-temporal 
    traffic violation prediction?

Author: VaishnaBala
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import pickle
import os
import sys

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestClassifier # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (  # type: ignore
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier # type: ignore
from scipy import stats # type: ignore

import mlflow # type: ignore
import mlflow.sklearn # type: ignore

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


# ============ PATH SETUP ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go UP from research/ to root

# Define paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Data Dir: {DATA_DIR}")
print(f"📁 Models Dir: {MODELS_DIR}")
print(f"📁 Results Dir: {RESULTS_DIR}")
# ====================================

def run_baseline_experiments():
    """Main function to run all baseline experiments."""
    
    print("\n" + "=" * 60)
    print(" RESEARCH EXPERIMENT 1: BASELINE MODEL COMPARISON")
    print("=" * 60)
    
    # 1. Load data
    X, y = load_data()
    
    # 2. Get models
    models = get_models()
    
    # 3. Cross-validation
    cv_results = run_cross_validation(X, y, models)
    
    # 4. Statistical tests
    stat_results = statistical_significance_test(cv_results)
    
    # 5. Final training
    final_results, trained_models, (X_test, y_test) = train_final_models(X, y, models)
    
    # 6. Select and SAVE best model  <-- THIS IS IMPORTANT!
    best_name, best_model = select_best_model(final_results, trained_models)
    
    # 7. Plots
    plot_cv_comparison(cv_results)
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_feature_importance(trained_models, FEATURE_COLS)
    
    # 8. Save results
    results = save_results(cv_results, stat_results, final_results, best_name)
    
    print("\n" + "=" * 60)
    print(" EXPERIMENT 1 COMPLETE")
    print("=" * 60)
    
    return results