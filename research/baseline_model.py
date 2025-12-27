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

