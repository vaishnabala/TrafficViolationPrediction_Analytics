"""
Data Preprocessing Pipeline

Transforms raw traffic violation data into ML-ready features.

Author: VaishnaBala
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
import pickle
import os
import sys


# ============ PATH SETUP ============
# Get directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Since script is in project root, PROJECT_ROOT = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR

# Define paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "traffic_violations.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Raw Data: {RAW_DATA_PATH}")
print(f"📁 Processed Dir: {PROCESSED_DIR}")
print(f"📁 Models Dir: {MODELS_DIR}")
# ====================================


def load_raw_data():
    """Load raw traffic violation data."""
    
    # Check if file exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"\n❌ Raw data not found: {RAW_DATA_PATH}\n"
            f"   Run data generator first:\n"
            f"   python data_generator.py\n"
        )
    
    df = pd.read_csv(RAW_DATA_PATH)
    df["observationDateTime"] = pd.to_datetime(df["observationDateTime"])
    
    print(f"✅ Loaded {len(df)} records")
    return df


def extract_temporal_features(df):
    """Extract time-based features from datetime."""
    df = df.copy()
    
    df["hour"] = df["observationDateTime"].dt.hour
    df["day_of_week"] = df["observationDateTime"].dt.dayofweek
    df["month"] = df["observationDateTime"].dt.month
    
    # Binary flags
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    print("✅ Extracted temporal features")
    return df


def add_cyclical_encoding(df):
    """Add cyclical encoding for hour and day_of_week."""
    df = df.copy()
    
    # Hour: 24-hour cycle
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Day of week: 7-day cycle
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    print("✅ Added cyclical encoding")
    return df


def encode_categorical(df, categorical_cols):
    """Encode categorical variables using LabelEncoder."""
    df = df.copy()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} classes")
    
    print("✅ Encoded categorical variables")
    return df, encoders


def scale_coordinates(df):
    """Standardize coordinates using StandardScaler."""
    df = df.copy()
    
    scaler = StandardScaler()
    df[["coords_long_scaled", "coords_lat_scaled"]] = scaler.fit_transform(
        df[["coordinates_long", "coordinates_lat"]]
    )
    
    print("✅ Scaled coordinates")
    return df, scaler


def save_artifacts(encoders, scaler):
    """Save encoders and scaler for inference."""
    
    # Create directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"   Created directory: {MODELS_DIR}")
    
    # Combine all artifacts
    artifacts = {**encoders, "coord_scaler": scaler}
    
    output_path = os.path.join(MODELS_DIR, "encoders.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(artifacts, f)
    
    # Verify file was created
    if os.path.exists(output_path):
        print(f"✅ Saved encoders: {output_path}")
    else:
        print(f"❌ Failed to save: {output_path}")


def save_processed_data(df):
    """Save processed data to CSV."""
    
    # Create directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"   Created directory: {PROCESSED_DIR}")
    
    # Define feature columns
    feature_cols = [
        "coordinates_long", "coordinates_lat",
        "coords_long_scaled", "coords_lat_scaled",
        "hour", "day_of_week", "month",
        "is_weekend", "is_peak_hour", "is_night",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "junctionName_encoded", "vehicleType_encoded",
        "junction_risk_score", "vehicle_risk_score",
        "alertType_encoded"
    ]
    
    # Check if all columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"   Available columns: {df.columns.tolist()}")
        return
    
    # Save feature-only data
    feature_path = os.path.join(PROCESSED_DIR, "processed_data.csv")
    df[feature_cols].to_csv(feature_path, index=False)
    
    # Verify
    if os.path.exists(feature_path):
        print(f"✅ Saved features: {feature_path}")
    else:
        print(f"❌ Failed to save: {feature_path}")
    
    # Save full data
    full_path = os.path.join(PROCESSED_DIR, "full_processed_data.csv")
    df.to_csv(full_path, index=False)
    
    if os.path.exists(full_path):
        print(f"✅ Saved full data: {full_path}")
    else:
        print(f"❌ Failed to save: {full_path}")


def preprocess_pipeline():
    """Run complete preprocessing pipeline."""
    
    print("\n" + "=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50 + "\n")
    
    try:
        # Step 1: Load
        df = load_raw_data()
        print(f"   Columns: {df.columns.tolist()}\n")
        
        # Step 2: Temporal features
        df = extract_temporal_features(df)
        
        # Step 3: Cyclical encoding
        df = add_cyclical_encoding(df)
        
        # Step 4: Categorical encoding
        categorical_cols = ["junctionName", "vehicleType", "alertType"]
        df, encoders = encode_categorical(df, categorical_cols)
        
        # Step 5: Scale coordinates
        df, scaler = scale_coordinates(df)
        
        # Step 6: Save
        print("\n📁 Saving files...")
        save_artifacts(encoders, scaler)
        save_processed_data(df)
        
        # Summary
        print("\n" + "=" * 50)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 50)
        print(f"\n📊 Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = preprocess_pipeline()
    
    if df is not None:
        print("\n📋 Sample data:")
        print(df.head())
    
    # Final verification
    print("\n" + "=" * 50)
    print("FILE VERIFICATION")
    print("=" * 50)
    
    files_to_check = [
        os.path.join(PROCESSED_DIR, "processed_data.csv"),
        os.path.join(PROCESSED_DIR, "full_processed_data.csv"),
        os.path.join(MODELS_DIR, "encoders.pkl")
    ]
    
    for f in files_to_check:
        status = "✅" if os.path.exists(f) else "❌"
        print(f"   {status} {f}")