"""
Traffic Violation Data Generator

Generates synthetic data with realistic spatio-temporal patterns:
- Peak hour violations (morning/evening rush)
- Junction-specific risk profiles
- Vehicle-type based violation tendencies

Author: VaishnaBala
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import random
from datetime import datetime, timedelta
import os


# ============ PATH SETUP ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # Adjust if script is in src/ folder

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DATA_DIR, "traffic_violations.csv")

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Output Path: {OUTPUT_PATH}")
# ====================================


# Configuration
JUNCTIONS = {
    "JPnagar-8thmain-9thcross": {"coords": [77.595183, 12.910289], "risk": 0.7},
    "Koramangala-5thblock": {"coords": [77.612320, 12.935620], "risk": 0.8},
    "Whitefield-main-road": {"coords": [77.750277, 12.969720], "risk": 0.5},
    "MG-Road-brigade": {"coords": [77.606430, 12.975040], "risk": 0.9},
    "Indiranagar-100ft-road": {"coords": [77.640877, 12.978012], "risk": 0.75},
    "Electronic-city-toll": {"coords": [77.670067, 12.839547], "risk": 0.4},
    "Hebbal-flyover": {"coords": [77.591508, 13.035542], "risk": 0.85},
    "Silk-board-junction": {"coords": [77.622627, 12.917153], "risk": 0.95}
}

VEHICLES = {
    "Motorbike": 0.8,
    "Car": 0.5,
    "Bus": 0.3,
    "Truck": 0.4,
    "Auto": 0.6
}

ALERT_TYPES = ["RED_LIGHT_VIOLATION", "SPEED_VIOLATION", "WRONG_WAY"]


def get_hourly_distribution():
    """
    Returns probability distribution for each hour.
    Peak hours: 8-10 AM and 5-8 PM
    
    Note: Probabilities are normalized to sum to exactly 1.
    """
    # Raw weights (don't need to sum to 1)
    weights = [
        2, 1, 1, 1, 2, 3,      # 0-5   (night - low)
        4, 6, 8, 9, 7, 5,      # 6-11  (morning rush)
        4, 4, 4, 5, 6, 8,      # 12-17 (afternoon)
        9, 7, 5, 4, 3, 2       # 18-23 (evening rush then decline)
    ]
    
    # Convert to numpy and normalize to sum to 1
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    
    return probs


def generate_record(start_date):
    """Generate a single violation record."""
    
    # Select junction (weighted by risk)
    junction_names = list(JUNCTIONS.keys())
    junction_weights = [JUNCTIONS[j]["risk"] for j in junction_names]
    junction = random.choices(junction_names, weights=junction_weights)[0]
    
    # Select hour (weighted by traffic pattern)
    hour = np.random.choice(range(24), p=get_hourly_distribution())
    
    # Select vehicle (weighted by violation tendency)
    vehicle = random.choices(
        list(VEHICLES.keys()),
        weights=list(VEHICLES.values())
    )[0]
    
    # Select alert type (context-dependent)
    if vehicle == "Motorbike" and hour in [22, 23, 0, 1, 2]:
        alert_weights = [0.3, 0.6, 0.1]  # More speed violations at night
    else:
        alert_weights = [0.6, 0.3, 0.1]
    
    alert_type = random.choices(ALERT_TYPES, weights=alert_weights)[0]
    
    # Generate timestamp
    obs_datetime = start_date + timedelta(
        days=random.randint(0, 180),
        hours=int(hour),
        minutes=random.randint(0, 59)
    )
    
    # Build record
    record = {
        "alertType": alert_type,
        "junctionName": junction,
        "coordinates_long": JUNCTIONS[junction]["coords"][0] + np.random.normal(0, 0.001),
        "coordinates_lat": JUNCTIONS[junction]["coords"][1] + np.random.normal(0, 0.001),
        "vehicleType": vehicle,
        "observationDateTime": obs_datetime,
        "junction_risk_score": JUNCTIONS[junction]["risk"],
        "vehicle_risk_score": VEHICLES[vehicle]
    }
    
    return record


def generate_data(n_samples=10000):
    """
    Generate synthetic traffic violation dataset.
    
    Args:
        n_samples: Number of records to generate
    
    Returns:
        DataFrame with generated data
    """
    print("\n" + "=" * 50)
    print("DATA GENERATION")
    print("=" * 50 + "\n")
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime(2024, 1, 1)
    
    # Generate records
    print(f"   Generating {n_samples} records...")
    data = [generate_record(start_date) for _ in range(n_samples)]
    
    df = pd.DataFrame(data)
    df = df.sort_values("observationDateTime").reset_index(drop=True)
    
    # Create directory
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"   Created directory: {RAW_DATA_DIR}")
    
    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Verify
    if os.path.exists(OUTPUT_PATH):
        print(f"✅ Saved: {OUTPUT_PATH}")
    else:
        print(f"❌ Failed to save: {OUTPUT_PATH}")
        return None
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Records: {len(df)}")
    print(f"   Date range: {df['observationDateTime'].min()} to {df['observationDateTime'].max()}")
    print(f"   Junctions: {df['junctionName'].nunique()}")
    print(f"   Vehicle types: {df['vehicleType'].nunique()}")
    print(f"   Alert types: {df['alertType'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    df = generate_data()
    
    if df is not None:
        print("\n📋 Sample data:")
        print(df.head())
        
        # Verify probability distribution worked
        print("\n📊 Hour distribution:")
        df['observationDateTime'] = pd.to_datetime(df['observationDateTime'])
        print(df['observationDateTime'].dt.hour.value_counts().sort_index()) # type: ignore