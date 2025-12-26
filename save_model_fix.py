"""Quick fix to train and save model."""

import pandas as pd # type: ignore
import pickle
import os
from sklearn.model_selection import train_test_split # type: ignore
from xgboost import XGBClassifier # type: ignore

# Paths
PROJECT_ROOT = r"D:\GitHub_Works\TrafficViolationPrediction_Analytics"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "processed_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_baseline_model.pkl")

# Features
FEATURE_COLS = [
    "coords_long_scaled", "coords_lat_scaled",
    "hour", "day_of_week",
    "is_weekend", "is_peak_hour", "is_night",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "junctionName_encoded", "vehicleType_encoded",
    "junction_risk_score", "vehicle_risk_score"
]
TARGET_COL = "alertType_encoded"

print("📁 Loading data...")
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"✅ Loaded {len(X)} samples")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🔧 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {acc:.4f}")

# Save
os.makedirs(MODELS_DIR, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# Verify
if os.path.exists(MODEL_PATH):
    print(f"✅ Model saved: {MODEL_PATH}")
else:
    print(f"❌ Failed to save!")

# Final check
print("\n📁 Models folder contents:")
for f in os.listdir(MODELS_DIR):
    print(f"   - {f}")