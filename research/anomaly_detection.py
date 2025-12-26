"""
Anomaly Detection in Traffic Violations

Research Question:
    Can we detect unusual traffic violation patterns that may indicate
    system malfunctions, special events, or emerging hotspots?

Methodology:
    - Isolation Forest (tree-based, unsupervised)
    - Autoencoder (deep learning, reconstruction-based)
    - Compare detection performance on synthetic anomalies

References:
    - Liu et al. (2008). "Isolation Forest"
    - Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"

Author: VaishnaBala
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import json
import os
import warnings

from sklearn.ensemble import IsolationForest # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.metrics import ( # type: ignore
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

warnings.filterwarnings('ignore')

# TensorFlow import with error handling
try:
    import tensorflow as tf # type: ignore
    from tensorflow import keras # type: ignore
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available - Autoencoder will be skipped")


# ============ PATH SETUP ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

print(f"📁 Project Root: {PROJECT_ROOT}")
# ====================================


# Features for anomaly detection
FEATURE_COLS = [
    "coords_long_scaled", "coords_lat_scaled",
    "hour", "day_of_week",
    "is_weekend", "is_peak_hour", "is_night",
    "junctionName_encoded", "vehicleType_encoded"
]


def load_data():
    """Load processed data for anomaly detection."""
    
    data_path = os.path.join(DATA_DIR, "processed", "processed_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS].values
    
    print(f"✅ Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X


def inject_synthetic_anomalies(X, anomaly_rate=0.05, random_state=42):
    """
    Inject synthetic anomalies for evaluation.
    
    Anomalies are created by generating points outside normal data distribution.
    This allows us to evaluate anomaly detection performance.
    
    Args:
        X: Original data
        anomaly_rate: Fraction of anomalies to inject
        random_state: Random seed
    
    Returns:
        X_with_anomalies: Data with injected anomalies
        y_true: Labels (0=normal, 1=anomaly)
    """
    np.random.seed(random_state)
    
    n_anomalies = int(len(X) * anomaly_rate)
    
    # Generate anomalies: points far from normal distribution
    # Method: Use values outside 3 standard deviations
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    
    anomalies = np.random.uniform(
        low=X_mean - 4 * X_std,
        high=X_mean + 4 * X_std,
        size=(n_anomalies, X.shape[1])
    )
    
    # Combine normal and anomaly data
    X_combined = np.vstack([X, anomalies])
    y_true = np.array([0] * len(X) + [1] * n_anomalies)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_idx]
    y_true = y_true[shuffle_idx]
    
    print(f"   Injected {n_anomalies} synthetic anomalies ({anomaly_rate*100:.1f}%)")
    print(f"   Total samples: {len(X_combined)}")
    
    return X_combined, y_true


def run_isolation_forest(X, y_true, contamination=0.05):
    """
    Run Isolation Forest anomaly detection.
    
    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies require fewer splits to isolate.
    
    Args:
        X: Feature matrix
        y_true: True labels
        contamination: Expected proportion of anomalies
    
    Returns:
        dict: Results including predictions and metrics
    """
    print("\n" + "-" * 40)
    print("ISOLATION FOREST")
    print("-" * 40)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit and predict
    y_pred_raw = iso_forest.fit_predict(X)
    
    # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
    y_pred = (y_pred_raw == -1).astype(int)
    
    # Get anomaly scores
    anomaly_scores = -iso_forest.decision_function(X)  # Higher = more anomalous
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n📊 Results:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "n_detected": int(y_pred.sum()),
        "n_actual": int(y_true.sum())
    }
    
    return results, y_pred, anomaly_scores, iso_forest


def run_autoencoder(X, y_true, contamination=0.05):
    """
    Run Autoencoder-based anomaly detection.
    
    Autoencoder learns to reconstruct normal data. Anomalies have
    higher reconstruction error since they differ from training data.
    
    Args:
        X: Feature matrix
        y_true: True labels
        contamination: Expected proportion of anomalies
    
    Returns:
        dict: Results including predictions and metrics
    """
    if not TF_AVAILABLE:
        print("\n⚠️ Skipping Autoencoder (TensorFlow not installed)")
        return None, None, None, None
    
    print("\n" + "-" * 40)
    print("AUTOENCODER")
    print("-" * 40)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split normal data for training (use only non-anomalies)
    X_normal = X_scaled[y_true == 0]
    
    print(f"   Training on {len(X_normal)} normal samples...")
    
    # Build Autoencoder
    input_dim = X_scaled.shape[1]
    encoding_dim = max(2, input_dim // 3)  # Bottleneck size
    
    autoencoder = keras.Sequential([ # type: ignore
        keras.layers.Dense(16, activation="relu", input_shape=(input_dim,)), # pyright: ignore[reportPossiblyUnboundVariable]
        keras.layers.Dense(encoding_dim, activation="relu"), # pyright: ignore[reportPossiblyUnboundVariable]
        keras.layers.Dense(16, activation="relu"), # type: ignore
        keras.layers.Dense(input_dim, activation="linear") # type: ignore
    ])
    
    autoencoder.compile(optimizer="adam", loss="mse")
    
    # Train (suppress output)
    autoencoder.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    print("   ✅ Training complete")
    
    # Calculate reconstruction error for all data
    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    
    # Set threshold based on contamination rate
    threshold = np.percentile(mse, (1 - contamination) * 100)
    
    # Predict anomalies
    y_pred = (mse > threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n📊 Results:")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "n_detected": int(y_pred.sum()),
        "n_actual": int(y_true.sum())
    }
    
    return results, y_pred, mse, autoencoder


def compare_methods(iso_results, ae_results):
    """Compare Isolation Forest and Autoencoder performance."""
    
    print("\n" + "=" * 50)
    print("METHOD COMPARISON")
    print("=" * 50)
    
    comparison = {}
    
    print("\n📊 Performance Summary:")
    print("-" * 45)
    print(f"{'Metric':<15} {'Isolation Forest':<18} {'Autoencoder':<15}")
    print("-" * 45)
    
    metrics = ["precision", "recall", "f1"]
    
    for metric in metrics:
        iso_val = iso_results[metric]
        ae_val = ae_results[metric] if ae_results else "N/A"
        
        if ae_results:
            print(f"{metric.capitalize():<15} {iso_val:<18.4f} {ae_val:<15.4f}")
        else:
            print(f"{metric.capitalize():<15} {iso_val:<18.4f} {ae_val:<15}")
    
    print("-" * 45)
    
    # Determine winner
    if ae_results:
        if iso_results["f1"] > ae_results["f1"]:
            winner = "Isolation Forest"
        elif ae_results["f1"] > iso_results["f1"]:
            winner = "Autoencoder"
        else:
            winner = "Tie"
        
        comparison["winner"] = winner
        comparison["f1_difference"] = abs(iso_results["f1"] - ae_results["f1"])
        print(f"\n🏆 Winner: {winner}")
    else:
        comparison["winner"] = "Isolation Forest (Autoencoder skipped)"
    
    return comparison


def plot_anomaly_comparison(iso_results, ae_results):
    """Generate comparison bar plot."""
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    metrics = ["precision", "recall", "f1"]
    iso_vals = [iso_results[m] for m in metrics]
    
    if ae_results:
        ae_vals = [ae_results[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, iso_vals, width, label="Isolation Forest", color="steelblue")
        bars2 = ax.bar(x + width/2, ae_vals, width, label="Autoencoder", color="coral")
        
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Anomaly Detection Method Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(metrics, iso_vals, color="steelblue")
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Isolation Forest Performance", fontsize=14)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "anomaly_comparison.png"), dpi=300)
    plt.close()
    print(f"✅ Saved: anomaly_comparison.png")


def plot_anomaly_scores(y_true, iso_scores, ae_scores=None):
    """Plot anomaly score distributions."""
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    n_plots = 2 if ae_scores is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Isolation Forest scores
    ax1 = axes[0]
    ax1.hist(iso_scores[y_true == 0], bins=50, alpha=0.7, label="Normal", color="steelblue")
    ax1.hist(iso_scores[y_true == 1], bins=50, alpha=0.7, label="Anomaly", color="coral")
    ax1.set_xlabel("Anomaly Score", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Isolation Forest Anomaly Scores", fontsize=14)
    ax1.legend()
    
    # Plot 2: Autoencoder reconstruction error
    if ae_scores is not None:
        ax2 = axes[1]
        ax2.hist(ae_scores[y_true == 0], bins=50, alpha=0.7, label="Normal", color="steelblue")
        ax2.hist(ae_scores[y_true == 1], bins=50, alpha=0.7, label="Anomaly", color="coral")
        ax2.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Autoencoder Reconstruction Error", fontsize=14)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "anomaly_scores.png"), dpi=300)
    plt.close()
    print(f"✅ Saved: anomaly_scores.png")


def plot_confusion_matrices(y_true, iso_pred, ae_pred=None):
    """Plot confusion matrices for both methods."""
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    n_plots = 2 if ae_pred is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    labels = ["Normal", "Anomaly"]
    
    # Isolation Forest
    cm1 = confusion_matrix(y_true, iso_pred)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title("Isolation Forest", fontsize=14)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # Autoencoder
    if ae_pred is not None:
        cm2 = confusion_matrix(y_true, ae_pred)
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                    xticklabels=labels, yticklabels=labels)
        axes[1].set_title("Autoencoder", fontsize=14)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "anomaly_confusion_matrices.png"), dpi=300)
    plt.close()
    print(f"✅ Saved: anomaly_confusion_matrices.png")


def save_results(iso_results, ae_results, comparison):
    """Save all anomaly detection results."""
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        "isolation_forest": iso_results,
        "autoencoder": ae_results,
        "comparison": comparison
    }
    
    output_path = os.path.join(RESULTS_DIR, "anomaly_detection_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"✅ Saved: {output_path}")
    
    return results


def run_anomaly_detection():
    """
    Main function to run anomaly detection experiments.
    
    Pipeline:
        1. Load data
        2. Inject synthetic anomalies
        3. Run Isolation Forest
        4. Run Autoencoder
        5. Compare methods
        6. Generate visualizations
        7. Save results
    """
    print("\n" + "=" * 60)
    print(" RESEARCH EXPERIMENT 3: ANOMALY DETECTION")
    print("=" * 60)
    
    # 1. Load data
    X = load_data()
    
    # 2. Inject synthetic anomalies
    print("\n📊 Preparing data with synthetic anomalies...")
    X_with_anomalies, y_true = inject_synthetic_anomalies(X, anomaly_rate=0.05)
    
    # 3. Isolation Forest
    iso_results, iso_pred, iso_scores, iso_model = run_isolation_forest(
        X_with_anomalies, y_true, contamination=0.05
    )
    
    # 4. Autoencoder
    ae_results, ae_pred, ae_scores, ae_model = run_autoencoder(
        X_with_anomalies, y_true, contamination=0.05
    )
    
    # 5. Compare methods
    comparison = compare_methods(iso_results, ae_results)
    
    # 6. Generate plots
    print("\n📊 Generating visualizations...")
    plot_anomaly_comparison(iso_results, ae_results)
    plot_anomaly_scores(y_true, iso_scores, ae_scores)
    plot_confusion_matrices(y_true, iso_pred, ae_pred)
    
    # 7. Save results
    print("\n📁 Saving results...")
    results = save_results(iso_results, ae_results, comparison)
    
    print("\n" + "=" * 60)
    print(" EXPERIMENT 3 COMPLETE")
    print("=" * 60)
    
    print("\n📁 Generated files:")
    print(f"   - {FIGURES_DIR}/anomaly_comparison.png")
    print(f"   - {FIGURES_DIR}/anomaly_scores.png")
    print(f"   - {FIGURES_DIR}/anomaly_confusion_matrices.png")
    print(f"   - {RESULTS_DIR}/anomaly_detection_results.json")
    
    return results


if __name__ == "__main__":
    results = run_anomaly_detection()