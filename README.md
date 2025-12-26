# 🚦 Traffic Violation Prediction & Explainable Analytics System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://docker.com)

## 📄 Abstract

An end-to-end machine learning system that analyzes traffic violation data from ANPR cameras to predict violation types and provide actionable insights using Explainable AI. This project demonstrates MLOps best practices with research-grade analysis.

**Author:** VaishnaBala

---

## 🔬 Research Components

| # | Research Question | Methods |
|---|------------------|---------|
| 1 | How do ML models compare for spatio-temporal prediction? | XGBoost vs Random Forest, Statistical significance testing |
| 2 | How can we provide interpretable explanations? | SHAP vs LIME comparison, Spearman correlation |
| 3 | Can we detect unusual violation patterns? | Isolation Forest vs Autoencoder |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/TrafficViolationPrediction_Analytics.git
cd TrafficViolationPrediction_Analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python data_generator.py
python preprocessing.py
python research/baseline_model.py
python research/explainability.py
python research/anomaly_detection.py

# Start API
cd api
python main.py

🐳 Docker
bash
docker build -t traffic-api .
docker run -p 8000:8000 traffic-api
📁 Project Structure
text
TrafficViolationPrediction_Analytics/
├── data/
│   ├── raw/                 # Raw generated data
│   └── processed/           # Preprocessed features
├── research/
│   ├── baseline_model.py    # Model comparison
│   ├── explainability.py    # SHAP & LIME analysis
│   └── anomaly_detection.py # Anomaly detection
├── api/
│   └── main.py              # FastAPI application
├── models/                  # Saved models & encoders
├── results/
│   ├── figures/             # Visualizations
│   └── *.json               # Experiment results
├── data_generator.py
├── preprocessing.py
├── Dockerfile
├── requirements.txt
└── README.md
📊 API Endpoints
Endpoint	Method	Description
/	GET	API info
/health	GET	Health check
/predict	POST	Predict violation type
/junctions	GET	List supported junctions
/vehicles	GET	List supported vehicles
Example Request
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"junction_name": "Silk-board-junction", "vehicle_type": "Motorbike", "hour": 18, "day_of_week": 4}'
Example Response
json
{
  "predicted_violation": "RED_LIGHT_VIOLATION",
  "confidence": 0.7234,
  "risk_level": "HIGH",
  "probabilities": {
    "RED_LIGHT_VIOLATION": 0.72,
    "SPEED_VIOLATION": 0.21,
    "WRONG_WAY": 0.06
  }
}
📈 Results Summary
Model Comparison
Best Model: XGBoost
Test Accuracy: ~70%
Statistical Significance: Paired t-test, Wilcoxon
Explainability
Top Features: Junction risk, Hour, Peak hour flag
SHAP-LIME Correlation: Strong agreement (ρ > 0.7)
Anomaly Detection
Isolation Forest F1: ~0.78
Autoencoder F1: ~0.79
🛠️ Tech Stack
ML: XGBoost, Scikit-learn, TensorFlow
XAI: SHAP, LIME
API: FastAPI, Uvicorn
MLOps: MLflow, Docker
Visualization: Matplotlib, Seaborn
📚 References
Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
Ribeiro et al. (2016). "Why Should I Trust You?: Explaining the Predictions"
Liu et al. (2008). "Isolation Forest"
Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
📜 License
MIT License



 For New Users (Clone & Run)
Step 1: Clone Repository
bash
git clone https://github.com/VaishnaBala/TrafficViolationPrediction_Analytics.git
cd TrafficViolationPrediction_Analytics
Step 2: Create Virtual Environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Run Pipeline (In Order)
bash
# 1. Generate data
python data_generator.py

# 2. Preprocess data
python preprocessing.py

# 3. Train model
python research/baseline_model.py

# 4. Run explainability (optional)
python research/explainability.py

# 5. Run anomaly detection (optional)
python research/anomaly_detection.py
Step 5: Start API
bash
cd api
python main.py
Step 6: Open Browser
text
http://localhost:8000/docs



👤 Author
VaishnaBala

GitHub: @Vaishnabala
