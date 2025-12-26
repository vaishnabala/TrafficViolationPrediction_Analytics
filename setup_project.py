import os

# Folder structure
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/research",
    "api",
    "models",
    "results/figures",
    "results/tables"
]

# Empty files to create
files = [
    "src/__init__.py",
    "src/data_generator.py",
    "src/preprocessing.py",
    "src/train.py",
    "src/research/__init__.py",
    "src/research/baseline_model.py",
    "src/research/explainability.py",
    "src/research/anomaly_detection.py",
    "src/run_research.py",
    "api/__init__.py",
    "api/main.py",
    "notebooks/01_eda.ipynb",
    "notebooks/02_modeling.ipynb"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"📁 Created: {folder}")

# Create empty files
for file in files:
    open(file, 'a').close()
    print(f"📄 Created: {file}")

# Create requirements.txt
with open("requirements.txt", "w") as f:
    f.write("""pandas
numpy
scikit-learn
xgboost
shap
lime
fastapi
uvicorn
mlflow
folium
matplotlib
seaborn
tensorflow
scipy
""")
print("📄 Created: requirements.txt")

# Create .gitignore
with open(".gitignore", "w") as f:
    f.write("""venv/
__pycache__/
*.pkl
mlruns/
.env
*.pyc
.ipynb_checkpoints/
""")
print("📄 Created: .gitignore")

# Create empty README
with open("README.md", "w") as f:
    f.write("# Traffic Violation Prediction System\n\nTODO: Add project description")
print("📄 Created: README.md")

# Create Dockerfile template
with open("Dockerfile", "w") as f:
    f.write("""FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""")
print("📄 Created: Dockerfile")

print("\n✅ Project structure created!")
print("\n📋 Next: Write code in each file separately.")