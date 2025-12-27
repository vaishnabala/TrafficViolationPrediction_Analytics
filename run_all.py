import os
import subprocess

scripts = [
    "data_generator.py",
    "preprocessing.py",
    "research/baseline_model.py",
    "research/explainability.py",
    "research/anomaly_detection.py"
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"Running: {script}")
    print('='*50)
    subprocess.run(["python", script])

print("\n✅ All scripts completed!")
print("Run API: cd api && python main.py")