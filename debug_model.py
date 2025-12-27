import os

project_root = r"D:\GitHub_Works\TrafficViolationPrediction_Analytics"
models_dir = os.path.join(project_root, "models")

print(f"📁 Checking: {models_dir}\n")

# Check if folder exists
if os.path.exists(models_dir):
    print("✅ Models folder exists\n")
    
    # List all files
    files = os.listdir(models_dir)
    
    if files:
        print("📄 Files found:")
        for f in files:
            full_path = os.path.join(models_dir, f)
            size = os.path.getsize(full_path)
            print(f"   - {f} ({size} bytes)")
    else:
        print("❌ Folder is empty!")
else:
    print("❌ Models folder does NOT exist!")

# Check specific file
print("\n" + "=" * 50)
expected_file = os.path.join(models_dir, "best_baseline_model.pkl")
print(f"\n🔍 Looking for: {expected_file}")
print(f"   Exists: {os.path.exists(expected_file)}")