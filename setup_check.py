"""
Quick Start Guide and Production Deployment Instructions.

This script helps you get the Neuro-Econometric Market Alpha Engine running.
"""

import subprocess
import sys
import time
from pathlib import Path
import json

print("="*70)
print(" NEURO-ECONOMETRIC MARKET ALPHA ENGINE - PRODUCTION SETUP")
print("="*70)
print()

def check_file_exists(filepath):
    """Check if required file exists."""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")
    return exists

def check_dependencies():
    """Check if all dependencies are installed."""
    print("[1/5] Checking dependencies...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'statsmodels', 
        'transformers', 'fastapi', 'uvicorn', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def check_data():
    """Check if data files exist."""
    print("\n[2/5] Checking data files...")
    
    data_file = "data/GSPC_ohlcv.csv"
    exists = check_file_exists(data_file)
    
    if not exists:
        print("\n  Generate sample data:")
        print("  python generate_sample_data.py")
        return False
    
    return True

def check_model():
    """Check if trained model exists."""
    print("\n[3/5] Checking trained model...")
    
    model_files = [
        "models_saved/best_model.pt",
        "models_saved/training_info.json"
    ]
    
    all_exist = all(check_file_exists(f) for f in model_files)
    
    if not all_exist:
        print("\n  Train the model:")
        print("  python train_model.py")
        return False
    
    # Load and display metrics
    try:
        with open("models_saved/training_info.json", 'r') as f:
            info = json.load(f)
        
        print(f"\n  Model Info:")
        print(f"    Trained: {info['timestamp']}")
        print(f"    Epochs: {info['total_epochs']}")
        print(f"    Directional Accuracy: {info['metrics']['directional_accuracy']:.2f}%")
    except:
        pass
    
    return True

def show_usage():
    """Show usage instructions."""
    print("\n[4/5] Available scripts...")
    
    scripts = {
        "generate_sample_data.py": "Generate synthetic S&P 500 data",
        "train_model.py": "Train the hybrid model (~40 min)",
        "serve_api.py": "Start FastAPI inference server",
        "evaluate_model.py": "Run model evaluation",
        "diagnostic.py": "Run system diagnostics"
    }
    
    for script, description in scripts.items():
        exists = Path(script).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {script} - {description}")

def show_api_examples():
    """Show API usage examples."""
    print("\n[5/5] API Usage Examples...")
    print("\n  Start API server:")
    print("    python serve_api.py")
    print("\n  Test endpoints (in new terminal):")
    print("    curl http://localhost:8000/")
    print("    curl http://localhost:8000/health")
    print("    curl http://localhost:8000/model/info")
    print('\n    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\\"ticker\\": \\"^GSPC\\", \\"horizon\\": 1}"')
    
    print("\n  Or visit interactive docs:")
    print("    http://localhost:8000/docs")

def main():
    """Main setup check."""
    
    # Run all checks
    deps_ok = check_dependencies()
    data_ok = check_data()
    model_ok = check_model()
    
    show_usage()
    show_api_examples()
    
    # Summary
    print("\n" + "="*70)
    print(" SYSTEM STATUS")
    print("="*70)
    
    status = []
    status.append(("Dependencies", "✓ Ready" if deps_ok else "✗ Install packages"))
    status.append(("Data", "✓ Ready" if data_ok else "✗ Generate data"))
    status.append(("Model", "✓ Ready" if model_ok else "✗ Train model"))
    
    for component, state in status:
        print(f"  {component:15s}: {state}")
    
    if deps_ok and data_ok and model_ok:
        print("\n✓ System ready for production!")
        print("\nQuick start:")
        print("  1. python serve_api.py          # Start API server")
        print("  2. Visit http://localhost:8000/docs  # Interactive API documentation")
        print("  3. Make predictions via API")
    else:
        print("\n⚠ Complete setup steps above before running")
    
    print()

if __name__ == "__main__":
    main()
