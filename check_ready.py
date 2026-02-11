"""
Pre-flight check for advanced model training
Verifies all dependencies and data are ready
"""

import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed"""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'keras': 'keras',
        'jax': 'jax',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'yaml': 'pyyaml'
    }
    
    optional = {
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }
    
    all_good = True
    
    # Check required
    print("\n✓ Required packages:")
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {module:15s} - installed")
        except ImportError:
            print(f"  ✗ {module:15s} - MISSING (install: pip install {package})")
            all_good = False
    
    # Check optional
    print("\n✓ Optional packages (for gradient boosting):")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {module:15s} - installed")
        except ImportError:
            print(f"  ⚠ {module:15s} - missing (will auto-install if needed)")
    
    return all_good


def check_data_files():
    """Check if required data files exist"""
    print("\n" + "="*70)
    print("CHECKING DATA FILES")
    print("="*70)
    
    required_files = [
        "data/processed_dataset.pkl",
    ]
    
    optional_files = [
        "models/lstm_wind_forecast.h5",
        "models/scaler.pkl",
        "config.yaml"
    ]
    
    all_good = True
    
    print("\n✓ Required data files:")
    for filepath in required_files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size / 1024 / 1024
            print(f"  ✓ {filepath:40s} ({size:.1f} MB)")
        else:
            print(f"  ✗ {filepath:40s} - MISSING")
            all_good = False
    
    print("\n✓ Optional files:")
    for filepath in optional_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath:40s}")
        else:
            print(f"  ⚠ {filepath:40s} - missing (not critical)")
    
    return all_good


def check_directories():
    """Check/create necessary directories"""
    print("\n" + "="*70)
    print("CHECKING DIRECTORIES")
    print("="*70)
    
    required_dirs = [
        "models",
        "models/ensemble",
        "results",
        "results/advanced_model"
    ]
    
    print("")
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"  ✓ {directory:40s} - exists")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory:40s} - created")
    
    return True


def check_system_resources():
    """Check available system resources"""
    print("\n" + "="*70)
    print("SYSTEM RESOURCES")
    print("="*70)
    
    import psutil
    
    # Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)
    
    print(f"\n✓ Memory:")
    print(f"  Total: {memory_gb:.1f} GB")
    print(f"  Available: {memory_available_gb:.1f} GB")
    
    if memory_available_gb < 4:
        print(f"  ⚠ Warning: Low memory. Recommend ≥4GB available")
        print(f"     Consider reducing batch_size or model size")
    else:
        print(f"  ✓ Sufficient memory for training")
    
    # CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    print(f"\n✓ CPU:")
    print(f"  Physical cores: {cpu_count}")
    print(f"  Logical cores: {cpu_count_logical}")
    
    # GPU (optional)
    try:
        import jax
        devices = jax.devices()
        if any('gpu' in str(d).lower() for d in devices):
            print(f"\n✓ GPU:")
            print(f"  Detected: {devices}")
            print(f"  ✓ Training will use GPU acceleration (3-5x faster)")
        else:
            print(f"\n✓ GPU:")
            print(f"  Not detected - will use CPU")
            print(f"  (This is fine, just slower training)")
    except:
        print(f"\n✓ GPU:")
        print(f"  JAX not configured for GPU check")
        print(f"  Will likely use CPU")
    
    return True


def estimate_training_time():
    """Estimate training time based on system"""
    print("\n" + "="*70)
    print("ESTIMATED TRAINING TIME")
    print("="*70)
    
    import psutil
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Check for GPU
    has_gpu = False
    try:
        import jax
        devices = jax.devices()
        has_gpu = any('gpu' in str(d).lower() for d in devices)
    except:
        pass
    
    print("\n✓ Training time estimates:")
    
    if has_gpu:
        print("  Transformer: ~10-15 minutes (with GPU)")
        print("  XGBoost: ~1 minute")
        print("  LightGBM: ~1 minute")
        print("  Ensemble optimization: < 1 minute")
        print("  ─────────────────────────────────")
        print("  Total: ~15-20 minutes")
    else:
        print("  Transformer: ~20-30 minutes (CPU only)")
        print("  XGBoost: ~2 minutes")
        print("  LightGBM: ~2 minutes")
        print("  Ensemble optimization: < 1 minute")
        print("  ─────────────────────────────────")
        print("  Total: ~25-35 minutes")
    
    print("\n💡 Tips to reduce training time:")
    print("  - Reduce epochs: transformer.train(..., epochs=30)")
    print("  - Smaller model: d_model=64, n_layers=2")
    print("  - Skip Transformer: Use XGBoost+LightGBM+LSTM only")


def main():
    """Run all checks"""
    print("="*70)
    print("PRE-FLIGHT CHECK FOR ADVANCED MODEL TRAINING")
    print("="*70)
    
    # Run checks
    deps_ok = check_dependencies()
    data_ok = check_data_files()
    dirs_ok = check_directories()
    
    # System info
    try:
        import psutil
        resources_ok = check_system_resources()
        estimate_training_time()
    except ImportError:
        print("\n⚠ psutil not installed (pip install psutil)")
        print("  Skipping system resource check...")
        resources_ok = True
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if deps_ok and data_ok:
        print("\n✅ ALL CHECKS PASSED - Ready to train!")
        print("\nRun:")
        print("  python train_advanced_model.py")
        print("\nOr for quick demo:")
        print("  python demo_features.py")
    else:
        print("\n❌ SOME CHECKS FAILED")
        
        if not deps_ok:
            print("\n⚠ Missing dependencies:")
            print("  Install with: pip install numpy pandas keras jax scikit-learn scipy pyyaml")
        
        if not data_ok:
            print("\n⚠ Missing data files:")
            print("  Run preprocessing first: python run_pipeline.py")
    
    print("\n" + "="*70)
    
    return deps_ok and data_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
