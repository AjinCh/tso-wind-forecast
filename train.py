"""
Main training script for wind forecasting LSTM
"""

import pickle
import numpy as np
from pathlib import Path

from src.model import WindForecastLSTM
from src.evaluation import TSOEvaluator


def train_model():
    """Complete training pipeline"""
    
    print("="*60)
    print("TSO WIND FORECASTING - LSTM TRAINING")
    print("="*60)
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Load preprocessed dataset
    print("\n1. Loading preprocessed dataset...")
    with open("data/processed_dataset.pkl", 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"   Training samples: {dataset['X_train'].shape[0]}")
    print(f"   Validation samples: {dataset['X_val'].shape[0]}")
    print(f"   Test samples: {dataset['X_test'].shape[0]}")
    
    # Initialize model
    print("\n2. Initializing LSTM model...")
    model = WindForecastLSTM()
    
    # Train model
    print("\n3. Training model...")
    model.train(
        dataset['X_train'], dataset['y_train'],
        dataset['X_val'], dataset['y_val']
    )
    
    # Save model
    print("\n4. Saving trained model...")
    model.save_model("models/lstm_wind_forecast.h5")
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    y_pred_scaled = model.predict(dataset['X_test'])
    y_true_scaled = dataset['y_test']
    
    # ✅ FIX: Inverse transform to real m/s BEFORE evaluation
    from src.preprocessing import WindDataPreprocessor
    preprocessor = WindDataPreprocessor()
    preprocessor.scaler = dataset['scaler']  # Use the same scaler
    
    y_pred = preprocessor.inverse_transform_wind_speed(y_pred_scaled)
    y_true = preprocessor.inverse_transform_wind_speed(y_true_scaled)
    
    print(f"\n   Predictions range: {y_pred.min():.2f} - {y_pred.max():.2f} m/s")
    print(f"   Ground truth range: {y_true.min():.2f} - {y_true.max():.2f} m/s")
    
    evaluator = TSOEvaluator()
    results = evaluator.full_evaluation(y_true, y_pred)
    
    # Create evaluation plots
    evaluator.plot_results(y_true, y_pred, "results/evaluation_plots.png")
    
    # Save results
    with open("results/evaluation_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save predictions for analysis (in real m/s units)
    np.save("results/predictions.npy", y_pred)
    np.save("results/ground_truth.npy", y_true)
    
    # Also save scaled versions for debugging
    np.save("results/predictions_scaled.npy", y_pred_scaled)
    np.save("results/ground_truth_scaled.npy", y_true_scaled)
    
    # Save sample predictions as CSV for easy viewing
    import pandas as pd
    sample_size = min(100, len(y_pred))
    pred_df = pd.DataFrame({
        'sample': range(sample_size),
        'predicted_wind_speed_avg': y_pred[:sample_size].mean(axis=1),
        'actual_wind_speed_avg': y_true[:sample_size].mean(axis=1)
    })
    pred_df.to_csv("results/sample_predictions.csv", index=False)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - models/lstm_wind_forecast.h5")
    print("  - results/evaluation_plots.png")
    print("  - results/evaluation_results.pkl")
    print("  - results/predictions.npy (full predictions)")
    print("  - results/ground_truth.npy (actual values)")
    print("  - results/sample_predictions.csv (first 100 samples)")
    print("\n✓ Model ready for TSO grid balancing support")
    

if __name__ == "__main__":
    train_model()
