"""
Complete pipeline: data fetch → preprocessing → training → evaluation
"""

from src.data_fetcher import WeatherDataFetcher
from src.preprocessing import WindDataPreprocessor
from pathlib import Path


def run_pipeline():
    """Execute complete pipeline from data to trained model"""
    
    print("\n" + "="*70)
    print("TSO WIND FORECASTING - COMPLETE PIPELINE")
    print("Short-Term Wind Forecasting for Grid Balancing Support in Germany")
    print("="*70)
    
    # Create directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Fetch data
    print("\n" + "="*70)
    print("STEP 1: FETCHING WEATHER DATA")
    print("="*70)
    print("Fetching historical reanalysis data for 4 German regions:")
    print("  - Schleswig-Holstein (high wind, coastal)")
    print("  - Lower Saxony (major onshore wind)")
    print("  - Brandenburg (eastern transmission zone)")
    print("  - Bavaria (southern contrast region)")
    
    fetcher = WeatherDataFetcher()
    data = fetcher.fetch_all_locations()
    fetcher.save_data(data, "data/raw_weather_data.csv")
    
    # Step 2: Preprocessing
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING & SEQUENCE GENERATION")
    print("="*70)
    print("Creating 72h → 24h sequences for LSTM training...")
    
    preprocessor = WindDataPreprocessor()
    df = preprocessor.load_data("data/raw_weather_data.csv")
    dataset = preprocessor.prepare_dataset(df)
    preprocessor.save_dataset(dataset, "data/processed_dataset.pkl")
    preprocessor.save_scaler("models/scaler.pkl")  # Save scaler for API inference
    
    # Step 3: Training (imported to avoid circular dependency)
    print("\n" + "="*70)
    print("STEP 3: TRAINING LSTM MODEL")
    print("="*70)
    
    from train import train_model
    train_model()
    
    # Step 4: Power curve analysis
    print("\n" + "="*70)
    print("STEP 4: WIND-TO-POWER ANALYSIS")
    print("="*70)
    
    from src.power_curve import demonstrate_uncertainty_amplification
    demonstrate_uncertainty_amplification()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\n✓ All components generated:")
    print("  1. Historical weather data (4 German regions)")
    print("  2. Preprocessed LSTM sequences (72h → 24h)")
    print("  3. Trained LSTM model")
    print("  4. TSO-focused evaluation metrics")
    print("  5. Wind-to-power uncertainty analysis")
    print("\nThis prototype demonstrates operational forecasting")
    print("capabilities relevant for transmission system operators.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_pipeline()
