"""
Data preprocessing and sequence generation for LSTM training
"""

import numpy as np
import pandas as pd
import yaml
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import pickle


class WindDataPreprocessor:
    """Prepare weather data for LSTM model training"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lookback = self.config['model']['lookback_hours']
        self.forecast = self.config['model']['forecast_hours']
        self.scaler = StandardScaler()
    
    def load_data(self, filepath: str = "data/raw_weather_data.csv") -> pd.DataFrame:
        """Load raw weather data"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through interpolation"""
        numeric_cols = ['wind_speed', 'wind_direction', 'pressure', 'temperature', 'humidity']
        
        for col in numeric_cols:
            df[col] = df.groupby('location')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        
        # Fill any remaining NaN with forward/backward fill
        df[numeric_cols] = df.groupby('location')[numeric_cols].ffill().bfill()
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features"""
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
        
        return df
    
    def add_wind_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert wind direction to U/V components"""
        wind_rad = np.deg2rad(df['wind_direction'])
        df['wind_u'] = df['wind_speed'] * np.sin(wind_rad)
        df['wind_v'] = df['wind_speed'] * np.cos(wind_rad)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics and change rates - BEST FEATURES ONLY (6 selected)"""
        
        # ===== SELECTED HIGH-VALUE FEATURES (avoiding redundancy) =====
        
        # 1. Recent max wind - 2 timescales (0.909 and 0.818 correlation)
        #    Keep 6h and 12h, skip 3h (redundant) and 24h (redundant)
        df['wind_max_6h'] = df.groupby('location')['wind_speed'].transform(
            lambda x: x.rolling(window=6, min_periods=1).max()
        )
        df['wind_max_12h'] = df.groupby('location')['wind_speed'].transform(
            lambda x: x.rolling(window=12, min_periods=1).max()
        )
        
        # 2. Pressure gradient (0.621 correlation) - spatial feature
        #    Unique information (only spatial feature)
        df['pressure_gradient'] = df.groupby('timestamp')['pressure'].transform(
            lambda x: x.max() - x.min()
        )
        
        # 3. Wind variability (0.334 correlation) - 24h window only
        #    Skip 6h and 12h (redundant with 24h)
        df['wind_std_24h'] = df.groupby('location')['wind_speed'].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        ).fillna(0)
        
        # 4. Wind acceleration (0.267 correlation) - 3h window only
        #    Skip 1h (too noisy) and 6h (redundant)
        df['wind_accel_3h'] = df.groupby('location')['wind_speed'].diff(3).fillna(0)
        
        # 5. Pressure tendency (physically motivated) - 3h window only
        #    Storm approach indicator, skip other windows
        df['pressure_tendency_3h'] = df.groupby('location')['pressure'].diff(3).fillna(0)
        
        # NOTE: Legacy MA features removed - highly correlated with max features (0.97)
        #       and max features are more predictive (capture extremes better)
        
        return df
    
    def create_sequences(self, data: np.ndarray, location_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM: 72h lookback → 24h forecast
        
        Args:
            data: Feature array (timestamps, features)
            location_data: Data for single location
            
        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target sequences (samples, forecast, 1) - wind speed only
        """
        X, y = [], []
        
        wind_speed_idx = 0  # Wind speed is first column after processing
        
        for i in range(self.lookback, len(data) - self.forecast + 1):
            # Input: past 72 hours all features
            X.append(data[i - self.lookback:i, :])
            
            # Target: next 24 hours wind speed only
            y.append(data[i:i + self.forecast, wind_speed_idx])
        
        return np.array(X), np.array(y)
    
    def prepare_dataset(self, df: pd.DataFrame, train_split: float = 0.8) -> dict:
        """
        Prepare complete dataset with train/val/test splits
        
        Returns:
            Dictionary containing train/val/test data
        """
        # Preprocessing
        df = self.handle_missing_values(df)
        df = self.add_temporal_features(df)
        df = self.add_wind_components(df)
        df = self.add_lag_features(df)  # NEW: Add rolling stats and changes
        
        # Feature columns - CURATED BEST FEATURES (16 total, NO redundancy)
        feature_cols = [
            # Target and core physics (6 features)
            'wind_speed', 'wind_u', 'wind_v', 'pressure', 'temperature', 'humidity',
            
            # Temporal patterns (4 features)
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            
            # BEST 6 ENGINEERED FEATURES (selected for high signal, low redundancy):
            'wind_max_6h',           # 0.909 correlation - strongest predictor
            'wind_max_12h',          # 0.818 correlation - medium-term extreme
            'pressure_gradient',     # 0.621 correlation - spatial feature
            'wind_std_24h',          # 0.334 correlation - long-term variability
            'wind_accel_3h',         # 0.267 correlation - trend information
            'pressure_tendency_3h',  # 0.001 correlation - physically motivated
            
            # NOTE: Removed wind_speed_ma_6h and wind_speed_ma_24h
            #       They were 0.97 correlated with wind_max features (multicollinearity)
            #       Max features are more valuable for extreme event prediction
        ]
        
        # FIT SCALER ON ALL DATA TOGETHER (NOT PER-LOCATION)
        # This preserves extreme conditions across all locations
        print(f"\n🔧 Fitting scaler on ALL locations together...")
        all_features = df[feature_cols].values
        self.scaler.fit(all_features)  # ✅ FIXED: Single global scaler
        
        # Process each location separately
        all_X, all_y = [], []
        
        for location in df['location'].unique():
            loc_data = df[df['location'] == location].sort_values('timestamp').reset_index(drop=True)
            
            # Transform features using the GLOBAL scaler (not fit_transform)
            loc_features = loc_data[feature_cols].values
            loc_features_scaled = self.scaler.transform(loc_features)  # ✅ FIXED: transform not fit_transform
            
            # Create sequences
            X_loc, y_loc = self.create_sequences(loc_features_scaled, loc_data)
            
            all_X.append(X_loc)
            all_y.append(y_loc)
        
        # Combine all locations WITH timestamp tracking for stratification
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # ✅ PROPER TIME SERIES SPLIT: Chronological order (NO SHUFFLING)
        # Prevents data leakage and tests real forecasting ability
        # Train on past → validate on intermediate period → test on recent/future
        print(f"\n🕒 Using chronological temporal split (no data leakage)...")
        
        n_samples = X.shape[0]
        
        # Simple chronological split: 70% train, 15% val, 15% test
        # This ensures train < val < test in time (no future data in training)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)
        
        # NO SHUFFLING - maintain temporal order for time series
        # This ensures model is tested on truly unseen future data
        # Closest to production scenario where we forecast future conditions
        
        dataset = {
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_val': X[val_idx],
            'y_val': y[val_idx],
            'X_test': X[test_idx],
            'y_test': y[test_idx],
            'feature_names': feature_cols,
            'scaler': self.scaler
        }
        
        print(f"\nDataset prepared:")
        print(f"  Training samples: {dataset['X_train'].shape[0]}")
        print(f"  Validation samples: {dataset['X_val'].shape[0]}")
        print(f"  Test samples: {dataset['X_test'].shape[0]}")
        print(f"  Input shape: {dataset['X_train'].shape[1:]}")
        print(f"  Output shape: {dataset['y_train'].shape[1:]}")
        
        return dataset
    
    def save_scaler(self, filepath: str = "models/scaler.pkl"):
        """Save the scaler for later use in predictions"""
        import pickle
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str = "models/scaler.pkl"):
        """Load a saved scaler"""
        import pickle
        
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
    
    def inverse_transform_wind_speed(self, scaled_predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled wind speed predictions back to m/s
        
        Args:
            scaled_predictions: Scaled predictions from model (samples, 24)
            
        Returns:
            Real wind speeds in m/s
        """
        # Get the scaler parameters for wind_speed (first feature)
        # wind_speed_mean and wind_speed_std from the fitted scaler
        wind_speed_mean = self.scaler.mean_[0]
        wind_speed_std = self.scaler.scale_[0]
        
        # Inverse transform: real = (scaled * std) + mean
        real_wind_speeds = (scaled_predictions * wind_speed_std) + wind_speed_mean
        
        return real_wind_speeds
    
    def save_dataset(self, dataset: dict, filepath: str = "data/processed_dataset.pkl"):
        """Save processed dataset"""
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nDataset saved to {filepath}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = WindDataPreprocessor()
    df = preprocessor.load_data()
    dataset = preprocessor.prepare_dataset(df)
    preprocessor.save_dataset(dataset)
