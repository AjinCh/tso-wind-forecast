"""
Advanced feature engineering for wind forecasting
Includes: Weather derivatives, temporal patterns, cross-region interactions
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


class AdvancedFeatureEngineer:
    """
    Enhanced feature engineering for wind forecasting
    
    Features created:
    1. Weather derivatives (rate of change, acceleration)
    2. Statistical features (rolling mean, std, quantiles)
    3. Temporal patterns (hour interactions, seasonality)
    4. Extreme event indicators
    5. Cross-region features (when multi-location data available)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def _weather_derivatives(self, df, column, windows=[1, 3, 6]):
        """
        Calculate rate of change and acceleration for weather variables
        Critical for detecting frontal passages and rapid wind changes
        
        Args:
            df: DataFrame with weather data
            column: Column name to compute derivatives
            windows: Time windows (hours) for derivatives
        
        Returns:
            DataFrame with derivative features
        """
        features = pd.DataFrame(index=df.index)
        
        for window in windows:
            # First derivative (rate of change)
            features[f'{column}_change_{window}h'] = df[column].diff(window)
            
            # Second derivative (acceleration)
            features[f'{column}_accel_{window}h'] = features[f'{column}_change_{window}h'].diff(window)
            
            # Rate of change percentage
            features[f'{column}_pct_change_{window}h'] = df[column].pct_change(window)
        
        return features
    
    def _rolling_statistics(self, df, column, windows=[6, 12, 24]):
        """
        Calculate rolling statistics to capture recent trends
        
        Args:
            df: DataFrame with weather data
            column: Column name
            windows: Rolling window sizes (hours)
        
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=df.index)
        
        for window in windows:
            # Rolling mean (trend)
            features[f'{column}_mean_{window}h'] = df[column].rolling(window=window, min_periods=1).mean()
            
            # Rolling std (variability)
            features[f'{column}_std_{window}h'] = df[column].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max (range)
            features[f'{column}_min_{window}h'] = df[column].rolling(window=window, min_periods=1).min()
            features[f'{column}_max_{window}h'] = df[column].rolling(window=window, min_periods=1).max()
            features[f'{column}_range_{window}h'] = features[f'{column}_max_{window}h'] - features[f'{column}_min_{window}h']
            
            # Rolling quantiles (distribution shape)
            features[f'{column}_q25_{window}h'] = df[column].rolling(window=window, min_periods=1).quantile(0.25)
            features[f'{column}_q75_{window}h'] = df[column].rolling(window=window, min_periods=1).quantile(0.75)
            features[f'{column}_iqr_{window}h'] = features[f'{column}_q75_{window}h'] - features[f'{column}_q25_{window}h']
        
        return features
    
    def _temporal_interactions(self, df, hour_col='hour_sin', month_col='month_sin'):
        """
        Create interaction features between weather and temporal patterns
        E.g., "low pressure in winter" behaves differently than "low pressure in summer"
        
        Args:
            df: DataFrame with weather and temporal data
        
        Returns:
            DataFrame with interaction features
        """
        features = pd.DataFrame(index=df.index)
        
        if 'wind_speed_10m' in df.columns and hour_col in df.columns:
            # Wind-hour interaction (diurnal patterns differ by wind regime)
            features['wind_hour_interaction'] = df['wind_speed_10m'] * df[hour_col]
            
        if 'pressure_msl' in df.columns and month_col in df.columns:
            # Pressure-season interaction (seasonal pressure patterns)
            features['pressure_season_interaction'] = df['pressure_msl'] * df[month_col]
        
        if 'temperature_2m' in df.columns and hour_col in df.columns:
            # Temperature-hour interaction (thermal effects)
            features['temp_hour_interaction'] = df['temperature_2m'] * df[hour_col]
        
        return features
    
    def _extreme_indicators(self, df):
        """
        Binary/categorical indicators for extreme weather conditions
        Critical for TSO operations
        
        Args:
            df: DataFrame with weather data
        
        Returns:
            DataFrame with extreme event indicators
        """
        features = pd.DataFrame(index=df.index)
        
        if 'wind_speed_10m' in df.columns:
            # High wind event (>12 m/s = critical for grid)
            features['is_high_wind'] = (df['wind_speed_10m'] > 12).astype(float)
            
            # Very high wind (>15 m/s = potential curtailment)
            features['is_very_high_wind'] = (df['wind_speed_10m'] > 15).astype(float)
            
            # Low wind (< 3 m/s = cut-in speed)
            features['is_low_wind'] = (df['wind_speed_10m'] < 3).astype(float)
        
        if 'pressure_msl' in df.columns:
            # Low pressure system (potential rapid changes)
            pressure_threshold = df['pressure_msl'].quantile(0.25)
            features['is_low_pressure'] = (df['pressure_msl'] < pressure_threshold).astype(float)
        
        # Rapid wind change indicator (frontal passage)
        if 'wind_speed_10m' in df.columns:
            wind_change_1h = df['wind_speed_10m'].diff(1).abs()
            features['is_rapid_change'] = (wind_change_1h > 3).astype(float)
        
        return features
    
    def _wind_regime_classification(self, df):
        """
        Categorize current wind regime
        Different regimes may have different predictability
        
        Args:
            df: DataFrame with wind speed data
        
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df.index)
        
        if 'wind_speed_10m' in df.columns:
            wind = df['wind_speed_10m']
            
            # Regime classification (one-hot encoded)
            features['regime_calm'] = (wind < 3).astype(float)
            features['regime_light'] = ((wind >= 3) & (wind < 6)).astype(float)
            features['regime_moderate'] = ((wind >= 6) & (wind < 12)).astype(float)
            features['regime_strong'] = ((wind >= 12) & (wind < 18)).astype(float)
            features['regime_gale'] = (wind >= 18).astype(float)
        
        return features
    
    def _persistence_features(self, df, column, lags=[1, 2, 3, 6, 12, 24]):
        """
        Lag features - simple but powerful for time series
        Persistence is a strong baseline in weather forecasting
        
        Args:
            df: DataFrame with weather data
            column: Column to create lags for
            lags: List of lag hours
        
        Returns:
            DataFrame with lag features
        """
        features = pd.DataFrame(index=df.index)
        
        for lag in lags:
            features[f'{column}_lag_{lag}h'] = df[column].shift(lag)
        
        return features
    
    def _fourier_features(self, df, period_hours=[24, 168, 8760], n_terms=3):
        """
        Fourier terms to capture periodic patterns
        - 24h: Diurnal cycle
        - 168h: Weekly cycle
        - 8760h: Annual cycle
        
        Args:
            df: DataFrame with time index
            period_hours: Periods to model
            n_terms: Number of Fourier terms per period
        
        Returns:
            DataFrame with Fourier features
        """
        features = pd.DataFrame(index=df.index)
        
        # Get hour of year for each timestamp
        if isinstance(df.index, pd.DatetimeIndex):
            hour_of_year = df.index.dayofyear * 24 + df.index.hour
        else:
            # Assume we have hour_of_year column or can reconstruct
            return features
        
        for period in period_hours:
            for k in range(1, n_terms + 1):
                features[f'fourier_sin_{period}h_term{k}'] = np.sin(2 * np.pi * k * hour_of_year / period)
                features[f'fourier_cos_{period}h_term{k}'] = np.cos(2 * np.pi * k * hour_of_year / period)
        
        return features
    
    def _atmospheric_stability(self, df):
        """
        Estimate atmospheric stability indicators
        Affects wind profile and turbulence
        
        Args:
            df: DataFrame with temperature and wind data
        
        Returns:
            DataFrame with stability features
        """
        features = pd.DataFrame(index=df.index)
        
        # Temperature gradient (if we had multiple levels)
        # For now, use temperature change as proxy for stability
        if 'temperature_2m' in df.columns:
            temp_change = df['temperature_2m'].diff(1)
            features['temp_gradient_1h'] = temp_change
            
            # Diurnal temperature range (last 24h)
            features['temp_range_24h'] = (
                df['temperature_2m'].rolling(24, min_periods=1).max() - 
                df['temperature_2m'].rolling(24, min_periods=1).min()
            )
        
        # Richardson number approximation (stability parameter)
        # Ri ≈ (g/T) * (dT/dz) / (dU/dz)²
        # Simplified version using available data
        if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
            temp_grad = df['temperature_2m'].diff(1)
            wind_grad = df['wind_speed_10m'].diff(1)
            # Avoid division by zero
            wind_grad_safe = wind_grad.replace(0, 0.01)
            features['richardson_approx'] = temp_grad / (wind_grad_safe ** 2)
            features['richardson_approx'] = features['richardson_approx'].clip(-10, 10)
        
        return features
    
    def create_all_features(self, df, weather_columns=['wind_speed_10m', 'pressure_msl', 'temperature_2m']):
        """
        Create full suite of advanced features
        
        Args:
            df: DataFrame with basic weather data
            weather_columns: List of weather variables to enhance
        
        Returns:
            DataFrame with all engineered features
        """
        print("\nEngineering advanced features...")
        all_features = [df.copy()]
        
        for col in weather_columns:
            if col not in df.columns:
                continue
            
            print(f"  Processing {col}...")
            
            # Weather derivatives
            all_features.append(self._weather_derivatives(df, col))
            
            # Rolling statistics
            all_features.append(self._rolling_statistics(df, col))
            
            # Persistence (lags)
            all_features.append(self._persistence_features(df, col))
        
        # Temporal interactions
        print("  Creating temporal interactions...")
        all_features.append(self._temporal_interactions(df))
        
        # Extreme indicators
        print("  Creating extreme event indicators...")
        all_features.append(self._extreme_indicators(df))
        
        # Wind regime classification
        print("  Creating wind regime features...")
        all_features.append(self._wind_regime_classification(df))
        
        # Fourier features
        print("  Creating Fourier features...")
        all_features.append(self._fourier_features(df))
        
        # Atmospheric stability
        print("  Creating stability features...")
        all_features.append(self._atmospheric_stability(df))
        
        # Combine all features
        result = pd.concat(all_features, axis=1)
        
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        # Handle any remaining NaNs (from rolling windows, diffs, etc.)
        # Use forward fill then backward fill
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaNs, fill with 0
        result = result.fillna(0)
        
        self.feature_names = result.columns.tolist()
        
        print(f"\n✓ Created {len(self.feature_names)} features")
        print(f"  Original: {len(df.columns)}")
        print(f"  Engineered: {len(self.feature_names) - len(df.columns)}")
        
        return result
    
    def get_feature_importance_groups(self):
        """
        Return feature groups for interpretability analysis
        """
        groups = {
            'derivatives': [f for f in self.feature_names if '_change_' in f or '_accel_' in f],
            'rolling_stats': [f for f in self.feature_names if '_mean_' in f or '_std_' in f or '_range_' in f],
            'lags': [f for f in self.feature_names if '_lag_' in f],
            'extremes': [f for f in self.feature_names if 'is_' in f or 'regime_' in f],
            'temporal': [f for f in self.feature_names if 'hour' in f or 'month' in f or 'fourier' in f],
            'stability': [f for f in self.feature_names if 'richardson' in f or 'temp_gradient' in f],
        }
        return groups


def demonstrate_feature_engineering():
    """
    Demonstrate feature engineering on sample data
    """
    # Create sample data
    np.random.seed(42)
    n_hours = 1000
    
    dates = pd.date_range('2024-01-01', periods=n_hours, freq='h')
    
    df = pd.DataFrame({
        'wind_speed_10m': 8 + 4 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.randn(n_hours) * 2,
        'pressure_msl': 1013 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 168) + np.random.randn(n_hours) * 3,
        'temperature_2m': 15 + 10 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) + np.random.randn(n_hours) * 2,
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'month_sin': np.sin(2 * np.pi * dates.month / 12),
        'month_cos': np.cos(2 * np.pi * dates.month / 12),
    }, index=dates)
    
    # Engineer features
    engineer = AdvancedFeatureEngineer()
    enhanced_df = engineer.create_all_features(df)
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("="*70)
    print(f"Input features: {df.shape[1]}")
    print(f"Output features: {enhanced_df.shape[1]}")
    print(f"Enhancement ratio: {enhanced_df.shape[1] / df.shape[1]:.1f}x")
    print("\nSample feature groups:")
    groups = engineer.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")
    print("="*70)
    
    return enhanced_df, engineer


if __name__ == "__main__":
    demonstrate_feature_engineering()
