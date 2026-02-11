"""
Real-Time Wind Forecast for Tomorrow
Fetches current conditions and predicts next 24 hours
"""

import numpy as np
import pandas as pd
import requests
import yaml
from datetime import datetime, timedelta
from src.model import WindForecastLSTM
from src.preprocessing import WindDataPreprocessor


class TomorrowForecast:
    """
    Generate real-time wind speed forecast for next 24 hours
    Uses current/recent observations to predict tomorrow's wind
    """
    
    def __init__(self):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = WindForecastLSTM()
        self.preprocessor = WindDataPreprocessor()
        
        # Load trained model
        print("Loading trained model...")
        self.model.load_model("models/lstm_wind_forecast.h5")
        
        # Load scaler
        print("Loading scaler...")
        self.preprocessor.load_scaler("models/scaler.pkl")
        
        print("✓ Ready to forecast!\n")
    
    def fetch_current_conditions(self, location, lookback_hours=72):
        """
        Fetch recent weather data up to current time
        
        Args:
            location: Dict with 'name', 'lat', 'lon'
            lookback_hours: How many hours of history to fetch (default: 72)
        
        Returns:
            DataFrame with recent weather observations
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        # Format for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        print(f"Fetching current conditions for {location['name']}...")
        print(f"  Range: {start_str} to {end_str}")
        
        # Open-Meteo API call
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'start_date': start_str,
            'end_date': end_str,
            'hourly': [
                'wind_speed_10m',
                'wind_direction_10m',
                'pressure_msl',
                'temperature_2m',
                'relative_humidity_2m'
            ],
            'timezone': self.config['timezone']
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'wind_speed': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'pressure': data['hourly']['pressure_msl'],
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'location': location['name']
        })
        
        print(f"  ✓ Fetched {len(df)} hourly observations")
        print(f"  Latest: {df['timestamp'].iloc[-1]}")
        
        return df
    
    def preprocess_for_prediction(self, df):
        """
        Prepare recent data for model input
        Same preprocessing as training
        """
        print("\nPreprocessing data...")
        
        # Apply same transformations as training
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.add_temporal_features(df)
        df = self.preprocessor.add_wind_components(df)
        df = self.preprocessor.add_lag_features(df)
        
        # Select features (same order as training)
        feature_cols = [
            'wind_speed', 'wind_u', 'wind_v', 'pressure', 'temperature', 'humidity',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'wind_max_6h', 'wind_max_12h', 'pressure_gradient',
            'wind_std_24h', 'wind_accel_3h', 'pressure_tendency_3h'
        ]
        
        # Get last 72 hours for prediction
        lookback = self.config['model']['lookback_hours']
        features = df[feature_cols].values[-lookback:]
        
        # Scale features
        features_scaled = self.preprocessor.scaler.transform(features)
        
        # Reshape for model: (1, 72, 16)
        X = features_scaled.reshape(1, lookback, -1)
        
        print(f"  ✓ Input shape: {X.shape}")
        
        return X
    
    def predict_tomorrow(self, location_name=None):
        """
        Generate 24-hour wind forecast starting from now
        
        Args:
            location_name: Specific location to forecast (or None for first location)
        
        Returns:
            DataFrame with forecast
        """
        print("="*70)
        print("REAL-TIME WIND FORECAST FOR TOMORROW")
        print("="*70)
        print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Select location
        if location_name:
            location = next(
                (loc for loc in self.config['locations'] if loc['name'] == location_name),
                self.config['locations'][0]
            )
        else:
            location = self.config['locations'][0]
        
        print(f"Location: {location['name']}")
        
        # Fetch current conditions
        df = self.fetch_current_conditions(location)
        
        # Check data completeness
        latest_obs = df['timestamp'].iloc[-1]
        hours_old = (datetime.now(latest_obs.tz) - latest_obs).total_seconds() / 3600
        
        if hours_old > 3:
            print(f"\n⚠️  WARNING: Latest data is {hours_old:.1f} hours old")
            print(f"   Forecast may be less accurate with stale data")
        
        # Preprocess
        X = self.preprocess_for_prediction(df)
        
        # Predict
        print("\nGenerating forecast...")
        predictions_scaled = self.model.predict(X)[0]  # Shape: (24,) in scaled units
        
        # ✅ CRITICAL: Inverse transform predictions back to real m/s
        predictions = self.preprocessor.inverse_transform_wind_speed(predictions_scaled.reshape(1, -1))[0]
        
        print(f"  ✓ Predicted wind range: {predictions.min():.2f} - {predictions.max():.2f} m/s")
        
        # Create forecast timestamps
        start_time = df['timestamp'].iloc[-1] + timedelta(hours=1)
        forecast_times = [start_time + timedelta(hours=h) for h in range(24)]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': forecast_times,
            'predicted_wind_speed_ms': predictions,
            'predicted_wind_speed_kmh': predictions * 3.6,  # Convert to km/h
            'hour_ahead': range(1, 25)
        })
        
        # Add descriptive categories
        def wind_category(speed):
            if speed < 1: return "Calm"
            elif speed < 5: return "Light breeze"
            elif speed < 10: return "Moderate"
            elif speed < 15: return "Fresh"
            elif speed < 20: return "Strong"
            else: return "Gale"
        
        forecast_df['category'] = forecast_df['predicted_wind_speed_ms'].apply(wind_category)
        
        print("✓ Forecast complete!")
        
        return forecast_df, location
    
    def display_forecast(self, forecast_df, location):
        """
        Display forecast in user-friendly format
        """
        print("\n" + "="*70)
        print(f"24-HOUR WIND FORECAST: {location['name']}")
        print("="*70)
        
        # Current conditions (last observation)
        print(f"\nForecast generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Coordinates: {location['lat']}°N, {location['lon']}°E")
        
        # Summary statistics
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        print(f"  Average wind:  {forecast_df['predicted_wind_speed_ms'].mean():.1f} m/s ({forecast_df['predicted_wind_speed_kmh'].mean():.1f} km/h)")
        print(f"  Minimum wind:  {forecast_df['predicted_wind_speed_ms'].min():.1f} m/s ({forecast_df['predicted_wind_speed_kmh'].min():.1f} km/h)")
        print(f"  Maximum wind:  {forecast_df['predicted_wind_speed_ms'].max():.1f} m/s ({forecast_df['predicted_wind_speed_kmh'].max():.1f} km/h)")
        
        # Key hours (every 3 hours)
        print("\n" + "-"*70)
        print("HOURLY FORECAST")
        print("-"*70)
        print(f"{'Time':<20} {'Wind (m/s)':<12} {'Wind (km/h)':<12} {'Category':<15}")
        print("-"*70)
        
        for idx in range(0, 24, 3):  # Every 3 hours
            row = forecast_df.iloc[idx]
            time_str = row['timestamp'].strftime('%a %b %d, %H:%M')
            print(f"{time_str:<20} {row['predicted_wind_speed_ms']:>6.1f}      "
                  f"{row['predicted_wind_speed_kmh']:>6.1f}      {row['category']:<15}")
        
        # Tomorrow specific forecast
        tomorrow_start = forecast_df[forecast_df['timestamp'].dt.date == 
                                     (datetime.now() + timedelta(days=1)).date()]
        
        if len(tomorrow_start) > 0:
            print("\n" + "-"*70)
            print(f"TOMORROW ({(datetime.now() + timedelta(days=1)).strftime('%A, %B %d')})")
            print("-"*70)
            print(f"  Average: {tomorrow_start['predicted_wind_speed_ms'].mean():.1f} m/s")
            print(f"  Peak:    {tomorrow_start['predicted_wind_speed_ms'].max():.1f} m/s at "
                  f"{tomorrow_start.loc[tomorrow_start['predicted_wind_speed_ms'].idxmax(), 'timestamp'].strftime('%H:%M')}")
            print(f"  Low:     {tomorrow_start['predicted_wind_speed_ms'].min():.1f} m/s at "
                  f"{tomorrow_start.loc[tomorrow_start['predicted_wind_speed_ms'].idxmin(), 'timestamp'].strftime('%H:%M')}")
        
        # Operational guidance for TSO
        print("\n" + "-"*70)
        print("TSO OPERATIONAL GUIDANCE")
        print("-"*70)
        avg_wind = forecast_df['predicted_wind_speed_ms'].mean()
        max_wind = forecast_df['predicted_wind_speed_ms'].max()
        
        if max_wind > 12:
            print("  ⚠️  HIGH WIND ALERT: Expect elevated power output")
            print("     Action: Pre-position reserves for high generation")
        elif max_wind > 10:
            print("  ✓ MODERATE-HIGH: Good wind generation expected")
            print("     Action: Standard reserve protocols")
        elif avg_wind < 4:
            print("  ⚠️  LOW WIND PERIOD: Reduced wind generation")
            print("     Action: Ensure backup capacity available")
        else:
            print("  ✓ NORMAL CONDITIONS: Standard operations")
        
        # Forecast uncertainty reminder
        print("\n" + "-"*70)
        print("UNCERTAINTY ESTIMATE (from validation)")
        print("-"*70)
        print("  Typical error (MAE):     ±0.62 m/s")
        print("  95% confidence interval: ±1.60 m/s")
        print("  Worst case observed:     ±5.16 m/s")
        print("\n  Note: Errors increase for longer forecast horizons")
        print("        Hour 1-6:  Most reliable (±0.3-0.5 m/s)")
        print("        Hour 12:   Good reliability (±0.6 m/s)")
        print("        Hour 24:   Moderate reliability (±0.7 m/s)")
        
        print("\n" + "="*70)
        
        # Save forecast
        output_file = f"results/forecast_{location['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        forecast_df.to_csv(output_file, index=False)
        print(f"\n✓ Forecast saved to: {output_file}")
        
        return forecast_df


def main():
    """
    Run tomorrow's forecast
    """
    forecaster = TomorrowForecast()
    
    # Generate forecast
    forecast_df, location = forecaster.predict_tomorrow()
    
    # Display
    forecaster.display_forecast(forecast_df, location)
    
    print("\n💡 TIP: Run this script hourly to get updated rolling forecasts!")
    print("    Or integrate into your n8n workflow for automated updates")


if __name__ == "__main__":
    main()
