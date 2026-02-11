"""
Simple FastAPI wrapper for wind forecasting model (no database)
Quick predictions for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import WindDataPreprocessor
from src.model import WindForecastLSTM

app = FastAPI(
    title="TenneT Wind Forecast API (Simple)",
    description="Quick wind forecasting API without database",
    version="1.0.0"
)

# Global model and preprocessor (loaded once at startup)
model = None
preprocessor = None


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor at startup"""
    global model, preprocessor
    
    print("Loading model and preprocessor...")
    
    try:
        # Load model
        model = WindForecastLSTM()
        model.load_model("models/lstm_wind_forecast.h5")
        
        # Load preprocessor with scaler
        preprocessor = WindDataPreprocessor()
        preprocessor.load_scaler("models/scaler.pkl")
        
        print("✓ Model and preprocessor loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool


class ForecastResponse(BaseModel):
    forecast_time: str
    location: str
    predictions: list
    statistics: dict


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "TenneT Wind Forecast API (Simple Version)",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "forecast": "/api/forecast",
            "quick_predict": "/api/quick-predict"
        },
        "note": "This is a simplified version without database support"
    }


@app.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "not ready",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }


@app.get("/api/forecast", response_model=ForecastResponse)
def generate_forecast():
    """
    Generate 24-hour wind forecast for Schleswig-Holstein
    
    Uses latest available weather data to predict next 24 hours
    """
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load config
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get first location from config
        location = config['locations'][0]
        
        # Fetch recent weather data (72h)
        import requests
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=72)
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d"),
            'hourly': [
                'wind_speed_10m',
                'wind_direction_10m',
                'pressure_msl',
                'temperature_2m',
                'relative_humidity_2m'
            ],
            'timezone': config['timezone']
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'wind_speed': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'pressure': data['hourly']['pressure_msl'],
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'location': location['name']
        })
        
        # Preprocess
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.add_temporal_features(df)
        df = preprocessor.add_wind_components(df)
        df = preprocessor.add_lag_features(df)
        
        # Select features
        feature_cols = [
            'wind_speed', 'wind_u', 'wind_v', 'pressure', 'temperature', 'humidity',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'wind_max_6h', 'wind_max_12h', 'pressure_gradient',
            'wind_std_24h', 'wind_accel_3h', 'pressure_tendency_3h'
        ]
        
        # Get last 72 hours
        lookback = config['model']['lookback_hours']
        features = df[feature_cols].values[-lookback:]
        
        # Scale and reshape
        features_scaled = preprocessor.scaler.transform(features)
        X = features_scaled.reshape(1, lookback, -1)
        
        # Predict
        predictions_scaled = model.predict(X)
        predictions = preprocessor.inverse_transform_wind_speed(predictions_scaled[0])
        
        # Create forecast times (starting from now)
        forecast_times = [datetime.now() + timedelta(hours=i) for i in range(1, 25)]
        
        # Build response
        forecast_list = []
        for i, (time, wind) in enumerate(zip(forecast_times, predictions)):
            forecast_list.append({
                "timestamp": time.isoformat(),
                "hour_ahead": i + 1,
                "wind_speed_ms": round(float(wind), 2),
                "wind_speed_kmh": round(float(wind * 3.6), 2)
            })
        
        stats = {
            "average_wind_ms": round(float(predictions.mean()), 2),
            "max_wind_ms": round(float(predictions.max()), 2),
            "min_wind_ms": round(float(predictions.min()), 2),
            "std_dev_ms": round(float(predictions.std()), 2)
        }
        
        return {
            "forecast_time": datetime.now().isoformat(),
            "location": location['name'],
            "predictions": forecast_list,
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@app.get("/api/quick-predict")
def quick_predict(wind_speed: float = 10.0):
    """
    Quick prediction demo (not using real data)
    
    Args:
        wind_speed: Current wind speed in m/s
    
    Returns:
        Simple demo prediction
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate simple demo forecast
    future_speeds = [wind_speed + np.random.randn() * 2 for _ in range(24)]
    
    return {
        "current_wind": wind_speed,
        "forecast_24h": [round(max(0, s), 2) for s in future_speeds],
        "note": "This is a demo endpoint. Use /api/forecast for real predictions."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
