"""
Data fetching module for historical weather data
Uses Open-Meteo API for ERA5-based reanalysis data
"""

import requests
import pandas as pd
import yaml
from typing import Dict, List
from datetime import datetime


class WeatherDataFetcher:
    """Fetch historical weather data from Open-Meteo API"""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def fetch_location_data(self, location: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data for a single location
        
        Args:
            location: Dict with 'name', 'lat', 'lon'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly weather data
        """
        params = {
            'latitude': location['lat'],
            'longitude': location['lon'],
            'start_date': start_date,
            'end_date': end_date,
            'hourly': [
                'wind_speed_10m',
                'wind_direction_10m',
                'pressure_msl',
                'temperature_2m',
                'relative_humidity_2m'
            ],
            'timezone': self.config['timezone']
        }
        
        print(f"Fetching data for {location['name']}...")
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'wind_speed': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'pressure': data['hourly']['pressure_msl'],
            'temperature': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'location': location['name']
        })
        
        return df
    
    def fetch_all_locations(self) -> pd.DataFrame:
        """
        Fetch data for all configured locations and combine
        
        Returns:
            Combined DataFrame with all locations
        """
        all_data = []
        
        start_date = self.config['date_start']
        end_date = self.config['date_end']
        
        for location in self.config['locations']:
            df = self.fetch_location_data(location, start_date, end_date)
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records fetched: {len(combined)}")
        print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        
        return combined
    
    def save_data(self, df: pd.DataFrame, filepath: str = "data/raw_weather_data.csv"):
        """Save fetched data to CSV"""
        df.to_csv(filepath, index=False)
        print(f"\nData saved to {filepath}")


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = WeatherDataFetcher()
    data = fetcher.fetch_all_locations()
    fetcher.save_data(data)
