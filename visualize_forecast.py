"""
Visualize wind forecast predictions with actual vs predicted comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import os


def plot_forecast(csv_file):
    """Create visualization for forecast data"""
    
    # Load forecast
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load actual historical data for context (last 48 hours of available data)
    actual_df = pd.read_csv('data/raw_weather_data.csv')
    actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'])
    actual_df = actual_df.tail(48)  # Last 48 hours of historical data
    
    # Extract location from filename
    filename = os.path.basename(csv_file)
    location = filename.split('_')[1].replace('-', ' ')
    forecast_time = datetime.strptime(filename.split('_')[-2] + filename.split('_')[-1].replace('.csv',''), 
                                      '%Y%m%d%H%M')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f'Wind Forecast for {location}\nGenerated: {forecast_time.strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=16, fontweight='bold')
    
    # ========== Plot 1: Actual vs Predicted Timeline ==========
    ax1 = axes[0]
    
    # Plot actual historical data
    ax1.plot(actual_df['timestamp'], actual_df['wind_speed'], 
             'g-', linewidth=2.5, marker='s', markersize=5, label='Actual (Historical)', alpha=0.8)
    
    # Plot forecast predictions
    ax1.plot(df['timestamp'], df['predicted_wind_speed_ms'], 
             'b-', linewidth=2.5, marker='o', markersize=4, label='Predicted (Forecast)')
    
    # Add vertical line to separate historical from forecast
    if len(actual_df) > 0:
        ax1.axvline(x=actual_df['timestamp'].iloc[-1], color='red', linestyle=':', 
                   linewidth=2, label='Forecast Start', alpha=0.7)
    
    # Formatting
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    ax1.set_title('Actual Historical vs Predicted Forecast Wind Speed', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== Plot 2: Forecast Detail with Uncertainty ==========
    ax2 = axes[1]
    
    # Main forecast line
    ax2.plot(df['timestamp'], df['predicted_wind_speed_ms'], 
             'b-', linewidth=2.5, marker='o', markersize=4, label='Predicted Wind Speed')
    
    # Add uncertainty bands (from model validation)
    mae = 0.62  # Mean Absolute Error from training
    ci_95 = 1.60  # 95% confidence interval
    
    ax2.fill_between(df['timestamp'], 
                     df['predicted_wind_speed_ms'] - mae,
                     df['predicted_wind_speed_ms'] + mae,
                     alpha=0.3, color='blue', label='±MAE (0.62 m/s)')
    
    ax2.fill_between(df['timestamp'], 
                     df['predicted_wind_speed_ms'] - ci_95,
                     df['predicted_wind_speed_ms'] + ci_95,
                     alpha=0.1, color='blue', label='±95% CI (1.60 m/s)')
    
    # Add threshold lines
    ax2.axhline(y=12, color='orange', linestyle='--', linewidth=1.5, label='High Wind (12 m/s)')
    ax2.axhline(y=4, color='red', linestyle='--', linewidth=1.5, label='Low Wind (4 m/s)')
    
    # Highlight peak and low
    peak_idx = df['predicted_wind_speed_ms'].idxmax()
    low_idx = df['predicted_wind_speed_ms'].idxmin()
    
    ax2.scatter(df.loc[peak_idx, 'timestamp'], df.loc[peak_idx, 'predicted_wind_speed_ms'], 
                color='green', s=200, zorder=5, marker='*', label='Peak Wind')
    ax2.scatter(df.loc[low_idx, 'timestamp'], df.loc[low_idx, 'predicted_wind_speed_ms'], 
                color='purple', s=200, zorder=5, marker='v', label='Min Wind')
    
    # Formatting
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    ax2.set_title('24-Hour Forecast Detail with Uncertainty Bands', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    # Rotate x-axis labels
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== Plot 3: Hourly Wind Distribution ==========
    ax3 = axes[2]
    
    # Create color map based on wind categories
    colors = []
    for speed in df['predicted_wind_speed_ms']:
        if speed < 4:
            colors.append('lightcoral')  # Low wind
        elif speed < 8:
            colors.append('gold')  # Moderate
        elif speed < 12:
            colors.append('lightgreen')  # Good wind
        elif speed < 17:
            colors.append('orange')  # High wind
        else:
            colors.append('red')  # Very high/gale
    
    # Bar chart
    bars = ax3.bar(df['hour_ahead'], df['predicted_wind_speed_ms'], 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (hour, speed) in enumerate(zip(df['hour_ahead'], df['predicted_wind_speed_ms'])):
        if i % 3 == 0:  # Label every 3rd hour to avoid clutter
            ax3.text(hour, speed + 0.5, f'{speed:.1f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Add threshold lines
    ax3.axhline(y=12, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=4, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Formatting
    ax3.set_xlabel('Hours Ahead', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    ax3.set_title('Hourly Wind Speed Distribution by Category', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(1, 25))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='< 4 m/s (Low)'),
        Patch(facecolor='gold', label='4-8 m/s (Moderate)'),
        Patch(facecolor='lightgreen', label='8-12 m/s (Good)'),
        Patch(facecolor='orange', label='12-17 m/s (High)'),
        Patch(facecolor='red', label='> 17 m/s (Gale)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = csv_file.replace('.csv', '_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\n" + "="*70)
    print("FORECAST STATISTICS")
    print("="*70)
    
    # Historical statistics
    if len(actual_df) > 0:
        print("\nRecent Historical (Last 48h):")
        print(f"  Average wind speed:   {actual_df['wind_speed'].mean():.2f} m/s")
        print(f"  Maximum wind speed:   {actual_df['wind_speed'].max():.2f} m/s")
        print(f"  Minimum wind speed:   {actual_df['wind_speed'].min():.2f} m/s")
    
    # Forecast statistics
    print("\nForecast (Next 24h):")
    print(f"  Average wind speed:   {df['predicted_wind_speed_ms'].mean():.2f} m/s")
    print(f"  Maximum wind speed:   {df['predicted_wind_speed_ms'].max():.2f} m/s (Hour {df.loc[peak_idx, 'hour_ahead']})")
    print(f"  Minimum wind speed:   {df['predicted_wind_speed_ms'].min():.2f} m/s (Hour {df.loc[low_idx, 'hour_ahead']})")
    print(f"  Standard deviation:   {df['predicted_wind_speed_ms'].std():.2f} m/s")
    print(f"  Wind variability:     {(df['predicted_wind_speed_ms'].std() / df['predicted_wind_speed_ms'].mean() * 100):.1f}%")
    
    # Wind category breakdown
    print("\nWind Category Breakdown:")
    category_counts = df['category'].value_counts()
    for cat, count in category_counts.items():
        print(f"  {cat:15s}: {count:2d} hours ({count/24*100:.1f}%)")
    
    # Comparison with historical average
    if len(actual_df) > 0:
        hist_avg = actual_df['wind_speed'].mean()
        forecast_avg = df['predicted_wind_speed_ms'].mean()
        diff = forecast_avg - hist_avg
        percent_change = (diff / hist_avg * 100)
        print(f"\nTrend: Forecast is {abs(diff):.2f} m/s ({'higher' if diff > 0 else 'lower'}) than recent historical average ({percent_change:+.1f}%)")


def main():
    """Visualize the most recent forecast or all forecasts"""
    
    # Find forecast files
    forecast_files = glob.glob("results/forecast_*.csv")
    
    if not forecast_files:
        print("❌ No forecast files found in results/ directory")
        print("   Run forecast_tomorrow.py first to generate predictions")
        return
    
    # Sort by modification time (most recent first)
    forecast_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n" + "="*70)
    print("WIND FORECAST VISUALIZATION - ACTUAL vs PREDICTED")
    print("="*70)
    print(f"\nFound {len(forecast_files)} forecast file(s)")
    
    # Visualize the most recent forecast
    latest_file = forecast_files[0]
    print(f"\nVisualizing most recent forecast: {os.path.basename(latest_file)}")
    print("-"*70)
    
    plot_forecast(latest_file)
    
    # Ask if user wants to see more
    if len(forecast_files) > 1:
        print(f"\n💡 {len(forecast_files)-1} older forecast(s) available")
        print("   Modify this script to visualize other forecasts")


if __name__ == "__main__":
    main()
