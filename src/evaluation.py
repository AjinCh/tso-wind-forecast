"""
TSO-focused evaluation metrics for wind forecasting
Emphasizes operational relevance over academic benchmarks
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class TSOEvaluator:
    """
    Evaluation metrics focused on grid operator needs
    
    Key TSO concerns:
    - Error growth over forecast horizon
    - Performance during high-wind events (critical for balancing)
    - Absolute errors in operational units
    - Uncertainty quantification
    """
    
    def __init__(self, forecast_hours: int = 24):
        self.forecast_hours = forecast_hours
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute standard forecast metrics
        
        Args:
            y_true: Actual wind speeds (samples, forecast_hours)
            y_pred: Predicted wind speeds (samples, forecast_hours)
            
        Returns:
            Dictionary of metrics
        """
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Mean Absolute Percentage Error (avoid division by zero)
        mask = y_true > 1e-6
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Bias': bias
        }
    
    def horizon_error_growth(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Analyze error growth across forecast horizon
        
        Critical for TSO: errors typically grow with lead time
        
        Args:
            y_true: Actual values (samples, forecast_hours)
            y_pred: Predictions (samples, forecast_hours)
            
        Returns:
            DataFrame with metrics per forecast hour
        """
        results = []
        
        for hour in range(self.forecast_hours):
            mae = np.mean(np.abs(y_true[:, hour] - y_pred[:, hour]))
            rmse = np.sqrt(np.mean((y_true[:, hour] - y_pred[:, hour]) ** 2))
            
            results.append({
                'forecast_hour': hour + 1,
                'MAE': mae,
                'RMSE': rmse
            })
        
        return pd.DataFrame(results)
    
    def high_wind_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                            threshold: float = 12.0) -> Dict[str, float]:
        """
        Evaluate performance during high-wind events
        
        High winds are most critical for grid balancing and congestion management.
        This is a key TSO concern.
        
        Args:
            y_true: Actual wind speeds (samples, forecast_hours)
            y_pred: Predicted wind speeds (samples, forecast_hours)
            threshold: Wind speed threshold defining "high wind" (m/s)
            
        Returns:
            Metrics for high-wind subset
        """
        high_wind_mask = y_true >= threshold
        
        if high_wind_mask.sum() == 0:
            return {'high_wind_samples': 0}
        
        y_true_high = y_true[high_wind_mask]
        y_pred_high = y_pred[high_wind_mask]
        
        return {
            'high_wind_samples': high_wind_mask.sum(),
            'high_wind_fraction': high_wind_mask.mean(),
            'high_wind_MAE': np.mean(np.abs(y_true_high - y_pred_high)),
            'high_wind_RMSE': np.sqrt(np.mean((y_true_high - y_pred_high) ** 2))
        }
    
    def rapid_change_detection(self, y_true: np.ndarray, y_pred: np.ndarray,
                              change_threshold: float = 3.0) -> Dict[str, float]:
        """
        Evaluate performance during rapid wind changes
        
        Rapid changes (frontal passages) are most challenging for grid operations.
        
        Args:
            y_true: Actual wind speeds (samples, forecast_hours)
            y_pred: Predicted wind speeds (samples, forecast_hours)
            change_threshold: Wind speed change threshold (m/s/hour)
            
        Returns:
            Metrics for rapid change periods
        """
        # Compute hourly changes
        changes = np.abs(np.diff(y_true, axis=1))
        rapid_change_mask = changes >= change_threshold
        
        if rapid_change_mask.sum() == 0:
            return {'rapid_change_samples': 0}
        
        # Evaluate on hours following rapid changes
        errors = np.abs(y_true[:, 1:] - y_pred[:, 1:])
        rapid_errors = errors[rapid_change_mask]
        
        return {
            'rapid_change_samples': rapid_change_mask.sum(),
            'rapid_change_MAE': np.mean(rapid_errors),
            'rapid_change_frequency': rapid_change_mask.mean()
        }
    
    def full_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Complete TSO-focused evaluation
        
        Args:
            y_true: Actual wind speeds (samples, forecast_hours)
            y_pred: Predicted wind speeds (samples, forecast_hours)
            
        Returns:
            Comprehensive evaluation dictionary
        """
        print("\n" + "="*60)
        print("TSO-FOCUSED EVALUATION REPORT")
        print("="*60)
        
        # Overall metrics
        overall = self.compute_metrics(y_true, y_pred)
        print("\n1. Overall Forecast Performance:")
        print(f"   MAE:  {overall['MAE']:.3f} m/s")
        print(f"   RMSE: {overall['RMSE']:.3f} m/s")
        print(f"   MAPE: {overall['MAPE']:.2f}%")
        print(f"   Bias: {overall['Bias']:+.3f} m/s")
        
        # Horizon analysis
        horizon_df = self.horizon_error_growth(y_true, y_pred)
        print("\n2. Error Growth Over Forecast Horizon:")
        print(f"   Hour 1 MAE:  {horizon_df.iloc[0]['MAE']:.3f} m/s")
        print(f"   Hour 12 MAE: {horizon_df.iloc[11]['MAE']:.3f} m/s")
        print(f"   Hour 24 MAE: {horizon_df.iloc[23]['MAE']:.3f} m/s")
        error_growth = (horizon_df.iloc[23]['MAE'] - horizon_df.iloc[0]['MAE']) / horizon_df.iloc[0]['MAE'] * 100
        print(f"   Error growth (1h→24h): {error_growth:.1f}%")
        
        # High-wind performance
        high_wind = self.high_wind_performance(y_true, y_pred)
        print("\n3. High-Wind Event Performance (≥12 m/s):")
        if high_wind.get('high_wind_samples', 0) > 0:
            print(f"   High-wind occurrence: {high_wind['high_wind_fraction']*100:.1f}%")
            print(f"   High-wind MAE:  {high_wind['high_wind_MAE']:.3f} m/s")
            print(f"   High-wind RMSE: {high_wind['high_wind_RMSE']:.3f} m/s")
        else:
            print("   No high-wind samples in test set")
        
        # Rapid changes
        rapid = self.rapid_change_detection(y_true, y_pred)
        print("\n4. Rapid Wind Change Events (≥3 m/s/hour):")
        if rapid.get('rapid_change_samples', 0) > 0:
            print(f"   Rapid change frequency: {rapid['rapid_change_frequency']*100:.1f}%")
            print(f"   Rapid change MAE: {rapid['rapid_change_MAE']:.3f} m/s")
        else:
            print("   No rapid change events detected")
        
        print("\n" + "="*60)
        print("KEY INSIGHT FOR TSO OPERATIONS:")
        print("Forecast errors increase during rapid wind changes,")
        print("which are also the most critical periods for grid operations.")
        print("="*60 + "\n")
        
        return {
            'overall': overall,
            'horizon': horizon_df,
            'high_wind': high_wind,
            'rapid_change': rapid
        }
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    save_path: str = "results/evaluation_plots.png"):
        """
        Create TSO-relevant evaluation plots
        
        Args:
            y_true: Actual values (samples, forecast_hours)
            y_pred: Predictions (samples, forecast_hours)
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Error growth over horizon
        horizon_df = self.horizon_error_growth(y_true, y_pred)
        axes[0, 0].plot(horizon_df['forecast_hour'], horizon_df['MAE'], 'b-', linewidth=2, label='MAE')
        axes[0, 0].plot(horizon_df['forecast_hour'], horizon_df['RMSE'], 'r--', linewidth=2, label='RMSE')
        axes[0, 0].set_xlabel('Forecast Hour', fontsize=11)
        axes[0, 0].set_ylabel('Error (m/s)', fontsize=11)
        axes[0, 0].set_title('Error Growth Over Forecast Horizon', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot for all hours
        axes[0, 1].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=1)
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect forecast')
        axes[0, 1].set_xlabel('Actual Wind Speed (m/s)', fontsize=11)
        axes[0, 1].set_ylabel('Predicted Wind Speed (m/s)', fontsize=11)
        axes[0, 1].set_title('Forecast vs Actual (All Hours)', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = (y_pred - y_true).flatten()
        axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        axes[1, 0].set_xlabel('Forecast Error (m/s)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sample forecast trajectory
        sample_idx = np.random.randint(0, len(y_true))
        hours = np.arange(1, self.forecast_hours + 1)
        axes[1, 1].plot(hours, y_true[sample_idx], 'b-o', linewidth=2, markersize=4, label='Actual')
        axes[1, 1].plot(hours, y_pred[sample_idx], 'r--s', linewidth=2, markersize=4, label='Forecast')
        axes[1, 1].fill_between(hours, y_true[sample_idx], y_pred[sample_idx], alpha=0.2)
        axes[1, 1].set_xlabel('Forecast Hour', fontsize=11)
        axes[1, 1].set_ylabel('Wind Speed (m/s)', fontsize=11)
        axes[1, 1].set_title('Sample 24-Hour Forecast', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test with dummy data
    evaluator = TSOEvaluator()
    
    # Simulate test data
    n_samples = 500
    y_true = np.random.uniform(3, 15, (n_samples, 24))
    y_pred = y_true + np.random.normal(0, 1.5, (n_samples, 24))
    
    results = evaluator.full_evaluation(y_true, y_pred)
