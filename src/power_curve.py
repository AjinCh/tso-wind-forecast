"""
Wind-to-power conversion using generic turbine power curve
Demonstrates uncertainty amplification from wind speed to power feed-in
"""

import numpy as np
import yaml


class WindPowerConverter:
    """
    Convert wind speed forecasts to normalized power output
    
    Uses a simplified turbine power curve typical for modern wind turbines.
    This demonstrates how forecast errors amplify near rated wind speed.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.curve = config['power_curve']
        self.cut_in = self.curve['cut_in_speed']
        self.rated = self.curve['rated_speed']
        self.cut_out = self.curve['cut_out_speed']
        self.rated_power = self.curve['rated_power']
    
    def wind_to_power(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        Convert wind speed (m/s) to normalized power output [0, 1]
        
        Power curve model:
        - Below cut-in: 0
        - Cut-in to rated: cubic increase (simplified)
        - Rated to cut-out: constant at rated power
        - Above cut-out: 0
        
        Args:
            wind_speed: Wind speed in m/s (can be array or scalar)
            
        Returns:
            Normalized power output [0, 1]
        """
        wind_speed = np.asarray(wind_speed)
        power = np.zeros_like(wind_speed, dtype=float)
        
        # Region 2: cubic power law (simplified)
        mask_ramp = (wind_speed >= self.cut_in) & (wind_speed < self.rated)
        power[mask_ramp] = self.rated_power * (
            (wind_speed[mask_ramp] - self.cut_in) / (self.rated - self.cut_in)
        ) ** 3
        
        # Region 3: rated power
        mask_rated = (wind_speed >= self.rated) & (wind_speed < self.cut_out)
        power[mask_rated] = self.rated_power
        
        return power
    
    def compute_forecast_uncertainty(self, wind_forecast: np.ndarray, 
                                    wind_error_std: float = 1.0) -> dict:
        """
        Analyze how wind speed uncertainty translates to power uncertainty
        
        This is critical for TSO operations: small wind errors can cause
        large power prediction errors, especially near rated wind speed.
        
        Args:
            wind_forecast: Predicted wind speeds (samples, hours)
            wind_error_std: Standard deviation of wind speed errors (m/s)
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Monte Carlo simulation of forecast uncertainty
        n_samples = 1000
        wind_perturbed = wind_forecast[:, np.newaxis, :] + \
                        np.random.normal(0, wind_error_std, 
                                       (wind_forecast.shape[0], n_samples, wind_forecast.shape[1]))
        
        # Convert to power
        power_nom = self.wind_to_power(wind_forecast)
        power_perturbed = self.wind_to_power(wind_perturbed)
        
        # Power uncertainty statistics
        power_std = power_perturbed.std(axis=1)
        power_range = power_perturbed.max(axis=1) - power_perturbed.min(axis=1)
        
        # Identify high-uncertainty regions (near rated wind)
        near_rated = np.abs(wind_forecast - self.rated) < 3.0  # Within 3 m/s of rated
        
        return {
            'power_forecast': power_nom,
            'power_std': power_std,
            'power_range': power_range,
            'high_uncertainty_fraction': near_rated.mean(),
            'avg_power_std': power_std.mean()
        }
    
    def analyze_critical_speed_range(self, wind_speeds: np.ndarray) -> dict:
        """
        Analyze sensitivity in the most critical wind speed range for TSOs
        
        The range around rated wind speed is where forecast errors have
        the largest impact on power prediction.
        
        Args:
            wind_speeds: Array of wind speeds to analyze
            
        Returns:
            Dictionary with sensitivity metrics
        """
        # Compute power for each speed
        power = self.wind_to_power(wind_speeds)
        
        # Find maximum gradient (highest sensitivity)
        dP_dV = np.gradient(power)
        max_sensitivity_idx = np.argmax(np.abs(dP_dV))
        
        return {
            'most_sensitive_speed': wind_speeds[max_sensitivity_idx],
            'max_sensitivity': dP_dV[max_sensitivity_idx],
            'critical_range': (self.rated - 3, self.rated + 3),
            'power_curve': power
        }


def demonstrate_uncertainty_amplification():
    """
    Demonstration for interview: show how wind errors amplify to power errors
    """
    converter = WindPowerConverter()
    
    # Simulate a forecast scenario
    wind_speeds = np.linspace(0, 25, 100)
    power = converter.wind_to_power(wind_speeds)
    
    print("\n=== Wind-to-Power Uncertainty Analysis ===")
    print(f"Turbine parameters:")
    print(f"  Cut-in speed: {converter.cut_in} m/s")
    print(f"  Rated speed: {converter.rated} m/s")
    print(f"  Cut-out speed: {converter.cut_out} m/s")
    
    # Analyze sensitivity
    analysis = converter.analyze_critical_speed_range(wind_speeds)
    print(f"\nCritical speed range for TSO operations:")
    print(f"  Most sensitive at: {analysis['most_sensitive_speed']:.1f} m/s")
    print(f"  Maximum sensitivity: {analysis['max_sensitivity']:.3f} (ΔP/Δv)")
    print(f"  Critical range: {analysis['critical_range'][0]:.0f}-{analysis['critical_range'][1]:.0f} m/s")
    
    # Uncertainty amplification example
    test_wind = np.array([8.0, 11.0, 14.0, 20.0]).reshape(1, -1)
    uncertainty = converter.compute_forecast_uncertainty(test_wind, wind_error_std=1.5)
    
    print(f"\nUncertainty amplification example:")
    print(f"  Wind forecast: {test_wind[0]} m/s")
    print(f"  Power forecast: {uncertainty['power_forecast'][0]}")
    print(f"  Power uncertainty (std): {uncertainty['power_std'][0]}")
    print(f"  High-uncertainty fraction: {uncertainty['high_uncertainty_fraction']:.1%}")
    
    print("\n✓ This analysis is critical for grid operators:")
    print("  Small wind errors near rated speed cause large power uncertainty")


if __name__ == "__main__":
    demonstrate_uncertainty_amplification()
