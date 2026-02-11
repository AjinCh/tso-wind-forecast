"""
Optimized Ensemble Framework
Combines multiple models with dynamic, performance-based weighting
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path


class DynamicEnsemble:
    """
    Advanced ensemble that combines predictions from multiple models
    
    Features:
    - Dynamic weighting optimized for each forecast hour
    - Performance-based weight adjustment
    - Uncertainty-aware combinations
    - Temporal adaptation
    """
    
    def __init__(self, models=None, method='optimized'):
        """
        Args:
            models: List of (name, model) tuples
            method: 'simple' (average), 'weighted' (fixed weights), 
                   'optimized' (learned weights), 'temporal' (hour-specific weights)
        """
        self.models = models or []
        self.method = method
        self.weights = None
        self.temporal_weights = None  # Weights per forecast hour
        self.performance_history = []
    
    def add_model(self, name, model):
        """Add a model to the ensemble"""
        self.models.append((name, model))
    
    def _optimize_weights(self, predictions, y_true, method='mae'):
        """
        Find optimal weights to minimize error
        
        Args:
            predictions: List of prediction arrays from each model
            y_true: Ground truth
            method: 'mae' or 'mse'
        
        Returns:
            Optimal weights (sum to 1)
        """
        n_models = len(predictions)
        
        def objective(weights):
            """Objective function to minimize"""
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Weighted combination
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * pred
            
            # Calculate error
            if method == 'mae':
                return mean_absolute_error(y_true.ravel(), ensemble_pred.ravel())
            else:
                return mean_squared_error(y_true.ravel(), ensemble_pred.ravel())
        
        # Constraints: weights >= 0, sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x / result.x.sum()  # Normalize
        else:
            print("⚠️  Optimization failed, using equal weights")
            return initial_weights
    
    def _optimize_temporal_weights(self, predictions, y_true, forecast_hours=24):
        """
        Optimize weights separately for each forecast hour
        Different models may be better at different horizons
        
        Args:
            predictions: List of prediction arrays (samples, forecast_hours)
            y_true: Ground truth (samples, forecast_hours)
            forecast_hours: Number of forecast hours
        
        Returns:
            Array of shape (forecast_hours, n_models) with optimal weights
        """
        n_models = len(predictions)
        temporal_weights = np.zeros((forecast_hours, n_models))
        
        print("\nOptimizing hour-specific weights...")
        for hour in range(forecast_hours):
            # Extract predictions for this hour
            hour_preds = [pred[:, hour] for pred in predictions]
            hour_true = y_true[:, hour]
            
            # Optimize weights for this hour
            weights = self._optimize_weights(hour_preds, hour_true)
            temporal_weights[hour, :] = weights
            
            if hour % 6 == 0 or hour == forecast_hours - 1:
                print(f"  Hour {hour+1:2d}: " + " | ".join([f"{w:.3f}" for w in weights]))
        
        return temporal_weights
    
    def _diversity_weighted_combination(self, predictions, y_true):
        """
        Weight models based on both accuracy and diversity
        Diverse models provide better ensemble performance
        
        Args:
            predictions: List of prediction arrays
            y_true: Ground truth
        
        Returns:
            Optimal weights considering diversity
        """
        n_models = len(predictions)
        
        # Calculate individual model errors
        errors = []
        for pred in predictions:
            mae = mean_absolute_error(y_true.ravel(), pred.ravel())
            errors.append(mae)
        
        # Calculate pairwise diversity (correlation)
        diversity_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Correlation between errors (lower = more diverse)
                error_i = (predictions[i] - y_true).ravel()
                error_j = (predictions[j] - y_true).ravel()
                correlation = np.corrcoef(error_i, error_j)[0, 1]
                diversity_matrix[i, j] = correlation
                diversity_matrix[j, i] = correlation
        
        # Average diversity for each model
        diversity_scores = 1 - np.abs(diversity_matrix).mean(axis=1)
        
        # Combine accuracy and diversity
        # Lower error = better, higher diversity = better
        accuracy_scores = 1 / (1 + np.array(errors))  # Normalize errors
        combined_scores = accuracy_scores * (1 + diversity_scores)
        
        # Normalize to get weights
        weights = combined_scores / combined_scores.sum()
        
        print("\nDiversity-Weighted Combination:")
        for i, (name, _) in enumerate(self.models):
            print(f"  {name:20s}: weight={weights[i]:.3f} | "
                  f"MAE={errors[i]:.3f} | diversity={diversity_scores[i]:.3f}")
        
        return weights
    
    def fit(self, val_predictions, y_val):
        """
        Learn optimal weights using validation data
        
        Args:
            val_predictions: Dictionary {model_name: predictions} or list of predictions
            y_val: Validation ground truth
        """
        # Convert dict to list if needed
        if isinstance(val_predictions, dict):
            predictions = [val_predictions[name] for name, _ in self.models]
        else:
            predictions = val_predictions
        
        print("\n" + "="*70)
        print("ENSEMBLE WEIGHT OPTIMIZATION")
        print("="*70)
        
        if self.method == 'simple':
            # Equal weights
            n_models = len(predictions)
            self.weights = np.ones(n_models) / n_models
            print("Method: Simple Average")
            print("Weights: " + " | ".join([f"{w:.3f}" for w in self.weights]))
        
        elif self.method == 'weighted':
            # Optimized global weights
            self.weights = self._optimize_weights(predictions, y_val)
            print("Method: Optimized Global Weights (MAE minimization)")
            for i, (name, _) in enumerate(self.models):
                print(f"  {name}: {self.weights[i]:.3f}")
        
        elif self.method == 'temporal':
            # Hour-specific weights
            self.temporal_weights = self._optimize_temporal_weights(predictions, y_val)
            print("Method: Hour-Specific Weights")
            print("  (Weights shown above)")
        
        elif self.method == 'diversity':
            # Diversity-aware weights
            self.weights = self._diversity_weighted_combination(predictions, y_val)
            print("Method: Diversity-Aware Weighting")
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Calculate ensemble validation performance
        ensemble_pred = self.predict(predictions)
        ensemble_mae = mean_absolute_error(y_val.ravel(), ensemble_pred.ravel())
        
        print(f"\n✓ Ensemble Validation MAE: {ensemble_mae:.3f} m/s")
        print("="*70)
    
    def predict(self, predictions):
        """
        Generate ensemble prediction
        
        Args:
            predictions: List of prediction arrays or dict {model_name: predictions}
        
        Returns:
            Ensemble prediction
        """
        # Convert dict to list if needed
        if isinstance(predictions, dict):
            predictions = [predictions[name] for name, _ in self.models]
        
        if self.method == 'temporal' and self.temporal_weights is not None:
            # Hour-specific weighting
            ensemble_pred = np.zeros_like(predictions[0])
            for hour in range(ensemble_pred.shape[1]):
                for i, pred in enumerate(predictions):
                    ensemble_pred[:, hour] += self.temporal_weights[hour, i] * pred[:, hour]
        
        else:
            # Global weighting
            if self.weights is None:
                # Default to equal weights if not fitted
                self.weights = np.ones(len(predictions)) / len(predictions)
            
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def evaluate_individual_models(self, predictions, y_true):
        """
        Evaluate each model's contribution to the ensemble
        
        Args:
            predictions: List of predictions or dict
            y_true: Ground truth
        
        Returns:
            DataFrame with model performance metrics
        """
        import pandas as pd
        
        if isinstance(predictions, dict):
            predictions = [predictions[name] for name, _ in self.models]
        
        results = []
        
        for i, (name, _) in enumerate(self.models):
            pred = predictions[i]
            mae = mean_absolute_error(y_true.ravel(), pred.ravel())
            rmse = np.sqrt(mean_squared_error(y_true.ravel(), pred.ravel()))
            
            # Hour-by-hour performance
            hour_maes = [mean_absolute_error(y_true[:, h], pred[:, h]) for h in range(y_true.shape[1])]
            
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'Hour_1_MAE': hour_maes[0],
                'Hour_12_MAE': hour_maes[11],
                'Hour_24_MAE': hour_maes[23],
                'Weight': self.weights[i] if self.weights is not None else 1/len(self.models)
            })
        
        # Add ensemble performance
        ensemble_pred = self.predict(predictions)
        ensemble_mae = mean_absolute_error(y_true.ravel(), ensemble_pred.ravel())
        ensemble_rmse = np.sqrt(mean_squared_error(y_true.ravel(), ensemble_pred.ravel()))
        hour_maes = [mean_absolute_error(y_true[:, h], ensemble_pred[:, h]) for h in range(y_true.shape[1])]
        
        results.append({
            'Model': '🏆 ENSEMBLE',
            'MAE': ensemble_mae,
            'RMSE': ensemble_rmse,
            'Hour_1_MAE': hour_maes[0],
            'Hour_12_MAE': hour_maes[11],
            'Hour_24_MAE': hour_maes[23],
            'Weight': 1.0
        })
        
        df = pd.DataFrame(results)
        
        return df
    
    def save(self, filepath):
        """Save ensemble configuration"""
        config = {
            'method': self.method,
            'weights': self.weights,
            'temporal_weights': self.temporal_weights,
            'model_names': [name for name, _ in self.models]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✓ Ensemble configuration saved to {filepath}")
    
    def load(self, filepath):
        """Load ensemble configuration"""
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.method = config['method']
        self.weights = config['weights']
        self.temporal_weights = config['temporal_weights']
        
        print(f"✓ Ensemble configuration loaded from {filepath}")
        print(f"  Method: {self.method}")
        if self.weights is not None:
            print(f"  Weights: {self.weights}")


class StackingEnsemble(DynamicEnsemble):
    """
    Stacking ensemble: Use meta-learner to combine base models
    Meta-learner learns optimal non-linear combinations
    """
    
    def __init__(self, base_models=None, meta_learner=None):
        """
        Args:
            base_models: List of (name, model) tuples
            meta_learner: Sklearn-compatible model for meta-learning
        """
        super().__init__(base_models, method='stacking')
        self.meta_learner = meta_learner
    
    def fit(self, val_predictions, y_val):
        """
        Train meta-learner on base model predictions
        
        Args:
            val_predictions: Dictionary or list of base model predictions
            y_val: Validation ground truth
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        if isinstance(val_predictions, dict):
            predictions = [val_predictions[name] for name, _ in self.models]
        else:
            predictions = val_predictions
        
        # Stack predictions as features (samples, n_models * forecast_hours)
        X_meta = np.hstack([pred.reshape(pred.shape[0], -1) for pred in predictions])
        y_meta = y_val.ravel()
        
        # Initialize meta-learner if not provided
        if self.meta_learner is None:
            self.meta_learner = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        
        print("\n" + "="*70)
        print("STACKING ENSEMBLE - Training Meta-Learner")
        print("="*70)
        print(f"Base models: {len(predictions)}")
        print(f"Meta-features: {X_meta.shape[1]}")
        print(f"Meta-learner: {type(self.meta_learner).__name__}")
        
        # Train meta-learner
        self.meta_learner.fit(X_meta, y_meta)
        
        # Evaluate
        meta_pred = self.meta_learner.predict(X_meta).reshape(y_val.shape)
        mae = mean_absolute_error(y_val.ravel(), meta_pred.ravel())
        
        print(f"✓ Meta-learner validation MAE: {mae:.3f} m/s")
        print("="*70)
    
    def predict(self, predictions):
        """Generate stacked prediction using meta-learner"""
        if isinstance(predictions, dict):
            predictions = [predictions[name] for name, _ in self.models]
        
        # Stack predictions
        X_meta = np.hstack([pred.reshape(pred.shape[0], -1) for pred in predictions])
        
        # Meta-prediction
        meta_pred = self.meta_learner.predict(X_meta)
        
        # Reshape to (samples, forecast_hours)
        ensemble_pred = meta_pred.reshape(predictions[0].shape)
        
        return ensemble_pred


if __name__ == "__main__":
    # Demonstration
    print("Dynamic Ensemble Framework")
    print("="*70)
    print("\nSupported methods:")
    print("  - simple: Equal weighting")
    print("  - weighted: Optimized global weights")
    print("  - temporal: Hour-specific weights")
    print("  - diversity: Diversity-aware weighting")
    print("  - stacking: Meta-learner (non-linear combination)")
    print("\n✓ Ready to combine LSTM, Transformer, XGBoost, LightGBM models")
