"""
GRU model architecture for short-term wind speed forecasting
(GRU chosen over LSTM for faster training with similar accuracy)
"""

import os
# Set JAX as Keras backend before importing keras
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import yaml
from typing import Tuple


class WindForecastLSTM:
    """
    Multivariate GRU with Multi-Head Attention for 24-hour wind speed forecasting
    (GRU provides ~30% faster training than LSTM with comparable accuracy)
    (Attention mechanism helps focus on most relevant past timesteps)
    
    Architecture: GRU → Multi-Head Attention → GRU → Dense → Output
    
    Uses past 72 hours of meteorological data to predict next 24 hours of wind speed.
    Designed for grid balancing support in high wind penetration scenarios.
    
    Attention Benefit: Identifies which past hours are most predictive (e.g., frontal
    passages 12 hours ago may be more relevant than steady conditions 2 hours ago).
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.model_config = config['model']
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build GRU architecture with Attention mechanism
        (Attention helps model focus on most relevant timesteps)
        
        Args:
            input_shape: (lookback_hours, n_features)
        """
        try:
            import keras
            from keras import layers, models, optimizers
        except ImportError:
            raise ImportError("Keras is required. Install with: pip install keras jax")
        
        # Functional API for attention mechanism
        inputs = layers.Input(shape=input_shape)
        
        # First GRU layer with return sequences (for attention)
        gru1 = layers.GRU(
            self.model_config['hidden_units'],
            return_sequences=True,
            dropout=self.model_config['dropout'],
            name='gru_layer_1'
        )(inputs)
        
        # Multi-Head Attention layer
        # Allows model to focus on most relevant past timesteps
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.model_config['hidden_units'] // 4,
            dropout=self.model_config['dropout'],
            name='attention'
        )(gru1, gru1)
        
        # Add & Norm (residual connection + layer normalization)
        attention_residual = layers.Add()([gru1, attention])
        attention_norm = layers.LayerNormalization(name='attention_norm')(attention_residual)
        
        # Second GRU layer (processes attention-weighted sequence)
        gru2 = layers.GRU(
            self.model_config['hidden_units'] // 2,
            dropout=self.model_config['dropout'],
            name='gru_layer_2'
        )(attention_norm)
        
        # Dense layers for forecast horizon
        dense1 = layers.Dense(
            self.model_config['hidden_units'] // 4,
            activation='relu',
            name='dense_1'
        )(gru2)
        dropout = layers.Dropout(self.model_config['dropout'])(dense1)
        
        # Output layer: 24-hour forecast
        outputs = layers.Dense(
            self.model_config['forecast_hours'],
            name='forecast_output'
        )(dropout)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='WindForecast_GRU_Attention')
        
        # Use MAE loss instead of MSE to reduce regression toward mean
        # MSE heavily penalizes extreme errors, causing model to predict conservatively
        # MAE treats all errors equally, better for capturing full wind speed range
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.model_config['learning_rate']),
            loss='mae',  # Changed from 'mse' - better for extremes
            metrics=['mse']  # Still track MSE for comparison
        )
        
        self.model = model
        
        print("\n🎯 Model Architecture with Attention Mechanism:")
        print(model.summary())
        print("\n✓ Attention allows model to focus on most relevant timesteps")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the LSTM model
        
        Args:
            X_train: Training input sequences (samples, lookback, features)
            y_train: Training targets (samples, forecast_hours)
            X_val: Validation input sequences
            y_val: Validation targets
        """
        try:
            import keras
            from keras import callbacks
        except ImportError:
            raise ImportError("Keras is required")
        
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        print("\nTraining LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate 24-hour wind speed forecasts
        
        Args:
            X: Input sequences (samples, lookback, features)
            
        Returns:
            Predictions (samples, 24)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath: str = "models/lstm_wind_forecast.h5"):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath: str = "models/lstm_wind_forecast.h5"):
        """Load trained model"""
        try:
            import keras
        except ImportError:
            raise ImportError("Keras is required")
        
        # Load without compilation to avoid Keras 2/3 compatibility issues
        self.model = keras.models.load_model(filepath, compile=False)
        
        # Recompile with Keras 3
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.model_config['learning_rate']),
            loss='mae',  # Changed from 'mse' - better for extremes
            metrics=['mse']
        )
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test model architecture
    import pickle
    
    print("Loading dataset...")
    with open("data/processed_dataset.pkl", 'rb') as f:
        dataset = pickle.load(f)
    
    model = WindForecastLSTM()
    model.build_model(input_shape=(dataset['X_train'].shape[1], dataset['X_train'].shape[2]))
