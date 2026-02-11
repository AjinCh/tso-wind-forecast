"""
Advanced Transformer-based model for wind forecasting with quantile regression
Implements: Temporal Fusion Transformer architecture + Multi-quantile prediction
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import keras
from keras import layers, models, optimizers, callbacks
import keras.ops as ops


class QuantileLoss:
    """
    Quantile loss for probabilistic forecasting
    Critical for TSO operations - provides uncertainty bounds (P10, P50, P90)
    """
    
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        """
        Args:
            quantiles: List of quantile levels to predict (e.g., [0.1, 0.5, 0.9])
        """
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
    
    def __call__(self, y_true, y_pred):
        """
        Compute quantile loss across all quantiles
        
        y_true: shape (batch, forecast_hours)
        y_pred: shape (batch, forecast_hours * n_quantiles)
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            # Extract predictions for this quantile
            y_pred_q = y_pred[:, i::self.n_quantiles]
            
            # Quantile loss: asymmetric penalty
            error = y_true - y_pred_q
            loss = ops.maximum(q * error, (q - 1) * error)
            losses.append(ops.mean(loss))
        
        return ops.mean(ops.stack(losses))
    
    def get_config(self):
        return {'quantiles': self.quantiles}


class TemporalFusionTransformer:
    """
    Temporal Fusion Transformer for multi-horizon wind forecasting
    
    Key innovations:
    - Multi-head self-attention for temporal dependencies
    - Gated residual connections
    - Variable selection networks
    - Multi-quantile outputs for uncertainty quantification
    """
    
    def __init__(self, 
                 lookback_hours=96,
                 forecast_hours=24,
                 n_features=10,
                 d_model=128,
                 n_heads=8,
                 n_layers=4,
                 dropout=0.1,
                 quantiles=[0.1, 0.5, 0.9]):
        """
        Args:
            lookback_hours: Input sequence length
            forecast_hours: Output forecast horizon
            n_features: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            quantiles: Quantile levels for probabilistic forecast
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.model = None
        self.history = None
    
    def _positional_encoding(self, length, depth):
        """Generate sinusoidal positional encodings"""
        positions = np.arange(length)[:, np.newaxis]
        # Use depth//2 since we concatenate sin and cos
        depths = np.arange(depth // 2)[np.newaxis, :] / (depth // 2)
        
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )
        
        return pos_encoding.astype(np.float32)
    
    def _gated_residual_network(self, x, units, name_prefix):
        """
        Gated Residual Network (GRN) for flexible feature transformation
        Allows model to skip transformations when not beneficial
        """
        # Dense transformation
        dense1 = layers.Dense(units, activation='elu', name=f'{name_prefix}_dense1')(x)
        dense2 = layers.Dense(units, name=f'{name_prefix}_dense2')(dense1)
        
        # Gating mechanism
        gate = layers.Dense(units, activation='sigmoid', name=f'{name_prefix}_gate')(x)
        
        # Gated output
        gated = layers.Multiply(name=f'{name_prefix}_multiply')([gate, dense2])
        
        # Residual connection if dimensions match
        if x.shape[-1] == units:
            output = layers.Add(name=f'{name_prefix}_add')([x, gated])
        else:
            skip = layers.Dense(units, name=f'{name_prefix}_skip')(x)
            output = layers.Add(name=f'{name_prefix}_add')([skip, gated])
        
        output = layers.LayerNormalization(name=f'{name_prefix}_norm')(output)
        
        return output
    
    def _variable_selection(self, x, name_prefix):
        """
        Variable Selection Network
        Learns importance weights for each input feature
        """
        # Flatten temporal dimension
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        n_features = x.shape[-1]
        
        # Flatten to (batch * time, features)
        x_flat = layers.Reshape((seq_len * n_features,))(x)
        
        # Feature selection weights
        weights = layers.Dense(seq_len * n_features, activation='softmax', 
                              name=f'{name_prefix}_weights')(x_flat)
        
        # Apply weights
        weighted = layers.Multiply(name=f'{name_prefix}_multiply')([x_flat, weights])
        
        # Reshape back
        output = layers.Reshape((seq_len, n_features), name=f'{name_prefix}_reshape')(weighted)
        
        return output
    
    def build_model(self):
        """Build Temporal Fusion Transformer architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(self.lookback_hours, self.n_features), name='input')
        
        # Variable selection
        x = self._variable_selection(inputs, 'var_select')
        
        # Input projection to d_model dimensions
        x = layers.Dense(self.d_model, name='input_projection')(x)
        
        # Add positional encoding
        positions = self._positional_encoding(self.lookback_hours, self.d_model)
        position_layer = layers.Embedding(
            input_dim=self.lookback_hours,
            output_dim=self.d_model,
            weights=[positions],
            trainable=False,
            name='positional_encoding'
        )
        # Create a layer that outputs the position indices for any batch
        pos_input = layers.Lambda(
            lambda x: ops.repeat(ops.arange(self.lookback_hours)[None, :], repeats=ops.shape(x)[0], axis=0),
            name='position_indices'
        )(x)
        pos_encoding = position_layer(pos_input)
        x = layers.Add(name='add_position')([x, pos_encoding])
        
        # Stack of Transformer layers
        for i in range(self.n_layers):
            # Multi-head self-attention
            attention = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout,
                name=f'attention_{i}'
            )(x, x)
            
            attention = layers.Dropout(self.dropout, name=f'attn_dropout_{i}')(attention)
            
            # Residual connection and normalization
            x = layers.Add(name=f'attn_add_{i}')([x, attention])
            x = layers.LayerNormalization(name=f'attn_norm_{i}')(x)
            
            # Feed-forward network with Gated Residual Network
            ffn = self._gated_residual_network(x, self.d_model, f'grn_{i}')
            ffn = layers.Dropout(self.dropout, name=f'ffn_dropout_{i}')(ffn)
            
            x = layers.Add(name=f'ffn_add_{i}')([x, ffn])
            x = layers.LayerNormalization(name=f'ffn_norm_{i}')(x)
        
        # Temporal pooling: use last timestep + global average
        last_step = layers.Lambda(lambda x: x[:, -1, :], name='last_step')(x)
        global_avg = layers.GlobalAveragePooling1D(name='global_avg')(x)
        x = layers.Concatenate(name='concat_pool')([last_step, global_avg])
        
        # Decoder with Gated Residual Network
        x = self._gated_residual_network(x, self.d_model * 2, 'decoder_grn1')
        x = layers.Dense(self.d_model, activation='relu', name='decoder_dense1')(x)
        x = layers.Dropout(self.dropout, name='decoder_dropout')(x)
        x = self._gated_residual_network(x, self.d_model, 'decoder_grn2')
        
        # Multi-quantile output layer
        # Shape: (batch, forecast_hours * n_quantiles)
        outputs = layers.Dense(
            self.forecast_hours * self.n_quantiles,
            name='quantile_outputs'
        )(x)
        
        # Build model
        model = models.Model(inputs=inputs, outputs=outputs, name='TemporalFusionTransformer')
        
        # Compile with quantile loss
        quantile_loss = QuantileLoss(self.quantiles)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=quantile_loss
        )
        
        self.model = model
        
        print("\n" + "="*70)
        print("TEMPORAL FUSION TRANSFORMER - ARCHITECTURE")
        print("="*70)
        model.summary()
        
        total_params = sum([np.prod(p.shape) for p in model.trainable_weights])
        print(f"\n✓ Total parameters: {total_params:,}")
        print(f"✓ Architecture: {self.n_layers}-layer Transformer")
        print(f"✓ Attention: {self.n_heads} heads, {self.d_model // self.n_heads} dim per head")
        print(f"✓ Quantiles: {self.quantiles} (P10, P50, P90)")
        print(f"✓ Features: Variable selection + Gated residual networks")
        print("="*70 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        """
        Train the Transformer model
        
        Args:
            X_train: Training sequences (samples, lookback, features)
            y_train: Training targets (samples, forecast_hours)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        
        print("Training Temporal Fusion Transformer...")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Training samples: {len(X_train):,}, Validation: {len(X_val):,}\n")
        
        # Replicate y for each quantile (model outputs stacked quantiles)
        y_train_rep = y_train
        y_val_rep = y_val
        
        self.history = self.model.fit(
            X_train, y_train_rep,
            validation_data=(X_val, y_val_rep),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
        return self.history
    
    def predict(self, X):
        """
        Generate multi-quantile predictions
        
        Args:
            X: Input sequences (samples, lookback, features)
        
        Returns:
            Dictionary with quantile predictions:
            {
                'P10': predictions at 10th percentile,
                'P50': predictions at 50th percentile (median),
                'P90': predictions at 90th percentile
            }
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get predictions: shape (samples, forecast_hours * n_quantiles)
        preds = self.model.predict(X, verbose=0)
        
        # Reshape to (samples, forecast_hours, n_quantiles)
        samples = preds.shape[0]
        preds_reshaped = preds.reshape(samples, self.forecast_hours, self.n_quantiles)
        
        # Extract each quantile
        quantile_preds = {}
        for i, q in enumerate(self.quantiles):
            q_name = f'P{int(q*100)}'
            quantile_preds[q_name] = preds_reshaped[:, :, i]
        
        return quantile_preds
    
    def predict_median(self, X):
        """Get median (P50) prediction only"""
        quantile_preds = self.predict(X)
        return quantile_preds['P50']
    
    def save_model(self, filepath):
        """Save trained model weights (JAX-compatible)"""
        if self.model is None:
            raise ValueError("No model to save")
        # Use weights-only format for JAX backend compatibility
        self.model.save_weights(filepath)
        print(f"[OK] Model weights saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model weights"""
        if self.model is None:
            # Need to rebuild architecture first
            raise ValueError("Model architecture must be built before loading weights")
        self.model.load_weights(filepath)
        print(f"[OK] Model weights loaded from {filepath}")
