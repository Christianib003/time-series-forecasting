from __future__ import annotations
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_lstm(input_len: int, n_features: int,
               units: int = 64, n_layers: int = 2,
               dropout: float = 0.2, recurrent_dropout: float = 0.0,
               clipnorm: float = 1.0, lr: float = 1e-3) -> tf.keras.Model:
    """
    Simple stacked LSTM → Dense(1). Returns a compiled model (MSE loss).
    """
    inputs = layers.Input(shape=(input_len, n_features))
    x = inputs
    for i in range(n_layers - 1):
        x = layers.LSTM(units, return_sequences=True,
                        dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = layers.LSTM(units, return_sequences=False,
                    dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = layers.Dense(1)(x)
    model = models.Model(inputs, x)
    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    model.compile(optimizer=opt, loss="mse")
    return model
