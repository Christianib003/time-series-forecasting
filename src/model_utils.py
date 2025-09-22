import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from io import StringIO
import contextlib

from .data_utils import (
    load_data, create_time_features, scale_features, 
    create_sequences, handle_missing_values, create_advanced_features
)

def build_model(input_shape, model_type='lstm', units=[32], bidirectional=False, dropout_rate=0.0, dense_units=[]):
    """
    Builds a flexible RNN model with support for stacked layers, bidirectional processing,
    dropout, and a final dense head.
    """
    model = Sequential()
    
    # Add recurrent layers
    for i, unit_count in enumerate(units):
        is_last_recurrent_layer = (i == len(units) - 1)
        
        # THE FIX IS HERE: The last recurrent layer should NOT return a sequence
        # before passing its output to the Dense layers.
        return_sequences = not is_last_recurrent_layer
        
        layer_type = LSTM if model_type.lower() == 'lstm' else GRU
        
        if i == 0:
            layer = layer_type(unit_count, return_sequences=return_sequences, input_shape=input_shape)
        else:
            layer = layer_type(unit_count, return_sequences=return_sequences)
        
        if bidirectional:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Add final dense head
    for unit_count in dense_units:
        model.add(Dense(unit_count, activation='relu'))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    # Add the final output layer
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    print("✅ Model built and compiled successfully.")
    return model
def train_model(model, X_train, y_train, X_val, y_val, model_path, epochs=50, batch_size=32):
    # ... (rest of the function is unchanged)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    return history

def evaluate_model(history, model, X_val, y_val):
    # ... (rest of the function is unchanged)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    y_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred.flatten())**2))
    print("--- Model Evaluation ---")
    print(f"✅ Final Validation RMSE: {rmse:.2f}")
    return rmse

def log_experiment(model, history, rmse, log_path, exp_id, model_type, sequence_length,
                   batch_size, scaler_type, optimizer, notes=""):
    # ... (rest of the function is unchanged)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    summary_stream = StringIO()
    with contextlib.redirect_stdout(summary_stream):
        model.summary()
    model_summary = summary_stream.getvalue()
    final_loss = history.history['val_loss'][-1]
    num_epochs = len(history.history['loss'])
    log_content = f"""
# --- Experiment Log ---
**Experiment ID:** {exp_id}
**Model Type:** {model_type}
---
## **Parameters**
---
- **Sequence Length (N_PAST):** {sequence_length}
- **Scaler Type:** {scaler_type}
- **Batch Size:** {batch_size}
- **Epochs Trained:** {num_epochs}
- **Optimizer:** {optimizer}
---
## **Performance Metrics**
---
- **Final Validation RMSE:** {rmse:.4f}
- **Final Validation Loss (MSE):** {final_loss:.4f}
---
## **Model Architecture**
---
{model_summary}
---
## **Notes**
---
{notes}
"""
    with open(log_path, 'w') as f:
        f.write(log_content)
    print(f"✅ Comprehensive experiment details logged to: {log_path}")

# ... (all other functions in model_utils.py remain the same) ...

def generate_submission(model_path, train_scaled, scaler, columns_to_scale, N_PAST, TARGET_COL, submission_filename):
    """Generates a Kaggle submission file using a stable batch prediction method."""
    test_raw_df = load_data('../data/test.csv')
    test_processed_df = handle_missing_values(test_raw_df) 
    test_processed_df = create_time_features(test_processed_df)
    
    # THE FIX IS HERE: Call the function with is_train=False
    test_processed_df = create_advanced_features(test_processed_df, is_train=False)
    
    test_processed_df[TARGET_COL] = 0
    
    _, test_scaled, _ = scale_features(
        train_df=train_scaled.copy(),
        val_df=test_processed_df.copy(), 
        columns_to_scale=columns_to_scale, 
        scaler=scaler
    )
    
    combined_data = pd.concat([train_scaled.tail(N_PAST), test_scaled])
    X_test, _ = create_sequences(combined_data.values, N_PAST, combined_data.columns.get_loc(TARGET_COL))

    best_model = load_model(model_path)
    y_pred_scaled = best_model.predict(X_test)
    
    pm25_col_index_in_scaler = columns_to_scale.index(TARGET_COL)
    dummy_array = np.zeros((len(y_pred_scaled), len(columns_to_scale)))
    dummy_array[:, pm25_col_index_in_scaler] = y_pred_scaled.flatten()
    actual_predictions = scaler.inverse_transform(dummy_array)[:, pm25_col_index_in_scaler]
    actual_predictions[actual_predictions < 0] = 0

    num_predictions = len(actual_predictions)
    prediction_datetimes = test_raw_df.index[-num_predictions:]
    
    submission_df = pd.DataFrame({
        "row ID": prediction_datetimes.strftime('%Y-%m-%d %-H:%M:%S'),
        "pm2.5": actual_predictions
    })

    submission_path = f'../submissions/{submission_filename}'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Saved {len(submission_df)} rows to {submission_path}")
    return submission_df