"""
sequence_models.py

Module C - LSTM & GRU Models
Team Member: Arthi

This module contains sequence-based deep learning models
for customer purchase prediction and transaction pattern analysis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
def create_sequences(data, sequence_length=10):
"""
    Convert time-series data into input sequences for LSTM/GRU.
"""
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y)
def build_lstm_model(input_shape):
    """
    Build an LSTM model for customer purchase sequence prediction.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
  def build_gru_model(input_shape):
    """
    Build a GRU model for customer purchase sequence prediction.
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),

        GRU(32),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model
    def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """
    Train the LSTM/GRU model.
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return history
