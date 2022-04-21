import math
import os
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import sys


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.solar_forecasting_model import HOURS_FORECASTED, HOURS_HX


def build_and_train_model_validation_hx_dropout(train_features, train_targets, validation_features, validation_targets, target_scaler:MinMaxScaler, num_layers, num_neurons, num_epochs, num_features, verbose):
  
  # Explanation of LSTM model: https://stackoverflow.com/questions/50488427/what-is-the-architecture-behind-the-keras-lstm-cell
  def lstm_layers(layer, len):
    if len == 1:
      return tf.keras.layers.LSTM(num_neurons, 
      input_shape=(HOURS_HX, num_features)
      )
    elif layer == 1:
      return tf.keras.layers.LSTM(num_neurons, return_sequences=True, 
      input_shape=(HOURS_HX, num_features)
      )
    elif layer == len:
      return tf.keras.layers.LSTM(num_neurons)
    else :
      return tf.keras.layers.LSTM(num_neurons, return_sequences=True)

  model_layers = [lstm_layers(layer+1, num_layers) for layer in range(num_layers)] + [ tf.keras.layers.Dense(HOURS_FORECASTED, activation='relu') ]

  model = tf.keras.Sequential(model_layers)

  model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mse']
  )

  train_rmse_history = list()
  validation_rmse_history = list()
  
  for epoch in range(num_epochs):
    history = model.fit(x = train_features, y = train_targets, epochs=1, verbose=verbose)
    train_true= target_scaler.inverse_transform(train_targets)
    validation_true = target_scaler.inverse_transform(validation_targets)
    train_pred = target_scaler.inverse_transform(model.predict(train_features))
    validation_pred = target_scaler.inverse_transform(model.predict(validation_features))
    train_rmse = math.sqrt(mean_squared_error(y_true=train_true, y_pred=train_pred))
    validation_rmse = math.sqrt(mean_squared_error(y_true=validation_true, y_pred=validation_pred))
  
    train_rmse_history = train_rmse_history + [train_rmse]
    validation_rmse_history = validation_rmse_history + [validation_rmse]

  return model, history, (train_rmse_history, validation_rmse_history)
