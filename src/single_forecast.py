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

from solar_forecasting_model import HARMONIC_FEATURES, HOURS_AHEAD, HOURS_FORECASTED, HOURS_HX, SELECTED_FEATURES, STATIONS, build_and_train_model, generate_harmonic_historical_features, generate_historical_features, generate_persistence_fcst, historical_feature

station = STATIONS[2]

print(station)
sample_df = pd.read_csv(f'data_cleaned/{station}_observations.csv', parse_dates=['timestamp']).dropna()
weather_features = set(sample_df.columns).difference(['ghi', 'timestamp'])
print(weather_features)
all_features = SELECTED_FEATURES + list(weather_features)
new_df = generate_historical_features(sample_df, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX, hours_forecasted=HOURS_FORECASTED, features=all_features)
new_df = generate_harmonic_historical_features(new_df, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX)
new_df = generate_persistence_fcst(new_df, hours_ahead=HOURS_AHEAD, hours_forecasted=HOURS_FORECASTED)
new_df = new_df.dropna()

feature_scaler = MinMaxScaler(feature_range=(0,1))
target_scaler = MinMaxScaler(feature_range=(0,1))

features_to_use = ['ghi']#all_features + HARMONIC_FEATURES
## Get relevant columns for predictions
selected_features = [historical_feature(feature, i+HOURS_AHEAD) for i in range(HOURS_HX) for feature in features_to_use]
num_features = len(features_to_use)
target_columns = [f'ghi_actual_{i}' for i in range(HOURS_FORECASTED)]

## Partition Dataset
train_set = new_df[new_df['timestamp'].dt.year < 2021]
test_set = new_df[new_df['timestamp'].dt.year >= 2021]

training_inputs = np.array(train_set.loc[:,selected_features].values)
train_features_len = len(training_inputs)

feature_scaler = feature_scaler.fit(training_inputs)
scaled_training_inputs = feature_scaler.transform(training_inputs)
scaled_training_inputs = scaled_training_inputs.reshape(train_features_len, HOURS_HX, num_features)
print(scaled_training_inputs.shape)

training_data_targets = train_set.loc[:,target_columns]
target_scaler = target_scaler.fit(training_data_targets)
scaled_train_targets = target_scaler.transform(training_data_targets)

model, history = build_and_train_model( \
  train_features=scaled_training_inputs, 
  train_targets=scaled_train_targets, 
  num_neurons=math.ceil(math.sqrt(len(selected_features))), 
  num_layers=2, 
  num_epochs=25, 
  num_features=num_features,
  verbose=1,
  )

#model.save_weights(f"../models/{station}/{features}/weightsFile.weights")

test_features = np.array(test_set.loc[:,selected_features].values)
scaled_test_features = feature_scaler.transform(test_features)
scaled_test_features = scaled_test_features.reshape(len(test_set), HOURS_HX, num_features)



model_predictions = model.predict(scaled_test_features)
model_predictions_unscaled = target_scaler.inverse_transform(model_predictions)

test_true_values = test_set.loc[:,target_columns]

mse_for_model = mean_squared_error(y_true=test_true_values, y_pred=model_predictions_unscaled) 

persistence_targets = [f'ghi_persistence_fcst_{i}' for i in range(HOURS_FORECASTED)]
persistence_predictions = test_set.loc[:,persistence_targets]
mse_for_persistence=mean_squared_error(y_true=test_true_values, y_pred=persistence_predictions)

print(f'Persistence Accuracy: {mse_for_persistence}, \t Model Accuracy: {mse_for_model}')

model_predictions_unscaled = pd.DataFrame(model_predictions_unscaled, columns=test_true_values.columns)

y_max = max(new_df['ghi']) * 1.10

for i in range(10):
  true_values = test_true_values.iloc[i*24,:]
  pers_values = persistence_predictions.iloc[i*24,:]
  model_values = model_predictions_unscaled.iloc[i*24,:]

  x_values = [x for x in range(len(test_true_values))]
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x_values, y=true_values, name='Actual'))
  fig.add_trace(go.Scatter(x=x_values, y=pers_values, name='Persistence'))
  fig.add_trace(go.Scatter(x=x_values, y=model_values, name='Model'))
  fig.update_yaxes(range=[0,y_max])
  fig.update_layout(title=f'Index {i*24} forecast')
  fig.show()