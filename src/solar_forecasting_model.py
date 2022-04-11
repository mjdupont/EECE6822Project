import math
import os
import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data_directory = "data_src/"
cleaned_data = "data_cleaned/"
list_of_files = []

# for root, dirs, files in os.walk(data_directory):
#     for file in files:
#         list_of_files.append(os.path.join(root,file))
# for name in list_of_files:
#     print(name)

# sample_file = list_of_files[0]

weather_measurements_to_try = ['air_temperature', 'wind_speed']
def clean_data():
  for root, dirs, files in os.walk(data_directory):
    for file in files:
      file_path = os.path.join(root,file)
      df = pd.read_csv(file_path, parse_dates=['timestamp']).set_index('timestamp')
      df = df.resample(rule='1H').mean()
      df.to_csv(os.path.join('data_cleaned', file))
      list_of_removed_substrings = ['dni', 'dhi', 'quality_flag']
      
      filtered_columns = df.columns
      for substring in list_of_removed_substrings:
        filtered_columns = [col_name for col_name in filtered_columns if not substring in col_name]
      
      new_df = df[filtered_columns]
      # Clean negative values to disallow 0s
      new_df.loc[:,'ghi'] = new_df.loc[:,'ghi'].apply(lambda x : max(0, x))
      new_df.to_csv(os.path.join('data_cleaned', (file)))
    
#clean_data()

def align_data_for_model():
  for root, dirs, files in os.walk(cleaned_data):
    for file in files:
      file_path = os.path.join(root,file)
      df = pd.read_csv(file_path)
      
      df_cpy = df.__deepcopy__()
      df_cpy['ghi_fcst'] = df_cpy['ghi'].shift(-1)
      df_cpy = df_cpy.dropna()
      rmse = math.sqrt(mean_squared_error(y_pred=df_cpy['ghi_fcst'], y_true=df_cpy['ghi']))
      print(f"RMSE of naive/persistence forecast for {file}: {rmse}")

sample_df = pd.read_csv('data_cleaned/Bondville IL_observations.csv', parse_dates=['timestamp'])

HOURS_HX = 24
HOURS_AHEAD = 1
HOURS_FORECASTED = 24

     
def generate_historical_ghi(df:pd.DataFrame, hours_ahead:int, hours_history:int, hours_forecasted:int):
  new_df = df.__deepcopy__()
  for hour_of_history in sorted(range(hours_history), reverse=True):
    hours_shifted = hour_of_history + hours_ahead
    new_df[f'ghi_km{hours_shifted}'] = df['ghi'].shift(-hours_shifted)
  
  for hour_of_forecast in range(hours_history):
      new_df[f'ghi_fcst{hour_of_forecast}'] = df['ghi'].shift(hour_of_forecast)
  
  new_df['ghi_fcst-1'] = df['ghi'].shift(-1)

  return new_df

new_df = generate_historical_ghi(sample_df, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX, hours_forecasted=HOURS_FORECASTED).dropna()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_columns = [column for column in new_df.columns if not 'timestamp' == column]
new_df[scaled_columns] = scaler.fit_transform(new_df[scaled_columns])

selected_features = [f'ghi_km{i+HOURS_AHEAD}' for i in range(HOURS_HX)]
selected_targets = [f'ghi_fcst{i}' for i in range(HOURS_FORECASTED)]

train_set = new_df[new_df['timestamp'].dt.year < 2021]
validation_set = new_df[new_df['timestamp'].dt.year >= 2021]

train_features = np.array(train_set.loc[:,selected_features].values)
train_features_len = len(train_features)
train_features = train_features.reshape(train_features_len, -1, HOURS_HX)

train_targets = train_set.loc[:,selected_targets]

validation_features = np.array(validation_set.loc[:,selected_features].values).reshape(len(validation_set), -1, HOURS_HX)

# Explanation of LSTM model: https://stackoverflow.com/questions/50488427/what-is-the-architecture-behind-the-keras-lstm-cell

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(48, return_sequences=True, input_shape=(None, HOURS_HX)),
    tf.keras.layers.LSTM(48, return_sequences=True),
    tf.keras.layers.LSTM(48, return_sequences=True),
    tf.keras.layers.LSTM(48, return_sequences=True),
    tf.keras.layers.LSTM(48, return_sequences=True),
    tf.keras.layers.LSTM(48),
    tf.keras.layers.Dense(HOURS_FORECASTED, activation='relu')
])
model.summary()

model.compile(
  loss='mse',
  optimizer='adam',
  metrics=['mse']
)

model.fit(x = train_features, y = train_targets, epochs=25)

model_pred = model.predict(validation_features)

validation_true_values = validation_set.loc[:,selected_targets]

mse_for_model = mean_squared_error(y_true=validation_true_values, y_pred=model_pred) 

persistence_targets = [f'ghi_fcst{i-1}' for i in range(HOURS_FORECASTED)]
persistence_predictions = validation_set.loc[:,persistence_targets]

mse_for_persistence=mean_squared_error(y_true=validation_true_values, y_pred=persistence_predictions)

print(mse_for_persistence, mse_for_model)