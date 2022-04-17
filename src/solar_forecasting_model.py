import math
import os
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

DATA_SRC_DIR = "data_src/"
DATA_CLEANED_DIR = "data_cleaned/"
HOURS_HX = 24
HOURS_AHEAD = 1
HOURS_FORECASTED = 24
SELECTED_FEATURES = ['ghi',]
ALL_HOURS = [x for x in range(24)]
SELECTED_HOURS = ALL_HOURS#[0]
STATIONS = [\
  'Natural Energy Laboratory of Hawaii Authority',
  'Seattle Washington',                           
  'Hanford California',                           
  'Salt Lake City Utah',                          
  'Table Mountain Boulder CO',                    
  'Goodwin Creek MS',                             
  'Bondville IL',                                 
  'Titusville FL',                                
  'Sterling Virginia',                            
  'Millbrook NY',                                 
  ]

def seasonal_harmonics(datetime:pd.Timestamp):

  dayofyear = datetime.dayofyear
  hourofday = datetime.hour
  results = \
    dict( \
      cos1doy = np.cos(np.pi * 1 * dayofyear / 365),
      sin1doy = np.sin(np.pi * 2 * dayofyear / 365),
      cos2doy = np.cos(np.pi * 2 * dayofyear / 365),
      sin2doy = np.sin(np.pi * 2 * dayofyear / 365),
      cos4doy = np.cos(np.pi * 4 * dayofyear / 365),
      sin4doy = np.sin(np.pi * 4 * dayofyear / 365),
      cos1hod = np.cos(np.pi * 1 * hourofday /  24),
      sin1hod = np.sin(np.pi * 1 * hourofday /  24),
      cos2hod = np.cos(np.pi * 2 * hourofday /  24),
      sin2hod = np.sin(np.pi * 2 * hourofday /  24),
      cos4hod = np.cos(np.pi * 4 * hourofday /  24),
      sin4hod = np.sin(np.pi * 4 * hourofday /  24),
      )
  return results

HARMONIC_FEATURES = list(seasonal_harmonics(pd.Timestamp(datetime.datetime.now())).keys())


def clean_data():
  for root, dirs, files in os.walk(DATA_SRC_DIR):
    for file in files:
      file_path = os.path.join(root,file)
      df = pd.read_csv(file_path, parse_dates=['timestamp']).set_index('timestamp')

      ## Resample to 1 hour groups by mean.
      df = df.resample(rule='1H').mean()

      ## Remove unused features
      ## Note 'quality flag' is a bitstring indicating quality issues with data, but these are calculated after data is uploaded to SFA.
      ## They're removed both due to difficulty parsing, and because they only identify features already present in the data.
      list_of_removed_substrings = ['dni', 'dhi', 'quality_flag']
      filtered_columns = df.columns
      for substring in list_of_removed_substrings:
        filtered_columns = [col_name for col_name in filtered_columns if not substring in col_name]
      df = df[filtered_columns]
      # Clean negative values to disallow 0s
      df.loc[:,'ghi'] = df.loc[:,'ghi'].apply(lambda x : max(0, x))
      df.to_csv(os.path.join(DATA_CLEANED_DIR, (file)))
    
## clean_data()

def historical_feature(feature:str, hours_shifted:int): return f'{feature}_history_km_{hours_shifted}'
def persistence_forecast(feature:str, hours_shifted:int): return f''

def generate_persistence_fcst(df:pd.DataFrame, hours_ahead:int, hours_forecasted:int):
  ## Calculate delay; Need the time-aligned 24 hour period before target hour in each array.
  ## For example, with 24 horus forecasted, and 6 hour delay, we want to grab the 24 hours of values before time -6 hours
  ## These data then need to be aligned with the 24 hours forecasted.
  df = df.__deepcopy__()

  for hour in range(hours_forecasted):
    ## Generate all ranges of actual values forecasted for this date
    hours_back = hour - (hours_forecasted + hours_ahead)
    hours_back_index = hours_back%24
    ## Note a shift 
    df[f'ghi_persistence_fcst_{hours_back_index}'] = df['ghi'].shift(-hours_back)
  return df

     
def generate_historical_features(df:pd.DataFrame, hours_ahead:int, hours_history:int, hours_forecasted:int, features:list[str]):
  new_df = df.__deepcopy__()
  ## All Weather Features
  new_features = dict()
  for feature in features:
    for hour_of_history in sorted(range(hours_history), reverse=True):
      hours_shifted = hour_of_history + hours_ahead
      new_features.update({historical_feature(feature, hours_shifted): new_df[feature].shift(hours_shifted)})

  ## Target values
  for hour in range(hours_forecasted):
    new_features.update({f'ghi_actual_{hour}': new_df['ghi'].shift(-hour)})

  new_features_dataframe = pd.DataFrame(new_features)
  new_df = pd.concat([new_df, new_features_dataframe], axis=1)

  ## Defragment DataFrame
  return new_df.copy()

def generate_harmonic_features(df:pd.DataFrame):
  new_df = df.__deepcopy__()
  seasonal_features = new_df['timestamp'].apply(lambda x : seasonal_harmonics(x))
  new_features = dict()
  for h_feature in HARMONIC_FEATURES:
    new_features.update({h_feature: seasonal_features.apply(lambda row: row[h_feature])})

  new_df = pd.concat([new_df,pd.DataFrame(new_features)], axis=1)

  ## Defragment DataFrame
  return new_df

def generate_harmonic_historical_features(df:pd.DataFrame, hours_ahead:int, hours_history:int):
  new_df = generate_harmonic_features(df)
  
  new_features = dict()
  for h_feature in HARMONIC_FEATURES:
    for hour_of_history in sorted(range(hours_history), reverse=True):
      hours_shifted = hour_of_history + hours_ahead
      new_features.update({historical_feature(h_feature, hours_shifted): new_df[h_feature].shift(hours_shifted)})

  new_df = pd.concat([new_df,pd.DataFrame(new_features)], axis=1)
  ## Defragment DataFrame
  return new_df.copy()

def prepare_dataframe(df:pd.DataFrame, hours_ahead:int, hours_history:int, hours_forecasted:int, features:list[str], include_harmonics:bool):
  new_df = generate_historical_features(df, hours_ahead=hours_ahead, hours_history=hours_history, hours_forecasted=hours_forecasted, features=features)
  if include_harmonics:
    new_df = generate_harmonic_historical_features(new_df, hours_ahead, hours_history)
  new_df = generate_persistence_fcst(new_df, hours_ahead, hours_forecasted)


def build_and_train_model(train_features, train_targets, num_layers, num_neurons, num_epochs, num_features):

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

  history = model.fit(x = train_features, y = train_targets, epochs=num_epochs)

  return model, history


#### Main Testing Loop
### Test Cases:

for station in STATIONS:
  filename = f'data_cleaned/{station}_observations.csv'
  dataframe = pd.read_csv(filename, parse_dates=['timestamp']).dropna()
  additional_weather_signals = set(dataframe.columns).difference(['timestamp', 'ghi'])




sample_df = pd.read_csv('data_cleaned/Bondville IL_observations.csv', parse_dates=['timestamp']).dropna()
new_df = generate_historical_features(sample_df, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX, hours_forecasted=HOURS_FORECASTED, features=SELECTED_FEATURES)
new_df = generate_harmonic_historical_features(new_df, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX)
new_df = generate_persistence_fcst(new_df, hours_ahead=HOURS_AHEAD, hours_forecasted=HOURS_FORECASTED)
new_df = new_df.dropna()
new_df = new_df[(new_df['timestamp'].dt.hour).isin(SELECTED_HOURS)]

feature_scaler = MinMaxScaler(feature_range=(0,1))
target_scaler = MinMaxScaler(feature_range=(0,1))

features_to_use = SELECTED_FEATURES + HARMONIC_FEATURES
## Get relevant columns for predictions
selected_features = [historical_feature(feature, i+HOURS_AHEAD) for i in range(HOURS_HX) for feature in features_to_use]
num_features = len(features_to_use)
target_columns = [f'ghi_actual_{i}' for i in range(HOURS_FORECASTED)]

## Partition Dataset
train_set = new_df[new_df['timestamp'].dt.year < 2021]
test_set = new_df[new_df['timestamp'].dt.year >= 2021]

train_features = np.array(train_set.loc[:,selected_features].values)
train_features_len = len(train_features)

feature_scaler = feature_scaler.fit(train_features)
scaled_train_features = feature_scaler.transform(train_features)
scaled_train_features = scaled_train_features.reshape(train_features_len, HOURS_HX, num_features)
print(scaled_train_features.shape)

train_targets = train_set.loc[:,target_columns]
target_scaler = target_scaler.fit(train_targets)
scaled_train_targets = target_scaler.transform(train_targets)

model, history = build_and_train_model(train_features=scaled_train_features, train_targets=scaled_train_targets, num_neurons=5, num_layers=1, num_epochs=10, num_features=num_features)


test_features = np.array(test_set.loc[:,selected_features].values)
scaled_test_features = feature_scaler.transform(test_features)
scaled_test_features = scaled_test_features.reshape(len(test_set), -1, HOURS_HX)


model_pred = model.predict(scaled_test_features)
model_pred_unscaled = target_scaler.inverse_transform(model_pred)

test_true_values = test_set.loc[:,target_columns]

mse_for_model = mean_squared_error(y_true=test_true_values, y_pred=model_pred_unscaled) 

persistence_targets = [f'ghi_persistence_fcst_{i}' for i in range(HOURS_FORECASTED)]
persistence_predictions = test_set.loc[:,persistence_targets]
mse_for_persistence=mean_squared_error(y_true=test_true_values, y_pred=persistence_predictions)

print(f'Persistence Accuracy: {mse_for_persistence}, \t Model Accuracy: {mse_for_model}')

model_pred_unscaled = pd.DataFrame(model_pred_unscaled, columns=test_true_values.columns)

test_true_values.to_csv("test.csv")
persistence_predictions.to_csv("persistence.csv")
model_pred_unscaled.to_csv("model_pred.csv")

y_max = max(new_df['ghi']) * 1.10

for i in range(10):
  true_values = test_true_values.iloc[i*24,:]
  pers_values = persistence_predictions.iloc[i*24,:]
  model_values = model_pred_unscaled.iloc[i*24,:]

  x_values = [x for x in range(len(test_true_values))]
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=x_values, y=true_values, name='Actual'))
  fig.add_trace(go.Scatter(x=x_values, y=pers_values, name='Persistence'))
  fig.add_trace(go.Scatter(x=x_values, y=model_values, name='Model'))
  fig.update_yaxes(range=[0,y_max])
  fig.update_layout(title=f'Index {i*24} forecast')
  fig.show()