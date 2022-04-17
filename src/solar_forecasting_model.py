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
NUM_EPOCHS = 25
NUM_LAYERS = 2


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


def build_and_train_model(train_features, train_targets, num_layers, num_neurons, num_epochs, num_features, verbose):

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

  history = model.fit(x = train_features, y = train_targets, epochs=num_epochs, verbose=verbose)

  return model, history


#### Main Testing Loop
if __name__ == '__main__':
  ## Set path to file directory
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)

  accumulated_results = pd.DataFrame(columns=['Station', 'Features', 'Iteration', 'Persistence_Performance', 'Model_Performance'])
  
  for iteration in range(5):
    for station in STATIONS:
      filename = f'../data_cleaned/{station}_observations.csv'
      dataframe = pd.read_csv(filename, parse_dates=['timestamp']).dropna()
      additional_weather_features = set(dataframe.columns).difference(['timestamp', 'ghi'])

      all_features = ['ghi'] + list(additional_weather_features)
      ## Prepare dataframe, with boxed values for historical data
      dataframe = generate_historical_features(dataframe, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX, hours_forecasted=HOURS_FORECASTED, features=all_features)
      dataframe = generate_harmonic_historical_features(dataframe, hours_ahead=HOURS_AHEAD, hours_history=HOURS_HX)
      dataframe = generate_persistence_fcst(dataframe, hours_ahead=HOURS_AHEAD, hours_forecasted=HOURS_FORECASTED).dropna()
      
      ## Partition Dataset
      train_set = dataframe[dataframe['timestamp'].dt.year < 2021]
      test_set = dataframe[dataframe['timestamp'].dt.year >= 2021]

      ## Select which subset of features to train on
      features_per_model = { \
      'GHI_Only': ['ghi'],
      'GHI_Harmonics': ['ghi'] + HARMONIC_FEATURES,
      }
      if len(additional_weather_features) > 0:
        features_per_model.update({'GHI_Harmonics_Weather': ['ghi'] + HARMONIC_FEATURES + list(additional_weather_features)})
      
      ## Select output columns
      target_columns = [f'ghi_actual_{i}' for i in range(HOURS_FORECASTED)]

      for features, feature_list in features_per_model.items():
        ## Make sure we have output directories made
        if not os.path.isdir('../logs'):
          try: os.mkdir('../logs')
          except OSError as error: print(error)

        ## Redirect output for logging
        progress_file = open(f'../logs/{station}_{features}_{iteration}_log.txt', 'w')
        summary_file = open(f'../logs/{station}_{features}_{iteration}_summary.txt', 'w')
        sys.stdout = progress_file

        num_features = len(feature_list) 
        selected_features = [historical_feature(feature, i+HOURS_AHEAD) for i in range(HOURS_HX) for feature in feature_list]
        num_neurons = math.ceil(math.sqrt(len(selected_features)))


        ## Initialize scalers
        feature_scaler = MinMaxScaler(feature_range=(0,1))
        target_scaler = MinMaxScaler(feature_range=(0,1))

        ## Scale train set features
        training_inputs = np.array(train_set.loc[:,selected_features].values)
        feature_scaler = feature_scaler.fit(training_inputs)
        scaled_training_inputs = feature_scaler.transform(training_inputs)
        scaled_training_inputs = scaled_training_inputs.reshape(len(training_inputs), HOURS_HX, num_features)

        ## Scale train set outputs
        training_data_targets = train_set.loc[:,target_columns]
        target_scaler = target_scaler.fit(training_data_targets)
        scaled_train_targets = target_scaler.transform(training_data_targets)

        ## Train Model
        model, history = build_and_train_model( \
          train_features=scaled_training_inputs, 
          train_targets=scaled_train_targets, 
          num_neurons=num_neurons, 
          num_layers=NUM_LAYERS, 
          num_epochs=NUM_EPOCHS, 
          num_features=num_features,
          verbose=2,
          )

        progressFolder = f'../models/{station}_{features}_{iteration}'
        ## Make sure we have a folder to write model to 
        if not os.path.isdir(progressFolder):
          try: os.mkdir(progressFolder)
          except OSError as error:
            print(error)
          
        ## Save model for later use
        model.save_weights(os.path.join(progressFolder,'weightsFile.weights'))

        sys.stdout.flush()
        sys.stdout=summary_file
        progress_file.close()
        
        ## Evaluate Performance on Test Set
        test_features = np.array(test_set.loc[:,selected_features].values)
        scaled_test_features = feature_scaler.transform(test_features)
        scaled_test_features = scaled_test_features.reshape(len(test_set), HOURS_HX, num_features)

        model_predictions = model.predict(scaled_test_features)
        model_predictions_unscaled = target_scaler.inverse_transform(model_predictions)

        test_true_values = test_set.loc[:,target_columns]

        rmse_for_model = math.sqrt(mean_squared_error(y_true=test_true_values, y_pred=model_predictions_unscaled)) 

        persistence_targets = [f'ghi_persistence_fcst_{i}' for i in range(HOURS_FORECASTED)]
        persistence_predictions = test_set.loc[:,persistence_targets]
        rmse_for_persistence=math.sqrt(mean_squared_error(y_true=test_true_values, y_pred=persistence_predictions))

        print( \
f'''Model:\t{station}_{features}
num_features:\t{num_features}
features:\t{feature_list}
hours_history:\t{HOURS_HX}
num_neurons:\t{num_neurons}
num_epochs:\t{NUM_EPOCHS}
num_layers:\t{NUM_LAYERS}
'''
        )
        print(f'Persistence Accuracy: {rmse_for_persistence}, \t Model Accuracy: {rmse_for_model}')
        sys.stdout.flush()
        sys.stdout = sys.__stdout__
        summary_file.close()

        accumulated_results.loc[len(accumulated_results.index)] = [station, features, iteration, rmse_for_persistence, rmse_for_model]
      ##End features loop
    ## End station loop
  accumulated_results.to_csv('accumulated_results.csv')