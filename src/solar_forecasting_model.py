import math
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from src.constants import HARMONIC_FEATURES, HOURS_AHEAD, HOURS_FORECASTED, HOURS_HX, NUM_EPOCHS, NUM_LAYERS, STATIONS
from src.generate_features import generate_harmonic_historical_features, generate_historical_features, generate_persistence_fcst, historical_feature

from src.models import build_and_train_model_validation_early_stop

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
      dataframe = pd.read_csv(filename, parse_dates=['timestamp'])
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
        num_neurons = 24

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
        model, history = build_and_train_model_validation_early_stop( \
          train_features=scaled_training_inputs, 
          train_targets=scaled_train_targets, 
          num_neurons=num_neurons, 
          num_layers=NUM_LAYERS, 
          num_epochs=NUM_EPOCHS, 
          num_features=num_features,
          verbose=2,
          patience=5,
          )

        progressFolder = f'../models/{station}_{features}_{iteration}'
        ## Make sure we have a folder to write model to 
        if not os.path.isdir(progressFolder):
          try: os.mkdir(progressFolder)
          except OSError as error:
            print(error)
          
        ## Save model for later use
        model.save(os.path.join(progressFolder,f'{station}_{features}_{iteration}_model.mdl'))

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
        
        epochs_run_for = len(history.history['loss'])

        print( \
f'''Model:\t{station}_{features}
num_features:\t{num_features}
features:\t{feature_list}
hours_history:\t{HOURS_HX}
num_neurons:\t{num_neurons}
num_epochs:\t{epochs_run_for}
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