

import pandas as pd

from src.constants import HARMONIC_FEATURES, seasonal_harmonics


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

def generate_harmonic_features(df:pd.DataFrame) -> pd.DataFrame:
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
