from datetime import datetime
import numpy as np
import pandas as pd
import datetime


DATA_SRC_DIR = "data_src/"
DATA_CLEANED_DIR = "data_cleaned/"
HOURS_HX = 24
HOURS_AHEAD = 1
HOURS_FORECASTED = 24
SELECTED_FEATURES = ['ghi',]
ALL_HOURS = [x for x in range(24)]
SELECTED_HOURS = ALL_HOURS
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
NUM_EPOCHS = 500
NUM_LAYERS = 2

def seasonal_harmonics(datetime:pd.Timestamp):
  
  dayofyear = datetime.dayofyear
  hourofday = datetime.hour
  results = \
    dict( \
      cos1doy = np.cos(2 * np.pi * 1 * dayofyear / 365),
      sin1doy = np.sin(2 * np.pi * 1 * dayofyear / 365),
      cos2doy = np.cos(2 * np.pi * 2 * dayofyear / 365),
      sin2doy = np.sin(2 * np.pi * 2 * dayofyear / 365),
      cos4doy = np.cos(2 * np.pi * 4 * dayofyear / 365),
      sin4doy = np.sin(2 * np.pi * 4 * dayofyear / 365),
      cos1hod = np.cos(2 * np.pi * 1 * hourofday /  24),
      sin1hod = np.sin(2 * np.pi * 1 * hourofday /  24),
      cos2hod = np.cos(2 * np.pi * 2 * hourofday /  24),
      sin2hod = np.sin(2 * np.pi * 2 * hourofday /  24),
      cos4hod = np.cos(2 * np.pi * 4 * hourofday /  24),
      sin4hod = np.sin(2 * np.pi * 4 * hourofday /  24),
      )
  return results

HARMONIC_FEATURES = list(seasonal_harmonics(pd.Timestamp(datetime.datetime.now())).keys())