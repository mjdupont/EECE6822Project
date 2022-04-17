import os
import pandas as pd

DATA_SRC_DIR = "data_src/"
DATA_CLEANED_DIR = "data_cleaned/"

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
    
if __name__ == "__main__":
  clean_data()