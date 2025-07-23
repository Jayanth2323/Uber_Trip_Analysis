import pandas as pd
import os

def load_and_resample(data_path="data/raw"):
    files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'uber-raw-data' in f]
    dfs = [pd.read_csv(os.path.join(data_path, f)) for f in files]
    
    df = pd.concat(dfs, ignore_index=True)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.rename(columns={'Date/Time': 'Date'}).sort_values('Date')
    df.set_index('Date', inplace=True)
    
    # Hourly aggregation
    hourly_df = df['Base'].resample('H').count().reset_index()
    hourly_df.columns = ['Date', 'Count']
    hourly_df.set_index('Date', inplace=True)
    
    return hourly_df
