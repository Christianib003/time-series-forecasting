import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df


def handle_missing_values(df, target_column='pm2.5', strategy='ffill'):
    if strategy == 'ffill':
        df[target_column] = df[target_column].fillna(method='ffill')
    
    df[target_column] = df[target_column].fillna(method='bfill')
    
    return df


def create_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    return df


def scale_features(df, columns_to_scale, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    else:
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    return df, scaler


def create_sequences(data, n_past, n_future, target_column):
    """
    Creates sequences of past data and future targets for time series forecasting.

    Args:
        data (pd.DataFrame): The input DataFrame.
        n_past (int): The number of past time steps to use as input features.
        n_future (int): The number of future time steps to predict.
        target_column (str): The name of the column to be predicted.

    Returns:
        np.array, np.array: A tuple containing the input sequences (X) and
                            target sequences (y).
    """
    X, y = [], []
    
    target_col_idx = data.columns.get_loc(target_column)

    for i in range(n_past, len(data) - n_future + 1):
        X.append(data.iloc[i - n_past:i].values)
        
        y.append(data.iloc[i:i + n_future, target_col_idx].values)

    return np.array(X), np.array(y)