import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def load_data(file_path):
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    df = df.drop('No', axis=1)
    print("✅ Data loaded successfully.")
    return df

def handle_missing_values(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    print(f"✅ Missing values handled.")
    return df

def create_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    print("✅ Time-based features created.")
    return df

def split_data(df, split_ratio=0.85):
    split_index = int(len(df) * split_ratio)
    df_train = df.iloc[:split_index]
    df_val = df.iloc[split_index:]
    print(f"✅ Data split into training and validation sets.")
    return df_train, df_val

# THIS FUNCTION IS UPGRADED
def scale_features(train_df, val_df, columns_to_scale, scaler_type='robust', scaler=None):
    """
    Scales features. If a scaler is provided, it uses it. 
    Otherwise, it creates and fits a new one.
    """
    if scaler is None:
        if scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        train_df.loc[:, columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    else:
        # If scaler is provided, we assume it's already fitted
        train_df.loc[:, columns_to_scale] = scaler.transform(train_df[columns_to_scale])

    # Always transform the validation/test set
    val_df.loc[:, columns_to_scale] = scaler.transform(val_df[columns_to_scale])
    
    print(f"✅ Features scaled using {scaler.__class__.__name__}.")
    return train_df, val_df, scaler

def create_sequences(data, n_past, target_col_idx):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i-n_past:i])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)


def create_advanced_features(df, target_column='pm2.5', lags=[24], is_train=True):
    """
    Creates advanced features. Target-dependent features are only created if is_train=True.
    """
    # These features can be created for any dataset
    df['DEWP_x_TEMP'] = df['DEWP'] * df['TEMP']
    df['PRES_x_Iws'] = df['PRES'] * df['Iws']
    
    # These features can ONLY be created if the target column exists
    if is_train:
        for lag in lags:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    # The initial rows will have NaNs after creating lags, so we must handle them
    df = df.fillna(method='bfill')
    print(f"✅ Advanced features created.")
    return df