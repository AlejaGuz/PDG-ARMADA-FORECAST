import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def preprocess(df, scaler_dict=None, fit_scaler=False):
    X_vars = [col for col in df.columns if col not in ["valid_time", "latitude", "longitude", "weather_event"]]
    y_vars = [col for col in df.columns if col not in ["valid_time", "latitude", "longitude", "weather_event","lsm"]]
    
    print(f"Preprocessing scaler_dict: {scaler_dict}")
    if scaler_dict is None:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
    else:
        # scaler = scaler_dict['global']
        x_scaler = scaler_dict['x']
        y_scaler = scaler_dict['y']

    if fit_scaler:
        X_scaled = x_scaler.fit_transform(df[X_vars])
        y_scaled = y_scaler.fit_transform(df[y_vars])
        # scaled_values = scaler.fit_transform(df[X_vars])
    else:
        X_scaled = x_scaler.transform(df[X_vars])
        y_scaled = y_scaler.transform(df[y_vars])
        # scaled_values = scaler.transform(df[X_vars])

    df[X_vars] = X_scaled
    print(f"X_scaled: {X_scaled}")
    return df, X_scaled, y_scaled, {"x": x_scaler, "y": y_scaler}
    # return df, scaler

def clean_and_merge(dfs):
    from functools import reduce
    df = reduce(lambda left, right: pd.merge(left, right, on=["valid_time", "latitude", "longitude"], how="outer"), dfs)
    df = df.sort_values(["latitude", "longitude", "valid_time"])
    df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
    return df
