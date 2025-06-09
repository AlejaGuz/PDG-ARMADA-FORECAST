import numpy as np
import pickle
import pandas as pd
from datetime import timedelta
from model.preprocess import preprocess
from model.utils import load_model_and_scalers
from data.fetch_era5 import load_era5_once

def predict_from_latest(region, model, scalers):
    
    df = load_era5_once(region)
    
    print(f"🔮 load_era5_once {df.info()}")
    
    print(f"🔮 scalers: {scalers}")
    df = df.sort_values(["latitude", "longitude", "valid_time"])
    variables_objetivo = ['swh', 't2m', 'u10', 'v10','msl', 'sst','lsm','q_100','q_250','q_500','q_850','t_100','t_250','t_500',
     't_850','u_100','u_250','u_500','u_850','v_100','v_250','v_500','v_850','z_100','z_250','z_500','z_850']
    
    df = df[['valid_time', 'latitude', 'longitude','weather_event']+variables_objetivo]
    
    variables_objetivo.remove('lsm')
    predicciones = []

    # Obtener combinaciones únicas de lat y lon
    ubicaciones = df[["latitude", "longitude"]].drop_duplicates()

    for _, row in ubicaciones.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        # 1. Filtrar por coordenada
        df_coord = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        if len(df_coord) < 12:
            continue  # No hay suficientes datos
    
    # df_scaled, _ = preprocess(df, scaler_dict=scalers, fit_scaler=False)
    df_scaled, X_scaled, y_scaled, _ = preprocess(df_coord, scaler_dict=scalers, fit_scaler=False)
    print(f"🔮 preprocess {len(X_scaled)}")
    print(f"🔮 preprocess {len(y_scaled)}")
    
    # sequence = df_scaled.tail(12).drop(columns=["valid_time", "latitude", "longitude"]).values
    X_vars = [c for c in df_scaled.columns if c not in ["valid_time", 
                                                        "latitude", 
                                                        "longitude", 
                                                        "weather_event"] and c in scalers['x'].feature_names_in_]
    input_seq = df_scaled[X_vars].values[-12:]  # (12, n_features_x)
    sequence = input_seq[np.newaxis, :, :]
    
    
    print(f"🔮 sequence shape: {sequence.shape}")
    pred_scaled = model.predict(sequence)[0]
    
    print(f"🔮 pred_scaled shape después de [0]: {pred_scaled.shape}")
    
    # pred = {}
    # columnas = df.drop(columns=["valid_time", "latitude", "longitude"]).columns
    # pred_desescalado = scalers['global'].inverse_transform(pred_scaled.reshape(1, -1)).flatten()
    pred_desescalado = scalers['y'].inverse_transform(pred_scaled)

    # pred = {col: [val] for col, val in zip(columnas, pred_desescalado)}
    
    valid_time_base = df["valid_time"].max()

    for i in range(12):
        future_time = valid_time_base + timedelta(hours=6 * (i + 1))  # +6h por paso
        pred_row = {
            "latitude": lat,
            "longitude": lon,
            "valid_time": future_time  # Reemplaza "step" por el timestamp real
        }
        pred_row.update({var: pred_desescalado[i, j] for j, var in enumerate(variables_objetivo)})
        predicciones.append(pred_row)
    
    return predicciones
