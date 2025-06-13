import numpy as np
import pickle
import pandas as pd
from datetime import timedelta, date
from model.preprocess import preprocess
from model.utils import load_model_and_scalers
from data.fetch_era5 import load_era5_once

def scaler_data(df_coord,scalers):
    
    df_scaled, X_scaled, y_scaled, _ = preprocess(df_coord, scaler_dict=scalers, fit_scaler=False)
    # print(f"🔮 preprocess {len(X_scaled)}")
    # print(f"🔮 preprocess {len(y_scaled)}")
    
    X_vars = [c for c in df_scaled.columns if c not in ["valid_time", 
                                                        "latitude", 
                                                        "longitude", 
                                                        "weather_event"] and c in scalers['x'].feature_names_in_]
    
    if df_scaled.shape[0] >= 12:
        input_seq = df_scaled[X_vars].values[-12:]
        sequence = input_seq[np.newaxis, :, :]
    else:
        raise ValueError(f"No hay suficientes pasos temporales. Se requieren 12 y hay {df_scaled.shape[0]}")

    return sequence

def process_dfs(df,scalers,model,variables_objetivo):
    
    predicciones = []
    ubicaciones = df[["latitude", "longitude"]].drop_duplicates()

    for _, row in ubicaciones.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        # 1. Filtrar por coordenada
        df_coord = df[(df["latitude"] == lat) & (df["longitude"] == lon)]

        if len(df_coord) < 12:
            print(f"para latitude: {lat} y longitude: {lon} No hay 12 pasos")
            print(df_coord)
            continue  # No hay suficientes datos
        else:
            sequence = scaler_data(df_coord,scalers)
            
            # print(f"🔮 sequence shape: {sequence.shape}")
            
            pred_scaled = model.predict(sequence)[0]
            # print(f"🔮 pred_scaled shape después de [0]: {pred_scaled.shape}")
            
            pred_desescalado = scalers['y'].inverse_transform(pred_scaled)
            
            valid_time_base = df_coord["valid_time"].max()

            for i in range(12):
                future_time = valid_time_base + timedelta(hours=6 * (i + 1))  # +6h por paso
                pred_row = {
                    "latitude": lat,
                    "longitude": lon,
                    "valid_time": future_time  
                }

                pred_row.update({var: float(pred_desescalado[i, j]) for j, var in enumerate(variables_objetivo)})

                predicciones.append(pred_row)

    return predicciones
    
def predict_from_latest(region, model, scalers):
    
    df = load_era5_once(region)
    
    print(f"🔮 load_era5_once {df.info()}")
    
    print(f"🔮 scalers: {scalers}")
    df = df.sort_values(["latitude", "longitude", "valid_time"])
    variables_objetivo = ['swh', 't2m', 'u10', 'v10','msl', 'sst','lsm','q_100','q_250','q_500','q_850','t_100','t_250','t_500',
     't_850','u_100','u_250','u_500','u_850','v_100','v_250','v_500','v_850','z_100','z_250','z_500','z_850']
    
    df = df[['valid_time', 'latitude', 'longitude','weather_event']+variables_objetivo]
    
    variables_objetivo.remove('lsm')
    
    predicciones = process_dfs(df,scalers,model,variables_objetivo)
    
    df_pred = pd.DataFrame(predicciones)
    min_valid = df_pred["valid_time"].min().strftime("%Y-%m-%dT%H%M")
    max_valid = df_pred["valid_time"].max()
    
    try:
        file_name = fr"D:\PDG-ARMADA-FORECAST\predictions\prediccion_{region}_{min_valid}_{max_valid.strftime('%Y-%m-%dT%H%M')}.csv"
        df_pred.to_csv(file_name, index=False)
    except:
        print(f"error generando csv de {file_name}-revise que no tenga el archivo abierto")
    
    if max_valid < pd.Timestamp(date.today() + timedelta(days=1)):
        df_pred_merge = df_pred.merge(df[['latitude', 
                                          'longitude',
                                          'weather_event',
                                          'lsm']].drop_duplicates())
        
        # predicciones = process_dfs(df_pred_merge,scalers,model,variables_objetivo)
        print(df_pred_merge.info())
        print(f"df_pred: {len(df_pred)}")
    
    return predicciones
