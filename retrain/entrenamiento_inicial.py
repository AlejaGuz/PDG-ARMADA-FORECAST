import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("PATHS:", sys.path)

from model.preprocess import preprocess
from model.trainer import build_lstm_model, get_callbacks
from model.utils import save_model_and_scalers
import pandas as pd
import numpy as np
import os
import logging
import time
from sklearn.preprocessing import StandardScaler


df_oni = pd.read_csv(r'datasets\oni.csv')
df_oni['Fecha'] = pd.to_datetime(df_oni['Fecha'])
df_oni['fecha_mes'] = df_oni['Fecha'].dt.to_period('M').dt.to_timestamp()

# logger = logging.getLogger("uvicorn.error")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_train_data(path_dataset,region):
    global df_oni
    
    df = pd.read_parquet(path_dataset)
    
    df.columns = [c.replace(".0","") for c in df.columns]
    df['fecha_mes'] = df['valid_time'].dt.to_period('M').dt.to_timestamp()


    df = df.merge(
        df_oni[['fecha_mes', 'Fase_ENSO']],
        on='fecha_mes',
        how='left'
    )

    df['weather_event'] = df['Fase_ENSO'].apply(
        lambda x: 1 if x in ['El Niño', 'La Niña'] else 0
    )

    df.drop(columns=['fecha_mes', 'Fase_ENSO'], inplace=True)
    
    return df
    
def entrenamiento_inicial(path_dataset, region):
    print(f"📂 Cargando dataset histórico de {region} desde {path_dataset}")
    
    df = get_train_data(path_dataset,region)
    df = df.sort_values(["latitude", "longitude", "valid_time"]).reset_index(drop=True)
    print(f"📂 Dataset cargado con {len(df)} filas y {len(df.columns)} columnas")
    
    t0 = time.time()
    logger.info("🔄 Inicio preprocess")
    # df_scaled, scalers = preprocess(df, fit_scaler=True)
    _, X_scaled, y_scaled, scalers = preprocess(df, fit_scaler=True)
    
    t1 = time.time()
    logger.info(f"✔ preprocess en {t0-t1:.1f}s")
    
    variables_objetivo = [
        "swh", "t2m", "u10", "v10", "msl", "sst",
        "q_100", "q_250", "q_500", "q_850",
        "t_100", "t_250", "t_500", "t_850",
        "u_100", "u_250", "u_500", "u_850",
        "v_100", "v_250", "v_500", "v_850",
        "z_100", "z_250", "z_500", "z_850"
    ]

    # X, y = [], []
    # for i in range(len(df_scaled) - 24):
    #     input_seq = df_scaled.iloc[i:i+12].drop(columns=["valid_time", "latitude", "longitude"]).values
    #     output_seq = df_scaled.iloc[i+12:i+24][variables_objetivo].values
    #     X.append(input_seq)
    #     y.append(output_seq)
    
    X, y = [], []
    unique_coords = df[["latitude", "longitude"]].drop_duplicates()

    for _, row in unique_coords.iterrows():
        lat, lon = row["latitude"], row["longitude"]

        # Filtrar por ubicación
        df_coord = df[(df["latitude"] == lat) & (df["longitude"] == lon)].sort_values("valid_time")
        idxs = df_coord.index.tolist()

        # Seleccionar secuencias usando los índices
        x_seq = X_scaled[idxs]
        y_seq = y_scaled[idxs]

        # Crear ventanas deslizantes
        if len(x_seq) >= 24:
            for i in range(len(x_seq) - 24):
                X.append(x_seq[i:i+12])
                y.append(y_seq[i+12:i+24])


    X = np.array(X)
    y = np.array(y)

    print(f"📐 Dimensiones - X: {X.shape}, y: {y.shape}")
    
    
    model = build_lstm_model(input_shape=X.shape[1:], output_dim=y.shape[2])
    model.fit(X, y, epochs=50, batch_size=64, verbose=1, callbacks=get_callbacks())

    save_model_and_scalers(model, scalers, f"models_{region}")
    print(f"✅ Modelo para {region} guardado en models_{region}")


if __name__ == "__main__":
    entrenamiento_inicial(r'datasets\pacifico.parquet', region="pacifico")
    entrenamiento_inicial(r'datasets\atlantico.parquet', region="atlantico")