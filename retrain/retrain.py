from data.fetch_era5 import load_era5_once
from model.preprocess import preprocess, clean_and_merge
from model.trainer import build_lstm_model, get_callbacks
from model.utils import save_model_and_scalers
import numpy as np

def scheduled_retrain(region="pacifico"):
    print(f"?? Iniciando reentrenamiento automático para {region}")

    variables = [
        "swh", "t2m", "u10", "v10", "msl", "sst",
        "q_100", "q_250", "q_500", "q_850",
        "t_100", "t_250", "t_500", "t_850",
        "u_100", "u_250", "u_500", "u_850",
        "v_100", "v_250", "v_500", "v_850",
        "z_100", "z_250", "z_500", "z_850"
    ]

    # latest_date = get_latest_date_from_api(variables[0])
    # dfs = [fetch_group(var, [var], region, latest_date) for var in variables]
    # df = clean_and_merge(dfs)
    df = load_era5_once(region)
    df_scaled, scalers = preprocess(df, fit_scaler=True)

    X, y = [], []
    for i in range(len(df_scaled) - 24):
        input_seq = df_scaled.iloc[i:i+12].drop(columns=["valid_time", "latitude", "longitude"]).values
        output_seq = df_scaled.iloc[i+12:i+24][variables].values
        X.append(input_seq)
        y.append(output_seq)

    X = np.array(X)
    y = np.array(y)

    print(f"?? Dimensiones - X: {X.shape}, y: {y.shape}")
    model = build_lstm_model(input_shape=X.shape[1:], output_dim=y.shape[2])
    model.fit(X, y, epochs=30, batch_size=64, verbose=1, callbacks=get_callbacks())

    save_model_and_scalers(model, scalers, f"models_{region}")
    print(f"? Reentrenamiento completado y modelo actualizado para {region}")


if __name__ == "__main__":
    scheduled_retrain("pacifico")
    scheduled_retrain("atlantico")
