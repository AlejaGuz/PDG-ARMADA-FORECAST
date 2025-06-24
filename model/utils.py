import os
import pickle

def save_model_and_scalers(model, scalers, path):
    os.makedirs(path, exist_ok=True)  # Crear la carpeta si no existe
    model.save(f"{path}/lstm3_model.h5")
    with open(f"{path}/scalers3.pkl", "wb") as f:
        pickle.dump(scalers, f)  # Guardar como {'x': x_scaler, 'y': y_scaler}

def load_model_and_scalers(path):
    from tensorflow.keras.models import load_model
    model = load_model(f"{path}/lstm_model.h5")
    with open(f"{path}/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    return model, scalers 

