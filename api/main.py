from fastapi import FastAPI
from api.utils import get_forecast
from model.utils import load_model_and_scalers

app = FastAPI()

@app.on_event("startup")
def load_models_on_startup():
    print(f"🔮 descargando Modelo")
    model_pacifico, scaler_pacifico = load_model_and_scalers("models_pacifico")
    model_atlantico, scaler_atlantico = load_model_and_scalers("models_atlantico")
    print(f"📦 Modelo pacífico descargado. {model_pacifico}")
    print(f"📦 Modelo atlantico descargado. {model_atlantico}")
    
    print(f"📦 Tipo: {type(scaler_atlantico)}")
    print(f"📦 Antes de asignar: {scaler_atlantico}")
    
    app.state.models = {
        "pacifico": {"model": model_pacifico, "scaler": scaler_pacifico},
        "atlantico": {"model": model_atlantico, "scaler": scaler_atlantico}
    }
    print(f"📦 Después de asignar: {app.state.models['atlantico']['scaler']}")

@app.get("/forecast/{region}")
def forecast(region: str = "pacifico"):
    model = app.state.models[region]["model"]
    scaler = app.state.models[region]["scaler"]
    print(f"📦 scaler region. {scaler}")
    return get_forecast(region,model, scaler)
