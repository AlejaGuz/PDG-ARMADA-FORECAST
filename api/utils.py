from model.predictor import predict_from_latest
def get_forecast(region, model=None, scalers=None):
    return predict_from_latest(region, model, scalers)