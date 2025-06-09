import numpy as np

def evaluate_model(model, X_test, y_test, scalers, target_columns):
    
    y_pred = model.predict(X_test)
    
    y_pred_inv = scalers['global'].inverse_transform(y_pred)
    y_test_inv = scalers['global'].inverse_transform(y_test)

    for i, var in enumerate(target_columns):
        mae = np.mean(np.abs(y_pred_inv[:, i] - y_test_inv[:, i]))
        print(f"{var} MAE: {mae:.4f}")