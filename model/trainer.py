from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_lstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(RepeatVector(12))  # ?? Repite la salida para 12 pasos futuros
    model.add(LSTM(128, return_sequences=True))  # ?? LSTM que devuelve secuencia
    model.add(TimeDistributed(Dense(output_dim)))  # ? Predicción por paso temporal
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def get_callbacks():
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
    return [early_stopping, reduce_lr]
