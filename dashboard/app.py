import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.title("🌊 Visualizador de Predicción Marina")

# Parámetros para enviar a FastAPI
region = st.selectbox("Selecciona región", ["pacifico", "atlantico"])

if st.button("Obtener Predicción"):
    with st.spinner("Solicitando datos..."):
        response = requests.get(
            f"http://127.0.0.1:8000/forecast/{region}",
        )
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)

            st.success("✅ Predicción obtenida")
            variable = st.selectbox("Variable a visualizar", [col for col in df.columns if col not in ['valid_time', 'latitude', 'longitude']])
            lat = st.selectbox("Latitud", sorted(df['latitude'].unique()))
            
            df_filtered = df[df['latitude'] == lat]
            fig = px.line(df_filtered, x="valid_time", y=variable, title=f"{variable} para latitud {lat}")
            st.plotly_chart(fig)
        else:
            st.error("❌ Error al obtener datos")
