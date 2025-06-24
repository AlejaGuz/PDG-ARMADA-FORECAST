import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import pydeck as pdk

st.title("🌊 Visualizador de Predicción Marina")

# Parámetros para enviar a FastAPI
region = st.selectbox("Selecciona región", ["pacifico", "atlantico"])

if st.button("📥 Cargar predicción"):
    with st.spinner("Solicitando datos..."):
        try:
            response = requests.get(f"http://127.0.0.1:8000/forecast/{region}")
            response.raise_for_status()
            data = response.json()
            st.session_state[f"df_pred_{region}"] = pd.DataFrame(data)
            st.success("✅ Predicción cargada correctamente.")
        except Exception as e:
            st.error(f"❌ Error al obtener la predicción: {e}")

# --- Verifica si hay predicción cargada
if f"df_pred_{region}" in st.session_state:
    df_pred = st.session_state[f"df_pred_{region}"]

    # --- Selector de variable a graficar
    var = st.selectbox("🌐 Selecciona la variable a graficar", df_pred.columns.drop(["latitude", "longitude", "valid_time"]))

    # --- Gráfico interactivo
    st.subheader(f"📊 Evolución temporal de {var}")
        # Seleccionar solo la latitud
    latitudes_unicas = df_pred["latitude"].drop_duplicates().sort_values().reset_index(drop=True)
    lat = st.selectbox("Selecciona la latitud", latitudes_unicas)

    # Filtrar todos los puntos para esa latitud
    df_lat = df_pred[df_pred["latitude"] == lat]

    # Pivotar para mostrar una línea por cada longitud
    df_plot = df_lat.pivot(index="valid_time", columns="longitude", values=var)
    df_plot = df_plot.sort_index()

    # Mostrar el gráfico con todas las longitudes para esa latitud
    st.line_chart(df_plot)
    # selected_coords = df_pred[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    # coord = st.selectbox("Selecciona la ubicación", selected_coords.index, format_func=lambda i: f"{selected_coords.loc[i, 'latitude']}, {selected_coords.loc[i, 'longitude']}")

    # lat, lon = selected_coords.loc[coord, "latitude"], selected_coords.loc[coord, "longitude"]
    # df_plot = df_pred[(df_pred.latitude == lat) & (df_pred.longitude == lon)].sort_values("valid_time")

    # st.line_chart(df_plot.set_index("valid_time")[var])
    

    # Filtra la variable, por ejemplo al primer timestep
    df_pred["valid_time"] = pd.to_datetime(df_pred["valid_time"])
    df_mapa = df_pred[df_pred["valid_time"] == df_pred["valid_time"].min()]
    
    # Selecciona variable
    # var = st.selectbox(" Variable a visualizar en mapa", df_mapa.columns.drop(["latitude", "longitude", "valid_time"]))

    st.subheader(f"🗺️ Mapa para {var} en {region} - {df_mapa['valid_time'].min().strftime('%Y-%m-%d %H:%M')}")

    # Escala de colores personalizada (opcional)
    COLOR_SCALE = [
        [0, 255, 0],     # verde para valores bajos
        [255, 255, 0],   # amarillo intermedio
        [255, 0, 0],     # rojo para altos
    ]

    # Normaliza valores
    min_val, max_val = df_mapa[var].min(), df_mapa[var].max()
    df_mapa["color"] = df_mapa[var].apply(lambda x: int(255 * (x - min_val) / (max_val - min_val)))
    df_mapa["color_rgb"] = df_mapa["color"].apply(lambda c: [c, 255 - c, 100])

    # Capa de puntos
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_mapa,
        get_position='[longitude, latitude]',
        get_radius=15000,
        get_fill_color="color_rgb",
        pickable=True,
    )

    # View
    view_state = pdk.ViewState(
        latitude=df_mapa.latitude.mean(),
        longitude=df_mapa.longitude.mean(),
        zoom=4,
        pitch=0,
    )

    # Mostrar
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": f"{var}: {{ {var} }}" }))

else:
    st.info("👈 Carga primero una predicción para visualizar los datos.")

# if st.button("Obtener Predicción"):
#     with st.spinner("Solicitando datos..."):
#         response = requests.get(
#             f"http://127.0.0.1:8000/forecast/{region}",
#         )
#         if response.status_code == 200:
#             data = response.json()
#             df = pd.DataFrame(data)

#             st.success("✅ Predicción obtenida")
#             variable = st.selectbox("Variable a visualizar", [col for col in df.columns if col not in ['valid_time', 'latitude', 'longitude']])
#             lat = st.selectbox("Latitud", sorted(df['latitude'].unique()))
            
#             df_filtered = df[df['latitude'] == lat]
#             fig = px.line(df_filtered, x="valid_time", y=variable, title=f"{variable} para latitud {lat}")
#             st.plotly_chart(fig)
#         else:
#             st.error("❌ Error al obtener datos")
