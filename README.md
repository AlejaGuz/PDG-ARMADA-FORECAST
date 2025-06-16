?? Marine Forecast LSTM
PredicciÃ³n meteorolÃ³gica marÃ­tima basada en datos del conjunto ERA5 usando redes LSTM. Este proyecto ofrece una API REST para consultar predicciones de variables meteorolÃ³gicas a partir de datos histÃ³ricos, y reentrena automÃ¡ticamente el modelo cada 3 dÃ­as con los datos mÃ¡s recientes.

?? Estructura del Proyecto
marine_forecast/
??? api/                  # API REST con FastAPI
??? data/                 # Scripts para extracciÃ³n de datos ERA5
??? model/                # Preprocesamiento, entrenamiento y predicciÃ³n
??? retrain/              # Reentrenamiento programado
??? visual/               # VisualizaciÃ³n de resultados (opcional)
??? docker/               # Dockerfile y dependencias
??? models/               # Modelos entrenados y escaladores
??? .env                  # Variables de entorno
??? docker-compose.yml    # OrquestaciÃ³n de contenedores
??? README.md             # Este archivo

?? Â¿QuÃ© hace este proyecto?
Obtiene los datos ERA5 de los Ãºltimos 3 dÃ­as para la regiÃ³n del PacÃ­fico o el AtlÃ¡ntico.

Preprocesa los datos con interpolaciÃ³n y manejo de nulos.

Predice los prÃ³ximos 3 dÃ­as (12 pasos de 6h) con un modelo LSTM.

Expone los resultados a travÃ©s de una API REST.

Se reentrena automÃ¡ticamente cada 3 dÃ­as con datos nuevos.

?? Requisitos
Docker

Cuenta y API Key de CDSAPI

?? InstalaciÃ³n y despliegue
1. Clonar el repositorio
git clone [https://github.com/tu_usuario/marine_forecast.git](https://github.com/AlejaGuz/PDG-ARMADA-FORECAST.git)
cd marine_forecast

2. Crear archivo .cdsapirc
En tu carpeta de usuario:
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: tu_user_id:tu_api_key

3. Construir el contenedor
docker-compose build

4. Ejecutar la API
docker-compose up

La API estarÃ¡ disponible en: http://localhost:8000/forecast/?region=pacifico

?? Ejemplo de uso del endpoint

GET /forecast/?region=pacifico

Response:
{
  "region": "pacifico",
  "forecast": {
    "swh": [...],
    "t2m": [...],
    ...
  }
}

?? Reentrenamiento automÃ¡tico
El archivo cron_retrain.sh se ejecuta cada 3 dÃ­as dentro del contenedor para:

Consultar los datos mÃ¡s recientes disponibles.

Fusionar e interpolar los datos.

Entrenar un nuevo modelo LSTM.

Guardar el modelo .h5 y los scalers.pkl.

?? MÃ©tricas
Se calcula el MAE por cada variable en cada reentrenamiento. Estas mÃ©tricas pueden almacenarse en un archivo CSV para monitoreo histÃ³rico.

?? Variables soportadas
Entrada:
swh, t2m, u10, v10, msl, sst, lsm, q_*, t_*, u_*, v_*, z_*, weather_event
Salida:
Todas excepto weather_event, latitude, longitude, valid_time y lsm

?? Pruebas y visualizaciÃ³n
Puedes visualizar la predicciÃ³n con scripts en visual/plot.py usando matplotlib, cartopy o plotly.

?? Roadmap futuro
Visualizador web embebido con grÃ¡ficos interactivos.

Dashboard de monitoreo del desempeÃ±o del modelo.

ValidaciÃ³n cruzada geogrÃ¡fica.

Control de versiones de datasets.

Para ejecutar el Backend se puede usar el siguiente comando por terminal:
uvicorn api.main:app --reload

Para ejecutar el Frontend se puede ejecutar el siguiente comando por terminal:
streamlit run dashboard/app.py

