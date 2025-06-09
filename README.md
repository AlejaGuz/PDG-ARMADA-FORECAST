?? Marine Forecast LSTM
Predicción meteorológica marítima basada en datos del conjunto ERA5 usando redes LSTM. Este proyecto ofrece una API REST para consultar predicciones de variables meteorológicas a partir de datos históricos, y reentrena automáticamente el modelo cada 3 días con los datos más recientes.

?? Estructura del Proyecto
bash
Copiar
Editar
marine_forecast/
??? api/                  # API REST con FastAPI
??? data/                 # Scripts para extracción de datos ERA5
??? model/                # Preprocesamiento, entrenamiento y predicción
??? retrain/              # Reentrenamiento programado
??? visual/               # Visualización de resultados (opcional)
??? docker/               # Dockerfile y dependencias
??? models/               # Modelos entrenados y escaladores
??? .env                  # Variables de entorno
??? docker-compose.yml    # Orquestación de contenedores
??? README.md             # Este archivo

?? ¿Qué hace este proyecto?
Obtiene los datos ERA5 de los últimos 3 días para la región del Pacífico o el Atlántico.

Preprocesa los datos con interpolación y manejo de nulos.

Predice los próximos 3 días (12 pasos de 6h) con un modelo LSTM.

Expone los resultados a través de una API REST.

Se reentrena automáticamente cada 3 días con datos nuevos.

?? Requisitos
Docker

Cuenta y API Key de CDSAPI

?? Instalación y despliegue
1. Clonar el repositorio
git clone https://github.com/tu_usuario/marine_forecast.git
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

La API estará disponible en: http://localhost:8000/forecast/?region=pacifico

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

?? Reentrenamiento automático
El archivo cron_retrain.sh se ejecuta cada 3 días dentro del contenedor para:

Consultar los datos más recientes disponibles.

Fusionar e interpolar los datos.

Entrenar un nuevo modelo LSTM.

Guardar el modelo .h5 y los scalers.pkl.

?? Métricas
Se calcula el MAE por cada variable en cada reentrenamiento. Estas métricas pueden almacenarse en un archivo CSV para monitoreo histórico.

?? Variables soportadas
Entrada:
swh, t2m, u10, v10, msl, sst, lsm, q_*, t_*, u_*, v_*, z_*, weather_event
Salida:
Todas excepto weather_event, latitude, longitude, valid_time y lsm

?? Pruebas y visualización
Puedes visualizar la predicción con scripts en visual/plot.py usando matplotlib, cartopy o plotly.

?? Roadmap futuro
Visualizador web embebido con gráficos interactivos.

Dashboard de monitoreo del desempeño del modelo.

Validación cruzada geográfica.

Control de versiones de datasets.
