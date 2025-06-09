?? Marine Forecast LSTM
Predicci�n meteorol�gica mar�tima basada en datos del conjunto ERA5 usando redes LSTM. Este proyecto ofrece una API REST para consultar predicciones de variables meteorol�gicas a partir de datos hist�ricos, y reentrena autom�ticamente el modelo cada 3 d�as con los datos m�s recientes.

?? Estructura del Proyecto
bash
Copiar
Editar
marine_forecast/
??? api/                  # API REST con FastAPI
??? data/                 # Scripts para extracci�n de datos ERA5
??? model/                # Preprocesamiento, entrenamiento y predicci�n
??? retrain/              # Reentrenamiento programado
??? visual/               # Visualizaci�n de resultados (opcional)
??? docker/               # Dockerfile y dependencias
??? models/               # Modelos entrenados y escaladores
??? .env                  # Variables de entorno
??? docker-compose.yml    # Orquestaci�n de contenedores
??? README.md             # Este archivo

?? �Qu� hace este proyecto?
Obtiene los datos ERA5 de los �ltimos 3 d�as para la regi�n del Pac�fico o el Atl�ntico.

Preprocesa los datos con interpolaci�n y manejo de nulos.

Predice los pr�ximos 3 d�as (12 pasos de 6h) con un modelo LSTM.

Expone los resultados a trav�s de una API REST.

Se reentrena autom�ticamente cada 3 d�as con datos nuevos.

?? Requisitos
Docker

Cuenta y API Key de CDSAPI

?? Instalaci�n y despliegue
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

La API estar� disponible en: http://localhost:8000/forecast/?region=pacifico

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

?? Reentrenamiento autom�tico
El archivo cron_retrain.sh se ejecuta cada 3 d�as dentro del contenedor para:

Consultar los datos m�s recientes disponibles.

Fusionar e interpolar los datos.

Entrenar un nuevo modelo LSTM.

Guardar el modelo .h5 y los scalers.pkl.

?? M�tricas
Se calcula el MAE por cada variable en cada reentrenamiento. Estas m�tricas pueden almacenarse en un archivo CSV para monitoreo hist�rico.

?? Variables soportadas
Entrada:
swh, t2m, u10, v10, msl, sst, lsm, q_*, t_*, u_*, v_*, z_*, weather_event
Salida:
Todas excepto weather_event, latitude, longitude, valid_time y lsm

?? Pruebas y visualizaci�n
Puedes visualizar la predicci�n con scripts en visual/plot.py usando matplotlib, cartopy o plotly.

?? Roadmap futuro
Visualizador web embebido con gr�ficos interactivos.

Dashboard de monitoreo del desempe�o del modelo.

Validaci�n cruzada geogr�fica.

Control de versiones de datasets.
