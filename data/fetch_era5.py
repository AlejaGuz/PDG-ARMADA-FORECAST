import cdsapi
from datetime import datetime, timedelta
import os
import pandas as pd
import re
import glob
from datetime import date
from fastapi import HTTPException
import xarray as xr
import logging
import time

from model.preprocess import clean_and_merge

c = cdsapi.Client()
# logger = logging.getLogger("uvicorn.error")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PRESSURE_NC = "pressure.nc"
SURFACE_NC = "surface.nc"
WAVE_NC     = "wave.nc"
STATIC_NC   = "surface_static.nc"

ATMOS_LEVELS = [100, 250, 500, 850]
ATMOS_VARS  = ['u','v','t','q','z']
SURF_VARS    = [
    "2m_temperature",         # t2m
    "10m_u_component_of_wind",# u10
    "10m_v_component_of_wind",# v10
    "mean_sea_level_pressure",# msl
    "sea_surface_temperature" # sst
]
STATIC_VARS  = ["land_sea_mask"] 
WAVE_VARS    = ["significant_height_of_combined_wind_waves_and_swell"]

SURF_KEYS   = ['t2m','u10','v10','msl','sst']
STATIC_KEYS = ['lsm']
WAVE_KEYS    = ["swh"] 
ATMOS_KEYS  = [f"{var}_{lvl}" for var in ATMOS_VARS for lvl in ATMOS_LEVELS]
WEATHER_KEYS = ['weather_event']

PACIFICO = [ 8, -82, -1, -75]
ATLANTICO = [12, -78,  8, -72]

AREA = PACIFICO  # Default area for Pacific region

DF_MERGE = None

META_KEYS   = ['latitude','longitude','valid_time']
FEATURE_KEYS = SURF_KEYS + WAVE_KEYS + ATMOS_KEYS + STATIC_KEYS + META_KEYS + WEATHER_KEYS

def retrieve_era5(product: str, params: dict, target: str):
    """
    Descarga ERA5; ante 400 extrae Última fecha/hora disponible y reintenta.
    """
    def call_api(p,t):
        return c.retrieve(product, p, t)

    try:
        call_api(params,target)
    except Exception as e:
        print(f"Error al descargar {target}: {e}")
        msg = str(e)
        m = re.search(r'latest date available for this dataset is:\s*([0-9\-]+)\s*([0-9]{2}):', msg)
        if not m:
            raise HTTPException(502, detail=f"ERA5 request failed: {msg}")
        
        date_ok_str, hour_ok = m.group(1), int(m.group(2))
        date_ok = datetime.strptime(date_ok_str, "%Y-%m-%d").date()
        
        print(f"date_ok: {date_ok}")
        print(f"date_ok_str: {date_ok_str}")

        n_days_prev = 2  # retroceder 2 días atrás
        fb_dates = [date_ok - timedelta(days=i) for i in range(1,n_days_prev + 1)]
        print(f"fb_dates: {fb_dates}")
        
        # 1) Llamada A: SOLO para date_ok, con horas [0..hour_ok] de 6 en 6
        fb_A = params.copy()
        fb_A['day'] = [date_ok.strftime("%d")]
        fb_A['time'] = [f"{h:02d}:00" for h in range(0, hour_ok+1, 6)]
        print(f"params: {fb_A}")
        tmp_A = target.replace(".nc", "_A.nc")
        print(f"tmp_A: {tmp_A}")
        try:
            call_api(fb_A,tmp_A)
            logger.info(f"✅ olas descarga parcial A")
        except Exception as e2:
            print(f"Error al descargar {target}: {e}")
            raise HTTPException(502, detail=f"ERA5 fallback (day OK) failed: {e2}")
        
        # 2) Llamada B: para los días anteriores,
        unique_year_month = sorted({(d.year, d.month) for d in fb_dates})
        day_strings = [d.strftime("%d") for d in fb_dates]
        year_strings = [str(y) for y, _ in unique_year_month]
        month_strings = [f"{m:02d}" for _, m in unique_year_month]

        fb_B = params.copy()
        fb_B['year']  = year_strings
        fb_B['month'] = month_strings
        fb_B['day']   = day_strings
        fb_B['time']  = [f"{h:02d}:00" for h in [0, 6, 12, 18]]

        tmp_B = target.replace(".nc", "_B.nc")

        try:
            print(f"Fallback B: días={fb_B['day']} horas={fb_B['time']}")
            call_api(fb_B, tmp_B)
            logger.info(f"✅ olas descarga parcial B")
        except Exception as e3:
            raise HTTPException(502, detail=f"ERA5 fallback (prev days) failed: {e3}")
        
        
        # 3) Fusiono tmp_A y tmp_B en 'target'

        import xarray as xr
        dsA = xr.open_dataset(tmp_A)
        dsB = xr.open_dataset(tmp_B)
        
        ds_comb = xr.concat([dsA, dsB], dim="valid_time")
        print(f"ds_comb: {ds_comb}")
        
        # Guardar resultado en 'target'
        ds_comb.to_netcdf(target)
        dsA.close(); dsB.close()
        os.remove(tmp_A); os.remove(tmp_B)

        if not os.path.exists(target):
            raise HTTPException(204, detail=f"No file generated: {target}")

        return target
        
def load_dataset(ncfile: str, time_col: str = 'time') -> pd.DataFrame:
    try:
        ds = xr.open_dataset(ncfile)
        print(ds)            
        print(ds.dims)       
        print(ds.sizes) 
        df = ds.to_dataframe().reset_index()
    except FileNotFoundError:
        raise HTTPException(500, detail=f"File not found: {ncfile}")
    except Exception as e:
        raise HTTPException(500, detail=f"Error reading {ncfile}: {e}")
    if time_col != 'valid_time':
        df = df.rename(columns={time_col: 'valid_time'})
    return df

import pandas as pd
from urllib.request import Request, urlopen
from io import StringIO

def descargar_oni_psl_urllib():
    url = 'https://psl.noaa.gov/data/correlation/oni.data'

    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://psl.noaa.gov/'
    }

    req = Request(url, headers=headers)
    
    try:
        response = urlopen(req)
        raw_data = response.read().decode('utf-8')
    except Exception as e:
        raise Exception(f"❌ Error al descargar: {e}")

    # Procesar líneas
    data = raw_data.splitlines()
    lines = [line for line in data if line and not line.startswith("Year")]

    records = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 13:  # Año + 12 meses
            year = int(parts[0])
            for i, val in enumerate(parts[1:], 1):
                try:
                    oni = float(val)
                except ValueError:
                    oni = None
                records.append({'Año': year, 'Mes': i, 'ONI': oni})

    df = pd.DataFrame(records)
    # Renombrar columnas como espera pandas
    df = df.rename(columns={'Año': 'year', 'Mes': 'month'})

    # Agregar día fijo y convertir a datetime
    df['Fecha'] = pd.to_datetime(df[['year', 'month']].assign(day=15))

    def clasificar_enso(valor):
        if valor is None:
            return None
        elif valor >= 0.5:
            return "El Niño"
        elif valor <= -0.5:
            return "La Niña"
        else:
            return "Neutral"

    df['Fase_ENSO'] = df['ONI'].apply(clasificar_enso)
    return df[['Fecha', 'ONI', 'Fase_ENSO']]


def load_era5_once(region: str = 'pacifico'):
    print(f"Loading ERA5 data for region: {region}")
    if region == 'pacifico':
        AREA = PACIFICO
    else:
        AREA = ATLANTICO
    
    today = date.today() #.strftime('%Y-%m-%d')
    dates = [today - timedelta(days=i) for i in range(3)]
    year = [dates[0].strftime("%Y")]
    month = [dates[0].strftime("%m")]
    days   = [d.strftime('%d') for d in dates]
    times = [f"{h:02d}:00" for h in range(0, 24, 6)]
    
    t0 = time.time()
    logger.info("📥 Inicio descarga ERA5")
    # 1) Pressure levels
    t1 = time.time()
    if not os.path.exists(f"datasets/{region}_" + PRESSURE_NC):
        params_pl = {
            'product_type':'reanalysis','format':'netcdf',
            'year':  year,
            'month': month,
            'day':   days,
            'time': times,
            'area':  AREA,
            'pressure_level':ATMOS_LEVELS,'variable':ATMOS_VARS
        }
        retrieve_era5('reanalysis-era5-pressure-levels', params_pl, f"datasets/{region}_" + PRESSURE_NC)
    logger.info(f"✅ Presión descargada en {time.time()-t1:.1f}s")
    
    # 2) Dynamic surface variables
    t2 = time.time()
    if not os.path.exists(f"datasets/{region}_" + SURFACE_NC):
        params_sf = {
            'product_type':'reanalysis','format':'netcdf',
            'year':  year,
            'month': month,
            'day':   days,
            'time':times,
            'area':  AREA,
            'variable':SURF_VARS
        }
        print("SURFACE_NC")
        retrieve_era5('reanalysis-era5-single-levels', params_sf, f"datasets/{region}_" + SURFACE_NC)
    logger.info(f"✅ Superficie dinámica en {time.time()-t2:.1f}s")
    
    # 3) Static surface variables (lsm) for first day of month
    t3 = time.time()
    if not os.path.exists(f"datasets/{region}_" + STATIC_NC):
        static_date = date.today().replace(day=1).strftime('%Y-%m-%d')
        print(f"static_date: {static_date}")
        params_static = {
            'product_type':'reanalysis','format':'netcdf','date':static_date,
            'time':['00:00'],
            'area':  AREA,
            'variable':STATIC_VARS
        }
        retrieve_era5('reanalysis-era5-single-levels', params_static, f"datasets/{region}_" + STATIC_NC)
    logger.info(f"✅ Superficie estática en {time.time()-t3:.1f}s")
    
    # 4) Wave variables
    t4 = time.time()
    if not os.path.exists(f"datasets/{region}_" + WAVE_NC):
        params_wv = {
            'product_type':'reanalysis','format':'netcdf',
            'year':  year,
            'month': month,
            'day':   days,
            'time':times,
            'area':  AREA,
            'variable':WAVE_VARS
        }
        retrieve_era5('reanalysis-era5-single-levels', params_wv, f"datasets/{region}_" + WAVE_NC)
    logger.info(f"✅ Olas en {time.time()-t4:.1f}s")
    
    # Load dataframes
    print("load PRESSURE_NC")
    t5 = time.time()
    df_pl     = load_dataset(f"datasets/{region}_" + PRESSURE_NC,'time')
    logger.info(f"✅ PRESSURE_NC en {time.time()-t5:.1f}s")
    
    print("load SURFACE_NC")
    t6 = time.time()
    df_sf     = load_dataset(f"datasets/{region}_" + SURFACE_NC,'time')
    logger.info(f"✅ SURFACE_NC en {time.time()-t6:.1f}s")
    
    print("load STATIC_NC")
    t7 = time.time()
    df_static = load_dataset(f"datasets/{region}_" + STATIC_NC,'time')
    print(f"df_static info: {df_static.info()}")
    logger.info(f"✅ STATIC_NC en {time.time()-t7:.1f}s")
    
    print("load WAVE_NC")
    t8 = time.time()
    df_wv     = load_dataset(f"datasets/{region}_" + WAVE_NC,'time')
    logger.info(f"✅ WAVE_NC en {time.time()-t8:.1f}s")
    
    logger.info(f"🚀 Procesamiento completo en {time.time()-t0:.1f}s")

    # Prepare static: drop time and valid_time
    df_static = df_static.drop(columns=['time','valid_time'], errors='ignore')
    
    # Pivot pressure-levels
    df_pl['pressure_level'] = df_pl['pressure_level'].astype(int)
    df_pl.rename(columns={'pressure_level':'level'}, inplace=True)
    pivot = df_pl.pivot_table(
        index=['valid_time','latitude','longitude'],
        columns='level', values=ATMOS_VARS
    )
    pivot.columns = [f"{v}_{lvl}" for v,lvl in pivot.columns]
    pivot = pivot.reset_index()

    ESSENTIAL = ['valid_time','latitude','longitude']
    
    cols_keep = ESSENTIAL + SURF_KEYS
    print(f"df_sf cols_keep: {cols_keep}")
    df_sf = df_sf[cols_keep]
    print(f"df_sf: {df_sf.info()}")
    
    cols_keep = ESSENTIAL + WAVE_KEYS
    print(f"df_wv cols_keep: {cols_keep}")
    df_wv = df_wv[cols_keep]
    print(f"df_wv: {df_wv.info()}")
    
    cols_keep = ['latitude','longitude'] + STATIC_KEYS
    df_static = df_static[cols_keep]
    print(f"df_static: {df_static.info()}")
    
    
    # print(f"lms: {df_static['lsm'].unique()}")
    # umbral deÃ± ocÃ©ano (0=todo agua, 1=toda tierra)
    ocean_thresh = 0.5   #50% agua

    # 1) filtrar df_static por fracción de tierra menor al umbral
    df_ocean = df_static[
        df_static['lsm'] < ocean_thresh
    ].drop_duplicates()
    print(f"df_ocean: {df_ocean.info()}")
    print(f"lms df_ocean: {df_ocean['lsm'].unique()}")
    
    print(df_wv['latitude'].unique())
    print("----------------------------------------")
    print(df_ocean['latitude'].unique())
    
    print("antes del primer merge:", df_sf.isnull().sum())
    merged = pd.merge(df_sf, df_ocean, on=['latitude','longitude'], how='inner')
    print("después del primer merge:", merged.isnull().sum())
    print(f"len df_sf: {len(df_sf)} - len df_ocean: {len(df_ocean)}")
    print(f"len merged: {len(merged)}")
    
    df_oni = descargar_oni_psl_urllib()
    
    ult_mes_oni = df_oni['Fecha'].max().date()
    min_mes_merge = merged['valid_time'].min().date()
    
    merged['weather_event'] = 0
    merged = clean_and_merge([merged, df_wv, pivot])
    # merged = pd.merge(merged, df_wv, on=ESSENTIAL, how='inner')
    # print("segundo merge:", merged.isnull().sum())
    # merged = pd.merge(merged, pivot, on=ESSENTIAL, how='inner')
    print("último merge:", merged.isnull().sum())
    
    return merged
