# -*- coding: utf-8 -*-
"""EMH_Predicción_número_especies_coleoptera_antioquia.py


Autor: Esteban Marentes Herrera
Enlace GitHub https://github.com/EstebanMH-SiB/modelPredictColeopteraSpecies

# Modelo de predicción para el número de especies de Coleoptera en el Departamento de Antioquia


"""

# Importa las librerías necesarias
import os 
import rasterstats
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import tensorflow as tf
import keras
import rasterio
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ndcg_score
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Rescaling
from keras.utils import plot_model
from keras.optimizers import Adam
from rasterio import mask
from rasterio.transform import from_origin
import dask.dataframe as dd


# Se define el directorio de trabajo

os.chdir("/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/TercerSemestre/TesisFinal/Datos")

# Se cargan los datos de coleoptera del mundo en chunks de 100000 datos

chunks_list = []
chunk_size = 1000000

# Iterar sobre los chunks
for chunk in pd.read_csv('coleoptera_completa_gbif.csv',  encoding = "utf8", sep="\t", chunksize=chunk_size, on_bad_lines='skip'):
    chunks_list.append(chunk)
coleoptera_completa_mundo = pd.concat(chunks_list, ignore_index=True) # concatenar los chunks juntos

# Quitar las columnas que no tienen nada que ver con lo biológico

coleoptera_completa_mundo_especie = coleoptera_completa_mundo.drop(columns=['datasetKey', 'publishingOrgKey', 'day', 'month', 'year', 'taxonKey', 'institutionCode', 'catalogNumber', 'recordNumber', 'rightsHolder', 'typeStatus', 'establishmentMeans', 'lastInterpreted', 'mediaType'])

# Quitar los datos sin coordenadas

coleoptera_completa_mundo_especie = coleoptera_completa_mundo_especie.dropna(subset=['decimalLatitude', 'decimalLongitude'])


# Crear la columna tipo punto usando geopandas para hacer el cruce con las capas vectoriales

coleoptera_completa_mundo_especie[['decimalLatitude', 'decimalLongitude']] = coleoptera_completa_mundo_especie[['decimalLatitude','decimalLongitude']].fillna(value=0)
coleoptera_completa_mundo_especie['Coordinates'] = list(zip(coleoptera_completa_mundo_especie.decimalLongitude, coleoptera_completa_mundo_especie.decimalLatitude))
coleoptera_completa_mundo_especie['Coordinates'] = coleoptera_completa_mundo_especie['Coordinates'].apply(Point)
coleoptera_completa_mundo_especie = gpd.GeoDataFrame(coleoptera_completa_mundo_especie, geometry='Coordinates')
coleoptera_completa_mundo_especie.crs = {'init' :'epsg:4326'}


#Realizar cruces con la capa de Colombia

departamento =gpd.read_file("departamento/MGN_DPTO_POLITICO.shp", encoding = "utf8") # cargar el archivo shapefile de los municipios de Colombia

coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie, departamento, how="inner", op="intersects")
# coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia[coleoptera_completa_mundo_especie_colombia.geometry.within(departamento.unary_union)]

#Quitar las columnas extras creadas con el cruce y cambiar el nombre por departamento
coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'DPTO_CCDGO', 'DPTO_NANO_', 'DPTO_CACTO', 'DPTO_NANO','Shape_Leng', 'Shape_Area', 'DPTO_CNMBR'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'stateProv': 'departamento'}, inplace=True)

# Eliminar las columnas completamente vacías
coleoptera_completa_mundo_especie_colombia.dropna(axis=1, how='all', inplace=True)

#### Cargar los archivos con información adicional


etiquetas_Coleoptera_Colombia = pd.read_csv('EtiquetasColeopteraColombia.txt', encoding = "utf8", sep="\t")

plantae_colombia = pd.read_csv('PlantaeColombiaCompleto/verbatim.txt', encoding = "utf8", sep="\t", usecols=['gbifID', 'occurrenceID',  'eventDate',  'country', 'stateProvince', 'county', 'municipality', 'locality','decimalLatitude', 'decimalLongitude', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'taxonID', 'scientificName', 'genus', 'specificEpithet', 'infraspecificEpithet', 'taxonRank', 'scientificNameAuthorship'])

Radiacion_solar_global_promedio_multianual =gpd.read_file("Radiacion_solar_global_promedio_multianual/Shape/GDBIDEAM.CN2_Rdcn_Solar_Global_ProAnual.shp", encoding = "utf8")

Humedad_Relativa_Anual_Promedio_Multianual_1981_2010 =gpd.read_file("Humedad_Relativa_Anual_Promedio_Multianual_1981_2010/SHP/ACC2014_HmRl_AA_MA_1981_2010.shp", encoding = "utf8")

Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMx_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMd_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMn_MM_1981_2010_01.shp", encoding = "utf8")

Velocidad_viento_10_mtrs_altura_Mensual_2000_2010 =gpd.read_file("Velocidad_viento_10_mtrs_altura_Mensual_2000_2010/SHP/AVC2014_VlVn_10m_MM_2000_2010_01.shp", encoding = "utf8")

with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/TercerSemestre/TesisFinal/Datos/astergdem2.tif') as src:
        astergdem2 = src.read(1)

with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/TercerSemestre/TesisFinal/Datos/Escenario_Precipitacion_1976_2005/ECC_Prcp_GeoTiff_2011_2040/ECC_Prcp_1976_2005_100K_2015.tif') as src:
        Escenario_Precipitacion_1976_2005 = src.read(1)


### Realizar los cruces con las distintas capas

#Realizar cruces Geo radiacion
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Radiacion_solar_global_promedio_multianual, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'OBJECTID', 'ID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Radiacion_solar_global_promedio_multianual'}, inplace=True)

#Realizar cruces Geo Humedad relativa
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Humedad_Relativa_Anual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'OBJECTID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura maxima
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)

#Realizar cruces Geo Temperatura media
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Temperatura_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE_right','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura media
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura media
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)


#Realizar cruces Velocidad viento
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE','GRIDCODE_left','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)


#Realizar cruces DEM elevación
ruta_dem = r"/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/TercerSemestre/TesisFinal/Datos/astergdem2.tif"
with rasterio.open(ruta_dem) as src:
    coords = [(x, y) for x, y in zip(coleoptera_completa_mundo_especie_colombia.geometry.x, coleoptera_completa_mundo_especie_colombia.geometry.y)]
    valores = [v[0] for v in src.sample(coords, indexes=1)]

coleoptera_completa_mundo_especie_colombia["elevacion_dem"] = valores

#Realizar cruces raster de Precipitación
ruta_precipitacion = r"/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/TercerSemestre/TesisFinal/Datos/Escenario_Precipitacion_1976_2005/ECC_Prcp_GeoTiff_2011_2040/ECC_Prcp_1976_2005_100K_2015.tif"
with rasterio.open(ruta_precipitacion) as src:
    coords = [(x, y) for x, y in zip(coleoptera_completa_mundo_especie_colombia.geometry.x, coleoptera_completa_mundo_especie_colombia.geometry.y)]
    valores = [v[0] for v in src.sample(coords, indexes=1)]

coleoptera_completa_mundo_especie_colombia["ECC_Prcp_1976_2005_100K_2015"] = valores



# Etiquetar con los datos con la información de las familias

etiquetas_Coleoptera_Colombia = etiquetas_Coleoptera_Colombia[['family','NumeroEspeciesFamilia']] #Quedarse solo con la columna del número de registros
coleoptera_completa_mundo_especie_colombia = pd.merge(coleoptera_completa_mundo_especie_colombia,etiquetas_Coleoptera_Colombia,how='left',on='family')


# Exportar los datos ya ajustados
coleoptera_completa_mundo_especie_colombia.to_csv( 'coleoptera_completa_mundo_especie_colombia'+'.txt', sep="\t", encoding = "utf8")

# Cargar solamente los datos ajustados
# etiquetas_Coleoptera_Colombia = pd.read_csv('coleoptera_completa_mundo_especie_colombia.txt', encoding = "utf8", sep="\t")


''' Dejar solamente los datos de especie en caso de que sea necesario
coleoptera_completa_mundo['taxonRank']=coleoptera_completa_mundo['taxonRank'].str.title()
coleoptera_completa_mundo['taxonRank']=coleoptera_completa_mundo['taxonRank'].replace('Specie', 'Especie').replace('Espécie', 'Especie').replace('Epecie', 'Especie').replace('Especies', 'Especie').replace('Species', 'Especie').replace('SUBSPECIES', 'Especie').replace('VARIETY', 'Especie')
coleoptera_completa_mundo_especie=coleoptera_completa_mundo.loc[coleoptera_completa_mundo['taxonRank'] == 'Especie' ]
coleoptera_completa_mundo_especie.dropna(axis=1, how='all', inplace=True)

'''

### hacer un plot del mapa de Colombia par ver los datos

# corret todo el bloque junto
fig, ax = plt.subplots(figsize=(10, 10))
departamento.plot(ax=ax, color='lightblue', edgecolor='black', label="Departamento")
coleoptera_completa_mundo_especie_colombia.plot(ax=ax, color='red', marker='o', markersize=5, label="Registros de especies de Coleoptera")
ax.set_title("Mapa de Colombia con registros de Coleoptera")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.legend()
plt.savefig('mapaColeoptera.png', dpi=300, bbox_inches='tight')
plt.show()


### generar descriptivos del conjunto de datos

#∫HASTA AQUI FUNCIONA BIEN SOLO



## Etapa 2: Análisis exploratorio de los datos.

# Información general sobre el dataset
print("Información del Dataset:")
print(coleopteros_colombia_etiquetados.info())


print(coleopteros_colombiaNumEspecie.info())


# Resumen estadístico de las características numéricas
print("\nResumen Estadístico de las Características Numéricas:")
print(coleoptera_completa_mundo_especie.describe())

# Resumen de las características categóricas
print("\nResumen de las Características Categóricas:")
print(coleopteros_colombia.describe(include=['object']))







# Seleccionar solamente las categóricas
categorical_columns = coleopteros_colombia_etiquetados.select_dtypes(include=['object']).columns

# Quitar nan usando la moda para las variales categóricas

for col in categorical_columns:
    mode_value = coleopteros_colombia_etiquetados[col].mode()
    if not mode_value.empty:  
        coleopteros_colombia_etiquetados[col] = coleopteros_colombia_etiquetados[col].fillna(mode_value[0])
    else:
        coleopteros_colombia_etiquetados[col] = coleopteros_colombia_etiquetados[col].fillna('Unknown')

# Seleccionar solamente las numéricas
numerical_columns = coleopteros_colombia_etiquetados.select_dtypes(include=['float64']).columns








coleopteros_colombiaNumEspecie = coleopteros_colombia[['type',  'basisOfRecord',  'recordedBy',  'individualCount',  'organismQuantity',  'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'minimumElevationInMeters',  'maximumElevationInMeters',  'minimumDepthInMeters',  'maximumDepthInMeters',  'decimalLatitude',  'decimalLongitude',  'geodeticDatum',  'coordinateUncertaintyInMeters', 'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',  'NumeroEspeciesFamilia',  'ECC_Prcp_1976_2005_100K_2015',  'elevaciontiff',  'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual',  'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']]


categorical_features = ['type',  'basisOfRecord',  'recordedBy',   'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'geodeticDatum',  'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',   'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']


cols = ['minimumElevationInMeters','maximumElevationInMeters','minimumDepthInMeters','decimalLatitude','coordinateUncertaintyInMeters','NumeroEspeciesFamilia','elevaciontiff']

coleopteros_colombiaNumEspecie[cols] = coleopteros_colombiaNumEspecie[cols].apply(pd.to_numeric, errors='coerce', axis=1)


 

coleopteros_colombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']]=coleopteros_colombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']].replace('kWh/m²', '', regex=True)




cols = ['Radiacion_solar_global_promedio_multianual','Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010','Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']

coleopteros_colombiaNumEspecie[cols] = coleopteros_colombiaNumEspecie[cols].astype(str)




coleopteros_colombiaNumEspecie['Radiacion_solar_global_promedio_multianual'] = le.fit_transform(coleopteros_colombiaNumEspecie['Radiacion_solar_global_promedio_multianual'])






#quitar nan con método forwarfill

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='ffill')

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='bfill')







# Quitar nan usando  para las variables numéricas

df_interpolated = coleopteros_colombia.interpolate(method='linear', axis=0)
gdf_interpolated = coleopteros_colombia.interpolate()



### Crear un transformador para codificar las características categóricas


le = LabelEncoder()

for column in categorical_columns:
    coleopteros_colombia_etiquetados[column] = le.fit_transform(coleopteros_colombia_etiquetados[column])

print(ColDP_coleopteros.info())



#### Se exportan los datos completos luego de todo el procesamiento
coleopteros_colombiaNumEspecie.to_csv( "coleopteros_colombiaNumEspecie"+'.txt', sep="\t", encoding = "utf8")


####  Se divide en los conjuntos de entrenamiento y prueba.

# Dejar solamente los datos etiquetados en Y
y = coleoptera_completa_mundo_especie_colombia['NumeroEspeciesFamilia'].values 
arg = tf.convert_to_tensor(y, dtype=tf.float32)
y = y.astype(np.float32) 

# Dejar en X todos los datos menos la columna con las etiquetas
X = coleoptera_completa_mundo_especie_colombia.loc[:, coleoptera_completa_mundo_especie_colombia.columns != "NumeroEspeciesFamilia"] 

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)





# Se generan mini-lotes de la base de datos
buffer_size = 10000
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)





## Etapa 3: Entrenamiento de los modelos.

#### Modelo número 1 MAE como métrica
model = Sequential()
#model.add(Rescaling(1./255, input_shape=(62,)))
model.add(Dense(64, input_dim=61, activation='relu'))
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(1))  # Output layer, single unit (no activation, linear by default)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_dim=61),  # Input layer with 1 feature
    layers.Dense(32, activation='relu'),              # Hidden layer
    layers.Dense(1)                                   # Output layer with 1 unit (for numeric output)
])


model.summary()
model = createmodel()
tf.keras.utils.plot_model(model)

# Compile the model


optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50,
          validation_data=(X_test, y_test))


#### Modelo número 2 CET como métrica


model = Sequential()
model.add(Dense(380, input_dim=61, activation='relu'))
model.add(Dense(380, activation='relu')) 
model.add(Dense(380, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(380, activation='relu')) 
model.add(Dense(380, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=250)
history = model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test))


#### Modelo número 2 CET como métrica

# basada en entropía cruzada y se evalúa el rendimiento en términos del accuracy.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Se entrena el modelo para 10 épocas.
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))


plt.figure(figsize=(6, 4), dpi=160)

plt.plot(history.history["mae"], label="train")
plt.plot(history.history["val_mae"], label="test")
plt.legend()
plt.show()


test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)
# Finalmente, se evalúa el rendimiento del modelo en el conjunto de entrenamiento.
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('Test accuracy:', test_acc)

#### Modelo número 3  de clasificación por regresión logística

logisticRegr = LogisticRegression(solver="saga", max_iter=50000, tol=0.01)
logisticRegr.fit(X_train, y_train)

model = LogisticRegression(max_iter=50000, solver='saga', tol=0.01)

model.fit(X_train, y_train)

# Check model performance
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Test Accuracy: {model.score(X_test, y_test)}")




model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.NDCGMetric()])


#### Comparación entre modelos
'''
 We used paired Wilcoxon tests and the Holm
correction34 for multiple comparisons to test for significant differences
between models.
'''

#### Prediccion numero especies resto familias

# Predict on new data
X_new = np.random.rand(10, 5)  # New input samples (10 samples, 5 features each)
predictions = model.predict(X_new)

print("Predictions:", predictions)



coleopteros_colombia_exportar = coleopteros_colombia[[ 'gbifID','decimalLatitude', 'decimalLongitude', 'geodeticDatum', 'coordinateUncertaintyInMeters','scientificName']]

plantae_colombia = plantae_colombia[[ 'gbifID','decimalLatitude', 'decimalLongitude', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'taxonID', 'scientificName']]

plantae_colombia.to_csv( 'plantas_coordenadas'+'.txt', sep="\t", encoding = "utf8")

coleopteros_colombia_exportar.to_csv( 'coleoptera_coordenadas'+'.txt', sep="\t", encoding = "utf8")




# predecir los datos para Antioquia


coleopteros_colombia_etiquetados_antioquia=coleopteros_colombia_etiquetados.loc[coleopteros_colombia['stateProvince'] == 'Antioquia' ]
Mal
predictions = model.predict(X_new)

# pivot table para la suma de especies únicas por familia en Antioquia






# Structured data input
input_structured = Input(shape=(num_features,))  # e.g., climate data, location info

# CNN for satellite images input (optional)
input_image = Input(shape=(image_height, image_width, channels))  # e.g., satellite images

x1 = Dense(64, activation='relu')(input_structured)
x1 = Dropout(0.2)(x1)

# CNN layers for image input (optional)
x2 = Conv2D(32, (3, 3), activation='relu')(input_image)
x2 = Flatten()(x2)

# Concatenate structured data and CNN output
x = Concatenate()([x1, x2])

# Further dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)

# Output layer (regression task)
output = Dense(1)(x)  # Single output (species number)

# Create model
model = Model(inputs=[input_structured, input_image], outputs=output)

# Compile model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Summary of the model architecture
model.summary()







coleoptera_completa_mundo_especie_colombia['scientificName'].count()


print(coleoptera_completa_mundo_especie.columns)






unique_values_A = coleoptera_completa_mundo_especie_colombia['scientificName'].unique()
unique_values_A.count()



# generar descriptivos del conjunto de datos

# separar entrenar y entrenar




'''
En el documento vamos a utilizar la metodologia CRISP-DM
contexto de los datos
exploracion de los datos
de una vez colocar la descarga, el fultrado y con ese subconjunto de datos que estan en Antioquia
antes de entrenar toca sacar de alguna manera que queremos hacer con la red, neuronal. Intentar buscarar las variables
quedarnos por ahi con 

dejar un solo punto en geopandas con formato point
luego lo que podemos es estimar o tener todos los datos de colombia excepto el shape de colombia y luego con todo Colombia 

'''
