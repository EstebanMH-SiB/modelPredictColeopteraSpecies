# -*- coding: utf-8 -*-
"""EMH_Predicción_número_especies_coleoptera_antioquia.py

Enlace GitHub https://github.com/EstebanMH-SiB/modelPredictColeopteraSpecies

# Modelo de predicción para el número de especies de Coleoptera en el Departamento de Antioquia


## Etapa 0: Cargan de librerías necesarias
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


## Etapa 1: Carga y limpieza de los datos

# Se cargan las bases de datos

os.chdir("/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos")
coleopteros_colombia_completo = pd.read_csv('coleopteraColombia/verbatim.txt', encoding = "utf8", sep="\t")
coleopteros_colombia = pd.read_csv('coleopteraColombia/combinadoRaster.csv', encoding = "utf8", sep=",")
coleopteros_colombia = pd.read_csv('coleopterosColombia2025/verbatim.txt', encoding = "utf8", sep="\t")

plantae_colombia = pd.read_csv('PlantaeColombia/verbatim.txt', encoding = "utf8", sep="\t", usecols=['gbifID', 'occurrenceID', 'basisOfRecord', 'institutionID', 'institutionCode', 'collectionCode', 'catalogNumber', 'type', 'license', 'datasetID', 'datasetName', 'occurrenceRemarks', 'recordedBy', 'individualCount', 'sex', 'eventID', 'samplingProtocol', 'samplingEffort', 'eventDate', 'year', 'month', 'day', 'habitat', 'continent', 'waterBody', 'country', 'countryCode', 'stateProvince', 'county', 'municipality', 'locality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDepthInMeters', 'maximumDepthInMeters', 'locationRemarks', 'decimalLatitude', 'decimalLongitude', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'taxonID', 'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'specificEpithet', 'infraspecificEpithet', 'taxonRank', 'scientificNameAuthorship', 'identifiedBy', 'dateIdentified', 'bibliographicCitation', 'previousIdentifications'])

Radiacion_solar_global_promedio_multianual =gpd.read_file("Radiacion_solar_global_promedio_multianual/Shape/GDBIDEAM.CN2_Rdcn_Solar_Global_ProAnual.shp", encoding = "utf8")

Humedad_Relativa_Anual_Promedio_Multianual_1981_2010 =gpd.read_file("Humedad_Relativa_Anual_Promedio_Multianual_1981_2010/SHP/ACC2014_HmRl_AA_MA_1981_2010.shp", encoding = "utf8")

Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMx_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMd_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMn_MM_1981_2010_01.shp", encoding = "utf8")

Velocidad_viento_10_mtrs_altura_Mensual_2000_2010 =gpd.read_file("Velocidad_viento_10_mtrs_altura_Mensual_2000_2010/SHP/AVC2014_VlVn_10m_MM_2000_2010_01.shp", encoding = "utf8")

ColombiaGrilla10x10 =gpd.read_file("Grilla10x10/ColombiaGrilla10x10-WGS84.shp", encoding = "utf8")


with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/astergdem2.tif') as src:
        astergdem2 = src.read(1)

with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/Escenario_Precipitacion_1976_2005/ECC_Prcp_GeoTiff_2011_2040/ECC_Prcp_1976_2005_100K_2015.tif') as src:
        Escenario_Precipitacion_1976_2005 = src.read(1)
 


##  Se realiza una limpieza general de los datos

### Quitar las columnas completamente vacias

coleopteros_colombia.dropna(axis=1, how='all', inplace=True)

### Quitar las columnas que no tienen nada que ver con lo biológico

coleopteros_colombia = coleopteros_colombia.drop(columns=['accessRights', 'language','bibliographicCitation', 'modified','references','rightsHolder','institutionID','institutionCode','collectionID','collectionCode','license', 'collectionID', 'datasetID', 'collectionCode', 'datasetName', 'ownerInstitutionCode', 'informationWithheld', 'dataGeneralizations', 'dynamicProperties', 'occurrenceID', 'catalogNumber', 'recordNumber', 'recordedByID', 'georeferenceVerificationStatus', 'preparations', 'disposition', 'associatedMedia', 'associatedOccurrences', 'associatedReferences', 'associatedSequences', 'otherCatalogNumbers', 'organismScope', 'verbatimLabel', 'materialSampleID', 'parentEventID', 'eventType', 'startDayOfYear', 'endDayOfYear', 'year', 'month', 'day', 'verbatimEventDate', 'sampleSizeValue', 'sampleSizeUnit', 'samplingEffort', 'higherGeographyID', 'higherGeography', 'continent', 'locationAccordingTo', 'locationRemarks', 'islandGroup', 'countryCode', 'verbatimElevation', 'verbatimDepth', 'coordinatePrecision', 'verbatimCoordinates', 'verbatimLatitude', 'verbatimLongitude', 'verbatimCoordinateSystem', 'verbatimSRS', 'footprintWKT', 'footprintSRS', 'georeferencedBy', 'georeferencedDate', 'georeferenceProtocol', 'georeferenceSources', 'georeferenceRemarks', 'identificationID', 'identifiedByID', 'dateIdentified', 'identificationReferences', 'identificationVerificationStatus', 'taxonID', 'scientificNameID', 'nameAccordingToID', 'namePublishedInID', 'taxonConceptID', 'acceptedNameUsage', 'parentNameUsage', 'nameAccordingTo', 'namePublishedIn', 'namePublishedInYear', 'higherClassification', 'nomenclaturalCode', 'taxonomicStatus', 'nomenclaturalStatus', 'taxonRemarks'])


### Quitar los datos sin coordenadas para Coleoptera y Plantae

coleopteros_colombia = coleopteros_colombia.dropna(subset=['decimalLatitude', 'decimalLongitude'])

plantae_colombia = plantae_colombia.dropna(subset=['decimalLatitude', 'decimalLongitude'])

# Reemplazar las , por. y quitar los espacios en blanco para las coordenadas   
coleopteros_colombia[['decimalLongitude']]=coleopteros_colombia[['decimalLongitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)
coleopteros_colombia[['decimalLatitude']]=coleopteros_colombia[['decimalLatitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)


coleopteros_colombia = pd.read_csv('ColeopteroCompletaExtra.txt', encoding = "utf8", sep="\t")

# Crear columna con las coordenadas para hacer el cruce con las capas vectoriales

coleopteros_colombia[['decimalLatitude', 'decimalLongitude']] = coleopteros_colombia[['decimalLatitude','decimalLongitude']].fillna(value=0)
coleopteros_colombia['Coordinates'] = list(zip(coleopteros_colombia.decimalLongitude, coleopteros_colombia.decimalLatitude))
coleopteros_colombia['Coordinates'] = coleopteros_colombia['Coordinates'].apply(Point)
coleopteros_colombia = gpd.GeoDataFrame(coleopteros_colombia, geometry='Coordinates')
coleopteros_colombia.crs = {'init' :'epsg:4326'}


#Realizar cruces Geo radiacion
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Radiacion_solar_global_promedio_multianual, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'OBJECTID', 'ID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleopteros_colombia.rename(columns={'RANGO': 'Radiacion_solar_global_promedio_multianual'}, inplace=True)

#Realizar cruces Geo Humedad relativa
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Humedad_Relativa_Anual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'OBJECTID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleopteros_colombia.rename(columns={'RANGO': 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura maxima
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'PeriodoIni', 'PeriodoFin'])
coleopteros_colombia.rename(columns={'RANGO': 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)

#Realizar cruces Geo Temperatura media
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Temperatura_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'GRIDCODE_right','PeriodoIni', 'PeriodoFin'])
coleopteros_colombia.rename(columns={'RANGO': 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura media
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleopteros_colombia.rename(columns={'RANGO': 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura media
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleopteros_colombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)


#Realizar cruces VelocidadViento
coleopteros_colombia = gpd.sjoin(coleopteros_colombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleopteros_colombia = coleopteros_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleopteros_colombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)




coleopteros_colombia['elevation'] = rasterstats.point_query(coleopteros_colombia.geometry, astergdem2, interpolate='nearest')


coleopteros_colombia['elevation'] = rasterstats.point_query(coleopteros_colombia.geometry, Escenario_Precipitacion_1976_2005, interpolate='nearest')


# Convert the DataFrame to a GeoDataFrame
# Create a GeoDataFrame from the DataFrame's lat/lon coordinates
geometry = [Point(lon, lat) for lon, lat in zip(coleopteros_colombia['decimalLongitude'], coleopteros_colombia['decimalLatitude'])]
gdf = gpd.GeoDataFrame(coleopteros_colombia, geometry=geometry)
gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)
raster_file = '/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/astergdem2.tif'
results = rasterstats.point_query(gdf, raster_file)
coleopteros_colombia['elevacion'] = results

raster_file = '/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/Escenario_Precipitacion_1976_2005/ECC_Prcp_GeoTiff_2011_2040/ECC_Prcp_1976_2005_100K_2015.tif' 
results = rasterstats.point_query(gdf, raster_file)
coleopteros_colombia['Escenario_Precipitacion_1976_2005'] = results


coleopteros_colombia.to_csv( "ColeopteroCompletaExtra"+'.txt', sep="\t", encoding = "utf8")
coleopteros_colombia = pd.read_csv('ColeopteroCompletaExtraRaster.csv', encoding = "utf8", sep=",")
coleopteros_colombia.rename(columns={'SAMPLE_1': 'elevacion'}, inplace=True)
coleopteros_colombia.rename(columns={'SAMPLE_1_2': 'Escenario_Precipitacion_1976_2005'}, inplace=True)



#### Exportar los datos completos luego de haber agregado las columnas

coleopteros_colombia.to_csv( "ColeopteroCompletaColombia"+'.txt', sep="\t", encoding = "utf8")

#coleopteros_colombia = pd.read_csv('ColeopteroCompletaColombia.txt', encoding = "utf8", sep="\t")
coleopteros_colombia = pd.read_csv('ColeopteroCompletaExtraRaster.txt', encoding = "utf8", sep="\t")


# descartar los datos no etiquetados
coleopteros_colombia = coleopteros_colombia.dropna(subset=['NumeroEspeciesFamilia'])
coleopteros_colombia.dropna(axis=1, how='all', inplace=True)




 
coleopteros_colombiaNumEspecie = coleopteros_colombia.dropna(subset=['NumeroEspeciesFamilia'])
coleopteros_colombiaNumEspecie.dropna(axis=1, how='all', inplace=True)

list(coleopteros_colombiaNumEspecie)

coleopteros_colombiaNumEspecie = coleopteros_colombiaNumEspecie[['type',  'basisOfRecord',  'recordedBy',  'individualCount',  'organismQuantity',  'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'minimumElevationInMeters',  'maximumElevationInMeters',  'minimumDepthInMeters',  'maximumDepthInMeters',  'decimalLatitude',  'decimalLongitude',  'geodeticDatum',  'coordinateUncertaintyInMeters', 'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',  'NumeroEspeciesFamilia',  'ECC_Prcp_1976_2005_100K_2015',  'elevaciontiff',  'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual',  'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']]


categorical_features = ['type',  'basisOfRecord',  'recordedBy',   'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'geodeticDatum',  'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',   'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']


cols = ['minimumElevationInMeters','maximumElevationInMeters','minimumDepthInMeters','decimalLatitude','coordinateUncertaintyInMeters','NumeroEspeciesFamilia','elevaciontiff']

coleopteros_colombiaNumEspecie[cols] = coleopteros_colombiaNumEspecie[cols].apply(pd.to_numeric, errors='coerce', axis=1)



categorical_features = ['Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']

 

coleopteros_colombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']]=coleopteros_colombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']].replace('kWh/m²', '', regex=True)


# Crear un transformador para codificar las características categóricas


le = LabelEncoder()

for column in categorical_features:
    coleopteros_colombia[column] = le.fit_transform(coleopteros_colombia[column])

print(coleopteros_colombia.info())



cols = ['Radiacion_solar_global_promedio_multianual','Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010','Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']

coleopteros_colombiaNumEspecie[cols] = coleopteros_colombiaNumEspecie[cols].astype(str)




coleopteros_colombiaNumEspecie['Radiacion_solar_global_promedio_multianual'] = le.fit_transform(coleopteros_colombiaNumEspecie['Radiacion_solar_global_promedio_multianual'])




## Etapa 2: Análisis exploratorio de los datos.

# Información general sobre el dataset
print("Información del Dataset:")
print(coleopteros_colombia.info())


print(coleopteros_colombiaNumEspecie.info())


# Resumen estadístico de las características numéricas
print("\nResumen Estadístico de las Características Numéricas:")
print(coleopteros_colombia.describe())

# Resumen de las características categóricas
print("\nResumen de las Características Categóricas:")
print(coleopteros_colombia.describe(include=['object']))






#quitar nan con método forwarfill

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='ffill')

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='bfill')






coleopteros_colombiaNumEspecie.to_csv( "coleopteros_colombiaNumEspecie"+'.txt', sep="\t", encoding = "utf8")

#coleopteros_colombiaNumEspecie = pd.read_csv('coleopteros_colombiaNumEspecie.txt', encoding = "utf8", sep="\t")


#  Se divide en los conjuntos de entrenamiento y prueba.

 
y = coleopteros_colombia['NumeroEspeciesFamilia'].values 

arg = tf.convert_to_tensor(y, dtype=tf.float32)


y = y.astype(np.float32) 

X = coleopteros_colombia.loc[:, coleopteros_colombia.columns != "NumeroEspeciesFamilia"] 

 
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
