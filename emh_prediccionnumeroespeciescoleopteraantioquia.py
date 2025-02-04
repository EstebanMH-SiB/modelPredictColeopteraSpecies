# -*- coding: utf-8 -*-
"""EMH_PredicciónNúmeroEspeciesColeopteraAntioquia.ipynb

Original file is located at
    https://colab.research.google.com/drive/1_yzQrLn3313dN8MyaY_yJ9T4i_XKFELy

# Modelo de predicción para el número de especies de Coleoptera en el Departamento de Antioquia


## Etapa 1: Carga de los datos.

Se cargan las librerías necesarias
"""

# Importa las librerías necesarias
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization
from keras.utils import plot_model
import os 
import rasterio
from rasterio import mask
from rasterio.transform import from_origin
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Rescaling



## Etapa 1: Carga y limpieza de los datos

# Se cargan las bases de datos

os.chdir("/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos")
coleopterosColombiaCompleto = pd.read_csv('coleopteraColombia/verbatim.txt', encoding = "utf8", sep="\t")
coleopterosColombia = pd.read_csv('coleopteraColombia/combinadoRaster.csv', encoding = "utf8", sep=",")

plantaeColombia = pd.read_csv('PlantaeColombia/verbatim.txt', encoding = "utf8", sep="\t", usecols=['gbifID', 'occurrenceID', 'basisOfRecord', 'institutionID', 'institutionCode', 'collectionCode', 'catalogNumber', 'type', 'license', 'datasetID', 'datasetName', 'occurrenceRemarks', 'recordedBy', 'individualCount', 'sex', 'eventID', 'samplingProtocol', 'samplingEffort', 'eventDate', 'year', 'month', 'day', 'habitat', 'continent', 'waterBody', 'country', 'countryCode', 'stateProvince', 'county', 'municipality', 'locality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDepthInMeters', 'maximumDepthInMeters', 'locationRemarks', 'decimalLatitude', 'decimalLongitude', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'taxonID', 'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'specificEpithet', 'infraspecificEpithet', 'taxonRank', 'scientificNameAuthorship', 'identifiedBy', 'dateIdentified', 'bibliographicCitation', 'previousIdentifications'])

Radiacion_solar_global_promedio_multianual =gpd.read_file("Radiacion_solar_global_promedio_multianual/Shape/GDBIDEAM.CN2_Rdcn_Solar_Global_ProAnual.shp", encoding = "utf8")

Humedad_Relativa_Anual_Promedio_Multianual_1981_2010 =gpd.read_file("Humedad_Relativa_Anual_Promedio_Multianual_1981_2010/SHP/ACC2014_HmRl_AA_MA_1981_2010.shp", encoding = "utf8")

Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMx_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMd_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010 =gpd.read_file("Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMn_MM_1981_2010_01.shp", encoding = "utf8")

Velocidad_viento_10_mtrs_altura_Mensual_2000_2010 =gpd.read_file("Velocidad_viento_10_mtrs_altura_Mensual_2000_2010/SHP/AVC2014_VlVn_10m_MM_2000_2010_01.shp", encoding = "utf8")

ColombiaGrilla10x10 =gpd.read_file("Grilla10x10/ColombiaGrilla10x10-WGS84.shp", encoding = "utf8")



with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/astergdem2.tif') as src:
        astergdem2 = src.read()

with rasterio.open('/Users/estebanmarentes/Desktop/EstebanMH/GBIFColombiaCompleto20200825/MaestriaUJaveriana/SegundoSemestre/TrabajoGradoII/Datos/Escenario_Precipitacion_1976_2005/ECC_Prcp_GeoTiff_2011_2040/ECC_Prcp_1976_2005_100K_2015.tif') as src:
        Escenario_Precipitacion_1976_2005 = src.read()
 
#  Se realiza una limpieza general de los datos

# Quitar los datos sin coordenadas para Coleoptera y Plantae

#Cual es el tipo de los atributos?

coleopterosColombia = coleopterosColombia.dropna(subset=['decimalLatitude', 'decimalLongitude'])

plantaeColombia = plantaeColombia.dropna(subset=['decimalLatitude', 'decimalLongitude'])

    
coleopterosColombia[['decimalLongitude']]=coleopterosColombia[['decimalLongitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)
coleopterosColombia[['decimalLatitude']]=coleopterosColombia[['decimalLatitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)





# Crear columna con las coordenadas para hacer el cruce con las capas vectoriales

coleopterosColombia[['decimalLatitude', 'decimalLongitude']] = coleopterosColombia[['decimalLatitude','decimalLongitude']].fillna(value=0)
coleopterosColombia['Coordinates'] = list(zip(coleopterosColombia.decimalLongitude, coleopterosColombia.decimalLatitude))
coleopterosColombia['Coordinates'] = coleopterosColombia['Coordinates'].apply(Point)
coleopterosColombia = gpd.GeoDataFrame(coleopterosColombia, geometry='Coordinates')
geo.crs = {'init' :'epsg:4326'}


#Realizar cruces Geo radiacion
coleopterosColombia = gpd.sjoin(coleopterosColombia, Radiacion_solar_global_promedio_multianual, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'OBJECTID', 'ID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleopterosColombia.rename(columns={'RANGO': 'Radiacion_solar_global_promedio_multianual'}, inplace=True)


#Realizar cruces Geo Humedad relativa
coleopterosColombia = gpd.sjoin(coleopterosColombia, Humedad_Relativa_Anual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'OBJECTID', 'GRIDCODE', 'Shape_Leng', 'Shape_Area','RULEID'])
coleopterosColombia.rename(columns={'RANGO': 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura maxima
coleopterosColombia = gpd.sjoin(coleopterosColombia, Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'PeriodoIni', 'PeriodoFin'])
coleopterosColombia.rename(columns={'RANGO': 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)

#Realizar cruces Geo Temperatura media
coleopterosColombia = gpd.sjoin(coleopterosColombia, Temperatura_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'GRIDCODE_right','PeriodoIni', 'PeriodoFin'])
coleopterosColombia.rename(columns={'RANGO': 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)


#Realizar cruces Geo Temperatura media
coleopterosColombia = gpd.sjoin(coleopterosColombia, Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleopterosColombia.rename(columns={'RANGO': 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)



#Realizar cruces Geo Temperatura media
coleopterosColombia = gpd.sjoin(coleopterosColombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleopterosColombia = coleopterosColombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleopterosColombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)




### Quitar las columnas completamente vacias

coleopterosColombia.dropna(axis=1, how='all', inplace=True)

### Quitar las columnas que no tienen nada que ver con lo biológico

coleopterosColombia = coleopterosColombia.drop(columns=['accessRights', 'language','bibliographicCitation', 'modified','references','rightsHolder','institutionID','institutionCode','collectionID','collectionCode','license', 'collectionID', 'datasetID', 'collectionCode', 'datasetName', 'ownerInstitutionCode', 'informationWithheld', 'dataGeneralizations', 'dynamicProperties', 'occurrenceID', 'catalogNumber', 'recordNumber', 'recordedByID', 'georeferenceVerificationStatus', 'preparations', 'disposition', 'associatedMedia', 'associatedOccurrences', 'associatedReferences', 'associatedSequences', 'otherCatalogNumbers', 'organismScope', 'verbatimLabel', 'materialSampleID', 'parentEventID', 'eventType', 'startDayOfYear', 'endDayOfYear', 'year', 'month', 'day', 'verbatimEventDate', 'sampleSizeValue', 'sampleSizeUnit', 'samplingEffort', 'higherGeographyID', 'higherGeography', 'continent', 'locationAccordingTo', 'locationRemarks', 'islandGroup', 'countryCode', 'verbatimElevation', 'verbatimDepth', 'coordinatePrecision', 'verbatimCoordinates', 'verbatimLatitude', 'verbatimLongitude', 'verbatimCoordinateSystem', 'verbatimSRS', 'footprintWKT', 'footprintSRS', 'georeferencedBy', 'georeferencedDate', 'georeferenceProtocol', 'georeferenceSources', 'georeferenceRemarks', 'identificationID', 'identifiedByID', 'dateIdentified', 'identificationReferences', 'identificationVerificationStatus', 'taxonID', 'scientificNameID', 'nameAccordingToID', 'namePublishedInID', 'taxonConceptID', 'acceptedNameUsage', 'parentNameUsage', 'nameAccordingTo', 'namePublishedIn', 'namePublishedInYear', 'higherClassification', 'nomenclaturalCode', 'taxonomicStatus', 'nomenclaturalStatus', 'taxonRemarks'])


coleopterosColombia[['decimalLongitude']]=coleopterosColombia[['decimalLongitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)




#### Exportar los datos completos luego de haber agregado las columnas

coleopterosColombia.to_csv( "ColeopteroCompletaColombia"+'.txt', sep="\t", encoding = "utf8")

#coleopterosColombia = pd.read_csv('ColeopteroCompletaColombia.txt', encoding = "utf8", sep="\t")


coleopterosColombia = coleopterosColombia.dropna(subset=['NumeroEspeciesFamilia'])


coleopterosColombia['NumeroEspeciesFamilia'].max()
coleopterosColombia['NumeroEspeciesFamilia'].min()

#  Se divide en los conjuntos de entrenamiento y prueba.


 
coleopterosColombiaNumEspecie = coleopterosColombia.dropna(subset=['NumeroEspeciesFamilia'])
coleopterosColombiaNumEspecie.dropna(axis=1, how='all', inplace=True)

list(coleopterosColombiaNumEspecie)

coleopterosColombiaNumEspecie = coleopterosColombiaNumEspecie[['type',  'basisOfRecord',  'recordedBy',  'individualCount',  'organismQuantity',  'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'minimumElevationInMeters',  'maximumElevationInMeters',  'minimumDepthInMeters',  'maximumDepthInMeters',  'decimalLatitude',  'decimalLongitude',  'geodeticDatum',  'coordinateUncertaintyInMeters',  'earliestEraOrLowestErathem',  'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',  'NumeroEspeciesFamilia',  'ECC_Prcp_1976_2005_100K_2015',  'elevaciontiff',  'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual',  'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']]


categorical_features = ['type',  'basisOfRecord',  'recordedBy',   'organismQuantityType',  'sex',  'lifeStage',  'reproductiveCondition',  'behavior',  'establishmentMeans',  'occurrenceStatus',  'associatedTaxa',  'occurrenceRemarks',  'previousIdentifications',  'organismRemarks',  'eventID',  'fieldNumber',  'eventDate',  'eventTime',  'habitat',  'samplingProtocol',  'fieldNotes',  'eventRemarks',  'waterBody',  'island',  'country',  'stateProvince',  'county',  'municipality',  'locality',  'verbatimLocality',  'geodeticDatum',   'earliestEraOrLowestErathem',  'verbatimIdentification',  'identificationQualifier',  'typeStatus',  'identifiedBy',  'identificationRemarks',  'scientificName',  'kingdom',  'phylum',  'class',  'order',  'superfamily',  'family',  'subfamily',  'tribe',  'genus',  'genericName',  'subgenus',  'specificEpithet',  'infraspecificEpithet',  'taxonRank',  'verbatimTaxonRank',  'scientificNameAuthorship',  'vernacularName',   'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010',  'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']


cols = ['minimumElevationInMeters','maximumElevationInMeters','minimumDepthInMeters','decimalLatitude','coordinateUncertaintyInMeters','NumeroEspeciesFamilia','elevaciontiff']

coleopterosColombiaNumEspecie[cols] = coleopterosColombiaNumEspecie[cols].apply(pd.to_numeric, errors='coerce', axis=1)



categorical_features = ['Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010',  'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010',  'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']

 

coleopterosColombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']]=coleopterosColombiaNumEspecie[['Radiacion_solar_global_promedio_multianual']].replace('kWh/m²', '', regex=True)

for column in categorical_features:
    coleopterosColombiaNumEspecie[column] = le.fit_transform(coleopterosColombiaNumEspecie[column])

print(coleopterosColombiaNumEspecie.info())



cols = ['Radiacion_solar_global_promedio_multianual','Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Media_Mensual_Promedio_Multianual_1981_2010','Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010','Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']

coleopterosColombiaNumEspecie[cols] = coleopterosColombiaNumEspecie[cols].astype(str)


le = LabelEncoder()

coleopterosColombiaNumEspecie['Radiacion_solar_global_promedio_multianual'] = le.fit_transform(coleopterosColombiaNumEspecie['Radiacion_solar_global_promedio_multianual'])







coleopterosColombiaNumEspecie.to_csv( "coleopterosColombiaNumEspecie"+'.txt', sep="\t", encoding = "utf8")

#coleopterosColombiaNumEspecie = pd.read_csv('coleopterosColombiaNumEspecie.txt', encoding = "utf8", sep="\t")

 
y = coleopterosColombia['NumeroEspeciesFamilia'].values 

arg = tf.convert_to_tensor(y, dtype=tf.float32)


y = y.astype(np.float32) 

X = coleopterosColombia.loc[:, coleopterosColombia.columns != "NumeroEspeciesFamilia"] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
 




## Etapa 2: Análisis exploratorio de los datos.

# Información general sobre el dataset
print("Información del Dataset:")
print(coleopterosColombia.info())


print(coleopterosColombiaNumEspecie.info())


# Resumen estadístico de las características numéricas
print("\nResumen Estadístico de las Características Numéricas:")
print(coleopterosColombia.describe())

# Resumen de las características categóricas
print("\nResumen de las Características Categóricas:")
print(coleopterosColombia.describe(include=['object']))













coleopterosColombia[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = coleopterosColombia[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].astype('str') 


coleopterosColombia['recordedBy'] = coleopterosColombia['recordedBy'].astype('str') 


# Crear un transformador para codificar las características categóricas


le = LabelEncoder()

coleopterosColombia['family'] = le.fit_transform(coleopterosColombia['family'])

# Definir las características categóricas que se van a codificar

resultados = coleopterosColombia[['basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']]



resultadocategoricos = coleopterosColombia[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks',  'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']]
print("Información del Dataset:")
print(coleopterosColombiaNumEspecie.info())

resultadocategoricos.dropna(axis=1, how='all', inplace=True)

#quitar nan con método forwarfill

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='ffill')

resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']] = resultadosTipoPublicador[['type', 'basisOfRecord', 'recordedBy', 'organismQuantity', 'organismQuantityType', 'sex', 'lifeStage', 'reproductiveCondition', 'behavior', 'vitality', 'establishmentMeans', 'degreeOfEstablishment', 'pathway', 'occurrenceStatus', 'associatedTaxa', 'occurrenceRemarks', 'organismID', 'associatedOrganisms', 'previousIdentifications', 'organismRemarks', 'materialEntityRemarks', 'eventID', 'fieldNumber', 'eventDate', 'eventTime', 'habitat', 'samplingProtocol', 'fieldNotes', 'eventRemarks', 'locationID', 'waterBody', 'island', 'country', 'stateProvince', 'county', 'municipality', 'locality', 'verbatimLocality', 'minimumElevationInMeters', 'maximumElevationInMeters', 'minimumDistanceAboveSurfaceInMeters', 'maximumDistanceAboveSurfaceInMeters', 'geodeticDatum', 'coordinateUncertaintyInMeters', 'footprintSpatialFit', 'earliestEraOrLowestErathem', 'earliestPeriodOrLowestSystem', 'latestPeriodOrHighestSystem', 'earliestEpochOrLowestSeries', 'latestEpochOrHighestSeries', 'member', 'bed', 'verbatimIdentification', 'identificationQualifier', 'typeStatus', 'identifiedBy', 'identificationRemarks', 'scientificName', 'originalNameUsage', 'kingdom', 'phylum', 'class', 'order', 'superfamily', 'family', 'subfamily', 'tribe', 'subtribe', 'genus', 'genericName', 'subgenus', 'infragenericEpithet', 'specificEpithet', 'infraspecificEpithet', 'cultivarEpithet', 'taxonRank', 'verbatimTaxonRank', 'scientificNameAuthorship', 'vernacularName', 'Coordinates', 'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010', 'Radiacion_solar_global_promedio_multianual', 'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010', 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010', 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010']].fillna(method='bfill')



categorical_features = ['type' ,'basisOfRecord' ,'recordedBy' ,'organismQuantityType' ,'sex' ,'lifeStage' ,'reproductiveCondition' ,'behavior' ,'establishmentMeans' ,'occurrenceStatus' ,'associatedTaxa' ,'occurrenceRemarks' ,'previousIdentifications' ,'organismRemarks' ,'eventID' ,'fieldNumber' ,'eventDate' ,'eventTime' ,'habitat' ,'samplingProtocol' ,'fieldNotes' ,'eventRemarks' ,'waterBody' ,'island' ,'country' ,'stateProvince' ,'county' ,'municipality' ,'locality' ,'verbatimLocality' ,'minimumElevationInMeters' ,'maximumElevationInMeters' ,'minimumDepthInMeters' ,'geodeticDatum' ,'coordinateUncertaintyInMeters' ,'earliestEraOrLowestErathem' ,'verbatimIdentification' ,'identificationQualifier' ,'typeStatus' ,'identifiedBy' ,'identificationRemarks' ,'scientificName' ,'kingdom' ,'phylum' ,'class' ,'order' ,'superfamily' ,'family' ,'subfamily' ,'tribe' ,'genus' ,'genericName' ,'subgenus' ,'specificEpithet' ,'infraspecificEpithet' ,'taxonRank' ,'verbatimTaxonRank' ,'scientificNameAuthorship' ,'vernacularName' ,'NumeroEspeciesFamilia' ,'Coordinates' ,'Humedad_Relativa_Anual_Promedio_Multianual_1981_2010' ,'Radiacion_solar_global_promedio_multianual' ,'GRIDCODE_left' ,'Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010' ,'Temperatura_Media_Mensual_Promedio_Multianual_1981_2010' ,'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010' ,'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010' ,'Column 99']

for column in categorical_features:
    coleopterosColombiaNumEspecie[column] = le.fit_transform(coleopterosColombiaNumEspecie[column])



for column in resultadocategoricos.columns:
    if resultadocategoricos[column].dtype == 'object':  # Check for categorical columns
        resultadosTipoPublicador[column] = le.fit_transform(resultadosTipoPublicador[column])


one_hot_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # Deja las características numéricas tal como están
)

# Aplicar la transformación al conjunto de datos
X_train_encoded = one_hot_encoder.fit_transform(X_train)
X_test_encoded = one_hot_encoder.transform(X_test)

# Mostrar el resultado
print(f'Número de características después de One-Hot Encoding: {X_train_encoded.shape[1]}')










coleopterosColombiaNumEspecie.replace([np.nan], 0)


coleopterosColombiaNumEspecie = coleopterosColombiaNumEspecie.drop(columns=['individualCount', 'organismQuantity','minimumElevationInMeters','maximumElevationInMeters','minimumDepthInMeters','maximumDepthInMeters','coordinateUncertaintyInMeters','decimalLatitude','decimalLongitude','NumeroEspeciesFamilia', 'elevaciontiff', 'ECC_Prcp_1976_2005_100K_2015', 'Unnamed: 0'])


print(coleopterosColombiaNumEspecie.info())



coleopterosColombiaNumEspecie.replace([np.nan], 0)

y = np.nan_to_num(y, nan=0)

y = coleopterosColombiaNumEspecie['NumeroEspeciesFamilia'].values 


arg = tf.convert_to_tensor(y, dtype=tf.float32)


y = y.astype(np.float32) 

X = coleopterosColombiaNumEspecie.loc[:, coleopterosColombiaNumEspecie.columns != "NumeroEspeciesFamilia"] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
 


X = X.loc[:, X.columns != "Unnamed: 0"] 



coleopterosColombiaNumEspecie = pd.read_csv('coleopterosColombiaNumEspecie.txt', encoding = "utf8", sep="\t")


X = coleopterosColombiaNumEspecie.loc[:, coleopterosColombiaNumEspecie.columns != "borrar"] 

y = coleopterosColombiaNumEspecie['NumeroEspeciesFamilia'].values 

X.to_csv( "coleopterosColombiaX"+'.txt', sep="\t", encoding = "utf8")

X = pd.read_csv('coleopterosColombiaX.txt', encoding = "utf8", sep="\t")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)







## Etapa 3: Entrenamiento de los modelos.

# Define the model
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

# Compile the model


optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])



model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])




# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=128)



# Se compila el modelo usando el optimizador ADAM, se fija una función de costo
# basada en entropía cruzada y se evalúa el rendimiento en términos del accuracy.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Se entrena el modelo para 10 épocas.
history = model.fit(X_train, y_train, epochs=50, batch_size=128)


plt.figure(figsize=(6, 4), dpi=160)

plt.plot(history.history["mae"], label="train")
plt.plot(history.history["val_mae"], label="test")
plt.legend()
plt.show()


history = model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test))

model.fit(X_train, y_train, epochs=50, batch_size=128)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)
# Finalmente, se evalúa el rendimiento del modelo en el conjunto de entrenamiento.
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('Test accuracy:', test_acc)


# Predict on new data
X_new = np.random.rand(10, 5)  # New input samples (10 samples, 5 features each)
predictions = model.predict(X_new)

print("Predictions:", predictions)

## Etapa 4: Validación de los modelos.




X_train = np.random.rand(100, 1)
y_train = 3 * X_train + 2 + np.random.randn(100, 1) * 0.1  # Linear relation with noise


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')



# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10)




coleopterosColombiasubset = coleopterosColombia[coleopterosColombia['county']=='Sibaté']

coleopterosColombiasubset.to_csv( "coleopterosColombiasubset"+'.txt', sep="\t", encoding = "utf8")




# Entrenamiento del modelo de clasificación por regresión logística


logisticRegr = LogisticRegression(solver="saga", max_iter=50000, tol=0.01)
logisticRegr.fit(X_train, y_train)

model = LogisticRegression(max_iter=50000, solver='saga', tol=0.01)

model.fit(X_train, y_train)

# Check model performance
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Test Accuracy: {model.score(X_test, y_test)}")













# Se generan mini-lotes de la base de datos
buffer_size = 10000
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])

# Se entrena el modelo para 10 épocas.
history = model.fit(train_dataset, epochs=1000,
          validation_data=test_dataset)


test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('Test accuracy:', test_acc)
