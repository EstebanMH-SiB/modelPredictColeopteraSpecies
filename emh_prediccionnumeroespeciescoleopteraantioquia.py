# -*- coding: utf-8 -*-
"""EMH_Predicción_número_especies_coleoptera_antioquia.py


Autor: Esteban Marentes Herrera
Enlace GitHub https://github.com/EstebanMH-SiB/modelPredictColeopteraSpecies

# Modelo de predicción para el número de especies de Coleoptera en el Departamento de Antioquia


"""

# Importa las librerías necesarias
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import tensorflow as tf
import keras
import keras_tuner as kt
import rasterio
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, ndcg_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import GridSearchCV
from docx import Document
from io import StringIO
from joblib import dump, load


# Definir el directorio de trabajo (modificar según sea necesario)

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

Radiacion_solar_global_promedio_multianual = gpd.read_file("Radiacion_solar_global_promedio_multianual/Shape/GDBIDEAM.CN2_Rdcn_Solar_Global_ProAnual.shp", encoding = "utf8")

Humedad_Relativa_Anual_Promedio_Multianual_1981_2010 = gpd.read_file("Humedad_Relativa_Anual_Promedio_Multianual_1981_2010/SHP/ACC2014_HmRl_AA_MA_1981_2010.shp", encoding = "utf8")

Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010 = gpd.read_file("Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMx_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Media_Mensual_Promedio_Multianual_1981_2010 = gpd.read_file("Temperatura_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMd_MM_1981_2010_01.shp", encoding = "utf8")

Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010 = gpd.read_file("Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010/SHP/ACC2014_TmMn_MM_1981_2010_01.shp", encoding = "utf8")

Velocidad_viento_10_mtrs_altura_Mensual_2000_2010 = gpd.read_file("Velocidad_viento_10_mtrs_altura_Mensual_2000_2010/SHP/AVC2014_VlVn_10m_MM_2000_2010_01.shp", encoding = "utf8")

ECCMC_Ver21_100K = gpd.read_file("Mapa_ecosistemas_Continentales_Marinos_Costeros_100K_V2.1_2017/Shape_E_ECCMC_Ver21_100K/E_ECCMC_Ver21_100K.shp", encoding = "utf8")

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

#Realizar cruces Geo Temperatura mínima
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010'}, inplace=True)

#Realizar cruces Velocidad viento
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, Velocidad_viento_10_mtrs_altura_Mensual_2000_2010, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'GRIDCODE','GRIDCODE_left','PeriodoIni', 'PeriodoFin'])
coleoptera_completa_mundo_especie_colombia.rename(columns={'RANGO': 'Velocidad_viento_10_mtrs_altura_Mensual_2000_2010'}, inplace=True)

#Realizar cruces Ecosistemas Continentales
coleoptera_completa_mundo_especie_colombia = gpd.sjoin(coleoptera_completa_mundo_especie_colombia, ECCMC_Ver21_100K, how="left", op="intersects")

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['index_right', 'OBJECTID','TIPO_ECOSI','GRADO_TRAN', 'GRAN_BIOMA', 'BIOMA_PREL','BIOMA_IAvH','ECOS_SINTE', 'AMBIENTE_A','SUBSISTEMA','ZONA_HIDRO', 'ORIGEN', 'TIPO_AGUA','PAISAJE','AMB_EDAFOG', 'DESC_AMB_E', 'SUSTRATO', 'ZONA', 'TEMPERATUR', 'SALINIDAD', 'PROVINCIA', 'ECO_REGION', 'ECO_ZONA', 'ORIGEN_MAR', 'CONFIGURAC', 'CLAS_BIOTI', 'SUBCLAS_BI', 'GRUPO_BIOT', 'SECTORES', 'Area_ha', 'UNI_BIOTIC', 'ANFIBIOS', 'AVES', 'MAGNOLIOPS', 'MAMIFEROS', 'REPTILES', 'SHAPE_Leng', 'SHAPE_Area', 'RULEID'])

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



### Cargar los datos de plantas y limpiarlos para dejar solamente las especies, los que tengan coordenadas y dejar datos sin duplicados
plantae_colombia = pd.read_csv('PlantaeColombiaCompleto/verbatim.txt', encoding = "utf8", sep="\t", usecols=['gbifID', 'occurrenceID',  'decimalLatitude', 'decimalLongitude','scientificName', 'taxonRank'])
plantae_colombia['taxonRank']=plantae_colombia['taxonRank'].str.title()
plantae_colombia['taxonRank']=plantae_colombia['taxonRank'].replace('Specie', 'Especie').replace('Espécie', 'Especie').replace('Epecie', 'Especie').replace('Especies', 'Especie').replace('Species', 'Especie')
plantae_colombia_especie=plantae_colombia.loc[plantae_colombia['taxonRank'] == 'Especie' ]
#Dejar solo las filas con información
plantae_colombia_especie = plantae_colombia_especie.dropna(subset=['decimalLatitude', 'decimalLongitude', 'scientificName'])
# Borrar las filas duplicadas
plantae_colombia_especie = plantae_colombia_especie.drop(columns=['occurrenceID'])
plantae_colombia_especie = plantae_colombia_especie.drop_duplicates(subset=[col for col in plantae_colombia_especie.columns if col != 'gbifID'])
# Limpiar las coordenadas
plantae_colombia_especie = plantae_colombia_especie[~plantae_colombia_especie['decimalLongitude'].str.contains('-', na=False)]
plantae_colombia_especie = plantae_colombia_especie[~plantae_colombia_especie['decimalLatitude'].isin(['71.122.222', '73.275.528','4.525694.'])]
plantae_colombia_especie[['decimalLongitude']]=plantae_colombia_especie[['decimalLongitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)
plantae_colombia_especie[['decimalLatitude']]=plantae_colombia_especie[['decimalLatitude']].replace(',', '.', regex=True).replace(' ', '', regex=True).astype(float)
plantae_colombia_especie['decimalLongitude'] = pd.to_numeric(plantae_colombia_especie['decimalLongitude'], errors='coerce')
plantae_colombia_especie['decimalLatitude'] = pd.to_numeric(plantae_colombia_especie['decimalLatitude'], errors='coerce')
# Trasnformar a GPD
plantae_colombia_especie[['decimalLatitude', 'decimalLongitude']] = plantae_colombia_especie[['decimalLatitude','decimalLongitude']].fillna(value=0)
plantae_colombia_especie['Coordinates'] = list(zip(plantae_colombia_especie.decimalLongitude, plantae_colombia_especie.decimalLatitude))
plantae_colombia_especie['Coordinates'] = plantae_colombia_especie['Coordinates'].apply(Point)
plantae_colombia_especie = gpd.GeoDataFrame(plantae_colombia_especie, geometry='Coordinates')
plantae_colombia_especie.crs = {'init' :'epsg:4326'}
# Cruzar a Colombia
plantae_colombia_especie = gpd.sjoin(plantae_colombia_especie, departamento, how="inner", op="intersects")
plantae_colombia_especie = plantae_colombia_especie.drop(columns=['index_right', 'DPTO_CCDGO', 'DPTO_NANO_', 'DPTO_CACTO', 'DPTO_NANO','Shape_Leng', 'Shape_Area', 'DPTO_CNMBR'])


# Dejar solamente las columnas de coordenadaas y gbifID para crear el buffer
gdf = coleoptera_completa_mundo_especie_colombia[[ "Coordinates", "gbifID"]]
#Crear un buffer aproximado de 500 metros de los puntos de Coleoptera
gdf['buffer'] = gdf.geometry.buffer(0.005)
gdf = gpd.GeoDataFrame(gdf, geometry='buffer')
# Dejar solamente las columnas de nombre científico y gbifID para los puntos de plantas
plantae_cruce = plantae_colombia_especie[[ "gbifID", "scientificName", "Coordinates"]]
plantae_cruce.rename(columns={'scientificName': 'especiePlanta'}, inplace=True) #cambiar nombre a especies plantas
# Cruce de los datos entre buffers y los puntos
coleopteros_colombia_plantas = gpd.sjoin_nearest(gdf, plantae_cruce , how='left')
# Crear la función para concatenar los datos
def concatenate_values(group):
    return pd.Series({
        'column1_concat': ' | '.join(group['especiePlanta'].astype(str))
    })
# Agrupar las especies por 'gbifID' y aplicar la función de concatenación
result = coleopteros_colombia_plantas.groupby('gbifID_left').apply(concatenate_values).reset_index()
# Cambiar los nombres de las columnas
result.rename(columns={'column1_concat': 'especiePlanta'}, inplace=True)
result.rename(columns={'gbifID_left': 'gbifID'}, inplace=True)

# Juntar la columna creada con el dataset Completo
coleoptera_completa_mundo_especie_colombia = pd.merge(coleoptera_completa_mundo_especie_colombia, result,  on='gbifID',  how='left')


# Etiquetar con los datos con la información de las familias
etiquetas_Coleoptera_Colombia = etiquetas_Coleoptera_Colombia[['family','NumeroEspeciesFamilia']] #Quedarse solo con la columna del número de registros
coleoptera_completa_mundo_especie_colombia = pd.merge(coleoptera_completa_mundo_especie_colombia,etiquetas_Coleoptera_Colombia,how='left',on='family')


# Exportar los datos completos
coleoptera_completa_colombia = coleoptera_completa_mundo_especie_colombia
coleoptera_completa_colombia.to_csv( 'coleoptera_completa'+'.txt', sep="\t", encoding = "utf8")


#### descartar los datos no etiquetados y borrar las columnas completamente vacias del nuevo dataset#######
coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.dropna(subset=['NumeroEspeciesFamilia'])
coleoptera_completa_mundo_especie_colombia.dropna(axis=1, how='all', inplace=True)


# Exportar los datos ajustados con etiquetas
coleoptera_completa_mundo_especie_colombia.to_csv( 'coleoptera_completa_mundo_especie_colombia'+'.txt', sep="\t", encoding = "utf8")


##########################################################
# Cargar solamente los datos ajustados
coleoptera_completa_mundo_especie_colombia = pd.read_csv('coleoptera_completa_mundo_especie_colombia.txt', encoding = "utf8", sep="\t")
coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=[ 'Unnamed: 0'])
# convertirlo en tipo gpd para poder hacer el mapa

coleoptera_completa_mundo_especie_colombia[['decimalLatitude', 'decimalLongitude']] = coleoptera_completa_mundo_especie_colombia[['decimalLatitude','decimalLongitude']].fillna(value=0)
coleoptera_completa_mundo_especie_colombia['Coordinates'] = list(zip(coleoptera_completa_mundo_especie_colombia.decimalLongitude, coleoptera_completa_mundo_especie_colombia.decimalLatitude))
coleoptera_completa_mundo_especie_colombia['Coordinates'] = coleoptera_completa_mundo_especie_colombia['Coordinates'].apply(Point)
coleoptera_completa_mundo_especie_colombia = gpd.GeoDataFrame(coleoptera_completa_mundo_especie_colombia, geometry='Coordinates')
coleoptera_completa_mundo_especie_colombia.crs = {'init' :'epsg:4326'}


### hacer un plot del mapa de Colombia par ver los datos
departamento =gpd.read_file("departamento/MGN_DPTO_POLITICO.shp", encoding = "utf8") # cargar el archivo shapefile de los municipios de Colombia
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


## Vista final de los datos y exportar una muestra
nan_count = coleoptera_completa_mundo_especie_colombia.isna().sum(axis=1)
vista_final = coleoptera_completa_mundo_especie_colombia[nan_count <= 4]
vista_final = vista_final.head(10)
vista_final = vista_final.transpose()
vista_final.to_excel("dataframe.xlsx", index=True)


### generar descriptivos del conjunto de datos



'''
En el documento vamos a utilizar la metodologia CRISP-DM
contexto de los datos
exploracion de los datos
de una vez colocar la descarga, el fultrado y con ese subconjunto de datos que estan en Antioquia
antes de entrenar toca sacar de alguna manera que queremos hacer con la red, neuronal. Intentar buscarar las variables
quedarnos por ahi con 

luego lo que podemos es estimar o tener todos los datos de colombia excepto el shape de colombia y luego con todo Colombia 



siempre tener presente los objetivos específicos, para poder irlos checando

dejar explicito todo el procesamiento de los datos, paso por paso
con la informacion de donde provienen
y al final decir cual es el conjunto de datos fianl, con las columnas

el enconder
'''

## Etapa 2: Análisis exploratorio de los datos.

# Información general sobre el dataset
print("Información del Dataset:")
print(coleoptera_completa_mundo_especie_colombia.info())

# Resumen estadístico de las características numéricas
print("\nResumen Estadístico de las Características Numéricas:")
print(coleoptera_completa_mundo_especie_colombia.describe())

# Resumen de las características categóricas
print("\nResumen de las Características Categóricas:")
print(coleoptera_completa_mundo_especie_colombia.describe(include=['object']))


def describe_dataframe_with_categorical_to_word(df, filename='dataframe_description.docx'):
    # Ensure pandas displays all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    # Create a new Document
    doc = Document()
    
    # Add a title
    doc.add_heading('General DataFrame Description', 0)
    
    # Adding basic information about the DataFrame
    doc.add_heading('1. Basic Information (Shape, Columns, Data Types):', level=1)
    doc.add_paragraph(f"Shape: {df.shape}")
    doc.add_paragraph(f"Columns: {list(df.columns)}")
    doc.add_paragraph(f"Data Types:\n{df.dtypes}")
    
    # Adding Summary Statistics for Numeric Columns (Include all columns)
    doc.add_heading('2. Summary Statistics for Numeric Columns:', level=1)
    numeric_summary = df.describe(include=[float, int])  # Show all numeric columns
    doc.add_paragraph(str(numeric_summary))
    
    # Adding Summary Statistics for Categorical Columns (Include all columns)
    doc.add_heading('3. Summary Statistics for Categorical Columns:', level=1)
    categorical_summary = df.describe(include=[object])  # Show all categorical columns
    doc.add_paragraph(str(categorical_summary))
    
    # Adding Missing Values Information
    doc.add_heading('4. Missing Values Information:', level=1)
    missing_values = df.isnull().sum()
    doc.add_paragraph(f"Missing Values per Column:\n{missing_values}")
    
    # Adding Non-null counts and memory usage
    doc.add_heading('5. Data Info (Non-null counts and memory usage):', level=1)
    buffer = StringIO()
    df.info(buf=buffer)
    doc.add_paragraph(buffer.getvalue())
    
    # Adding Count of Unique Values in Each Column
    doc.add_heading('6. Count of Unique Values per Column:', level=1)
    unique_values_count = df.nunique()
    doc.add_paragraph(f"{unique_values_count}")
    
    # Adding First 5 Rows of Data
    doc.add_heading('7. First 5 Rows of the DataFrame:', level=1)
    doc.add_paragraph(f"{df.head()}")
    
    # Save the document
    doc.save(filename)
    print(f"Description saved to {filename}")

# Assuming your DataFrame is called df:
describe_dataframe_with_categorical_to_word(coleoptera_completa_mundo_especie_colombia)


# Seleccionar solamente las variables categóricas
categorical_columns = coleoptera_completa_mundo_especie_colombia.select_dtypes(include=['object']).columns
# Seleccionar solamente variables las numéricas
numerical_columns = coleoptera_completa_mundo_especie_colombia.select_dtypes(include=['float64']).columns


# Crea gráficos de distribución para cada variable numérica
plt.figure(figsize=(18, 6))  # Ajustar el tamaño de la figura para múltiples subplots

for i, feature in enumerate(numerical_columns, 1):
    plt.subplot(1, len(numerical_columns), i)
    sns.histplot(coleoptera_completa_mundo_especie_colombia[feature], kde=True, bins=30)  # kde=True añade la curva de densidad
    plt.title(f'Distribución de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frecuencia')

plt.tight_layout()  # Ajustar el espaciado entre subplots
plt.show()



def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

categorical_corr = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))), index=categorical_columns, columns=categorical_columns)

for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 == col2:
            categorical_corr.loc[col1, col2] = 1.0
        else:
            confusion_matrix = pd.crosstab(coleoptera_completa_mundo_especie_colombia[col1], coleoptera_completa_mundo_especie_colombia[col2])
            categorical_corr.loc[col1, col2] = cramers_v(confusion_matrix.to_numpy())

# Grafica la matriz de Cramér's V
plt.figure(figsize=(12, 10))
sns.heatmap(categorical_corr, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Matriz de Asociación de Cramér\'s V para Variables Categóricas')
plt.show()





############ Quitar los NA ###############
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Loop through each column and apply imputation based on the column type
for column in coleoptera_completa_mundo_especie_colombia.columns:
    if coleoptera_completa_mundo_especie_colombia[column].dtype in ['float64', 'int64']:  # Numeric columns
        coleoptera_completa_mundo_especie_colombia[column] = numeric_imputer.fit_transform(coleoptera_completa_mundo_especie_colombia[[column]]).flatten()  # Impute with median
    else:  # Non-numeric columns (assume categorical)
        coleoptera_completa_mundo_especie_colombia[column] = categorical_imputer.fit_transform(coleoptera_completa_mundo_especie_colombia[[column]]).flatten()  # Impute with most frequent
############ Quitar los NA ###############


##### Crear un transformador para codificar las características categóricas

label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()  # Create a new LabelEncoder for each column
    coleoptera_completa_mundo_especie_colombia[column] = le.fit_transform(coleoptera_completa_mundo_especie_colombia[column])
    label_encoders[column] = le 
    
coleoptera_completa_mundo_especie_colombia.info()



## Vista final de los datos y exportar una muestra luego LabelEncoder
vista_final_le = coleoptera_completa_mundo_especie_colombia.head(5)
vista_final_le = vista_final_le.transpose()
vista_final_le.to_excel("dataframe_le.xlsx", index=True)



# Se eliminan las columnas que se identificaron como redundantes en el conjunto de datos

coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['Coordinates','gbifID','occurrenceID','kingdom','phylum','class', 'order', 'stateProvince','infraspecificEpithet','license'])

####  Se divide en los conjuntos de entrenamiento y prueba.

# Dejar solamente los datos etiquetados en Y
y = coleoptera_completa_mundo_especie_colombia['NumeroEspeciesFamilia'].values 
arg = tf.convert_to_tensor(y, dtype=tf.float32)
y = y.astype(np.float32) 

# Dejar en X todos los datos menos la columna con las etiquetas
X = coleoptera_completa_mundo_especie_colombia.loc[:, coleoptera_completa_mundo_especie_colombia.columns != "NumeroEspeciesFamilia"] 

# Primer split de 75%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
# Segundo split del 25% restante en test y validación
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.8, random_state=42)



## Etapa 3: Entrenamiento de los modelos.

#### Modelo número 0, regresión linear múltiple como base para las comparaciones

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred = model_linear.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


#### Modelo número 1, red neuronal profunda sencilla
model = Sequential()
model.add(Dense(64, input_dim=56, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(1))


model.summary()
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
# Calculate R-squared

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")
r2 = r2_score(y_test, y_pred)
# Print the R-squared value
print("R-squared:", r2)


#### Modelo número 2, red neuronal profunda sin datos estandarizados optimizando los hiperparámetros
np.random.seed(42)
K.clear_session()

# Define the model-building function for KerasTuner
def build_model(hp):
    model = Sequential()

    # First hidden layer with tunable number of neurons and activation function
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32),
                    activation=hp.Choice('activation_0', values=['relu', 'tanh', 'elu', 'leaky_relu', 'swish']),
                    input_dim=X_train.shape[1]))

    # Add tunable hidden layers
    for i in range(hp.Int('num_layers', 1, 5)):  # 1 to 5 hidden layers
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                        activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'elu', 'leaky_relu', 'swish'])))
        model.add(Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.2, max_value=0.5, step=0.1)))  # Tunable dropout

    # Output layer
    model.add(Dense(1))  # Single output for regression

    # Compile the model with a tunable learning rate
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error', metrics=['mae'])

    return model

# Create the KerasTuner Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',  # Minimize validation loss
    max_epochs=50,          # Max number of epochs for each trial
    hyperband_iterations=3, # Number of search iterations
    directory='my_dir',
    project_name='dnn_tuning3'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values

# Print the best hyperparameters
print("Best Hyperparameters: ", best_hyperparameters)

# Evaluate the best model on the test set
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")






# Salvar el mejor modelo
best_model.save('best_model_sinescalar.keras')
best_model_dnn = tf.keras.models.load_model('best_model_sinescalar.keras')
# Show the model architecture
best_model_dnn.summary()

y_pred = best_model_dnn.predict(X_test)
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# Print the R-squared value
print("R-squared:", r2)

loss, mae = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")


historySinEscalar = best_model_dnn.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test))



# Summary of the model
best_model.summary()

# Graficar los resultados
plt.plot(historySinEscalar.history['mae'])
plt.plot(historySinEscalar.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(historySinEscalar.history['loss'])
plt.plot(historySinEscalar.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#### Modelo número 3, red neuronal profunda con los datos escalados optimizando los hiperparámetros
np.random.seed(42)
K.clear_session()
# Escalamiento de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model-building function for KerasTuner
def build_model(hp):
    model = Sequential()

    # First hidden layer with tunable number of neurons and activation function
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32),
                    activation=hp.Choice('activation_0', values=['relu', 'tanh', 'elu', 'leaky_relu', 'swish']),
                    input_dim=X_train.shape[1]))

    # Add tunable hidden layers
    for i in range(hp.Int('num_layers', 1, 5)):  # 1 to 5 hidden layers
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                        activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'elu', 'leaky_relu', 'swish'])))
        model.add(Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.2, max_value=0.5, step=0.1)))  # Tunable dropout

    # Output layer
    model.add(Dense(1))  # Single output for regression

    # Compile the model with a tunable learning rate
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error', metrics=['mae'])

    return model

# Create the KerasTuner Hyperband tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',  # Minimize validation loss
    max_epochs=50,          # Max number of epochs for each trial
    hyperband_iterations=3, # Number of search iterations
    directory='my_dir',
    project_name='dnn_escalado'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Get the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values

# Print the best hyperparameters
print("Best Hyperparameters: ", best_hyperparameters)

# Evaluate the best model on the test set
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")


# Salvar el mejor modelo
best_model.save('best_model_escalado.keras')
best_model_dnn_escalado = tf.keras.models.load_model('best_model_escalado.keras')
# Show the model architecture
best_model_dnn_escalado.summary()

y_pred = best_model_dnn_escalado.predict(X_test)
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# Print the R-squared value
print("R-squared:", r2)

loss, mae = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")


historySinEscalar = best_model_dnn.fit(X_train, y_train, epochs=50, batch_size=12, validation_data=(X_test, y_test))



# Summary of the model
best_model.summary()

# Graficar los resultados
plt.plot(historySinEscalar.history['mae'])
plt.plot(historySinEscalar.history['val_mae'])
plt.title('model mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(historySinEscalar.history['loss'])
plt.plot(historySinEscalar.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#### Modelo número 4, Random Forest optimizando los hiperparámetros

# Definir el RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Definir los parámetros a probar en el Grid Search
param_grid_rf = {
    'n_estimators': [50, 100, 150],        # Número de árboles
    'max_depth': [None, 16, 32],           # Profundidad máxima del árbol
    'min_samples_split': [2, 3, 4]         # Número mínimo de muestras para hacer un split
}

# Crear el GridSearchCV con 5 folds de cross-validation
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='neg_mean_squared_error',     # Para regresión usamos MSE negativo (porque GridSearchCV maximiza)
    n_jobs=-1,
    verbose=2
)

# Ajustar el Grid Search a los datos de entrenamiento
grid_search_rf.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print(f"Mejores parámetros encontrados para RandomForestRegressor: {grid_search_rf.best_params_}")

# Evaluar el mejor modelo con el dataset de validación
best_model_rf = grid_search_rf.best_estimator_
y_pred_test_rf = best_model_rf.predict(X_test)

# Calcular métricas en el conjunto de validación
mse_val = mean_squared_error(y_test, y_pred_test_rf)
mae_val = mean_absolute_error(y_test, y_pred_test_rf)
r2_val = r2_score(y_test, y_pred_test_rf)

# Calcular las métricas promedio en la validación cruzada
mean_mse_cv = -grid_search_rf.cv_results_['mean_test_score'][grid_search_rf.best_index_]
std_mse_cv = grid_search_rf.cv_results_['std_test_score'][grid_search_rf.best_index_]

# Obtener las predicciones en el dataset de entrenamiento
y_pred_train_rf = grid_search_rf.predict(X_train)

# Calcular métricas en el set de entrenamiento
mse_train = mean_squared_error(y_train, y_pred_train_rf)
mae_train = mean_absolute_error(y_train, y_pred_train_rf)
r2_train = r2_score(y_train, y_pred_train_rf)

# Mostrar los resultados finales del RandomForestRegressor con GridSearch
print("\nComparación de resultados entre Training (Cross-Validation) y Validación para el RandomForestRegressor:\n")

# Resultados de cross-validation
print(f"Cross-Validation (5 folds) en el dataset de Training:")
print(f"MSE: {mean_mse_cv:.4f} (+/- {std_mse_cv:.4f})")
print(f"MAE (Train): {mae_train:.4f}")
print(f"R2 (Train): {r2_train:.4f}")

# Resultados en el dataset de validación
print(f"\nEvaluación de modelo ajustado con Training en dataset de Validación:")
print(f"MSE (Test): {mse_val:.4f}")
print(f"MAE (Test): {mae_val:.4f}")
print(f"R2 (Test): {r2_val:.4f}")

# Mostrar los mejores hiperparámetros encontrados
print(f"\nMejores parámetros encontrados: {grid_search_rf.best_params_}")





# Instantiate the RandomForestRegressor with the best hyperparameters
best_rf = RandomForestRegressor(
    max_depth=None,
    min_samples_split=4,
    n_estimators=50,
    random_state=42
)

# Fit the model on the training data
best_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Optionally, you can also evaluate the R-squared score:
r2 = best_rf.score(X_test, y_test)
print(f"R-squared: {r2}")


# Make predictions on the training set
y_train_pred = best_rf.predict(X_train)

# Calculate the Mean Squared Error (MSE) for training and testing sets
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared for training and testing sets
train_r2 = best_rf.score(X_train, y_train)
test_r2 = best_rf.score(X_test, y_test)

# Print out the results
print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Test R-squared: {test_r2}")



# 5-fold cross-validation
cross_val_scores = cross_val_score(best_rf, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE: {-cross_val_scores.mean()}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# Salvar el mejor modelo RF
dump(grid_search_rf, 'best_model_randomforest.joblib')
# best_rf = load('best_model_randomforest.joblib')



##### Predicción número de especies para Coleoptera de Colombia con el mejor modelo

model = best_model_dnn # Cambiar el model según el que tenga el mejor resultado entre: model_linear, best_model_dnn_escalado, best_model_dnn, best_model_rf

prediccion_especies = model.predict(coleoptera_completa_colombia)

# Unir las predicciones a la tabla
estimacion_colombia = pd.DataFrame(coleoptera_completa_colombia) 
estimacion_colombia['prediccion_especies'] = prediccion_especies 

for column, encoder in label_encoders.items():
    estimacion_colombia[column] = encoder.inverse_transform(estimacion_colombia[column])


# pivot table para la suma de especies únicas por familia en Colombia
tabla_resumen_estimaciones = pd.pivot_table(estimacion_colombia, values='prediccion_especies', index=['departamento','family'], aggfunc='mean')

# Redondear los datos al entero más cercan
pivot_table_rounded = np.ceil(tabla_resumen_estimaciones).astype(int)

# Exportar la tabla dinamica
pivot_table_rounded.to_excel('pivot_table_predictions.xlsx')

###############################################################################


##### Predicción número de especies para Coleoptera de Antioquia con el mejor modelo
prediccion_especies_antioquia = model.predict(coleoptera_completa_colombia)


for column, encoder in label_encoders.items():
    coleoptera_completa_mundo_especie_colombia[column] = encoder.inverse_transform(coleoptera_completa_mundo_especie_colombia[column])

coleopteros_colombia_etiquetados_antioquia=coleoptera_completa_mundo_especie_colombia.loc[coleoptera_completa_mundo_especie_colombia['departamento'] == 'Antioquia' ]

# Unir las predicciones a la tabla
estimacion_colombia = pd.DataFrame(coleopteros_colombia_etiquetados_antioquia)  # Convert X_test to DataFrame if it's not already
estimacion_colombia['prediccion_especies_antioquia'] = prediccion_especies_antioquia  # Add the predicted values as a new column


# pivot table para la suma de especies únicas por familia en Colombia
tabla_resumen_estimaciones = pd.pivot_table(estimacion_colombia, values='prediccion_especies_antioquia', index=['departamento','family'], aggfunc='mean')

# Redondear los datos al entero más cercan
pivot_table_rounded = np.ceil(tabla_resumen_estimaciones).astype(int)

# Exportar la tabla dinamica
pivot_table_rounded.to_excel('tabla_resumen_estimaciones_antioquia.xlsx')








# Aquí se acaba el script funcional______________________________________________________________________________________________________________________________________________________


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





# generar descriptivos del conjunto de datos

# separar entrenar y entrenar

def train_and_evaluate_regression_model(model, X_train, y_train, X_test, y_test, n_iterations=10):
    r2_scores, mae_scores, mse_scores, rmse_scores = [], [], [], []

    for _ in range(n_iterations):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        rmse_scores.append(mean_squared_error(y_test, y_pred, squared=False))  # RMSE

    return (
        np.mean(r2_scores),
        np.mean(mae_scores),
        np.mean(mse_scores),
        np.mean(rmse_scores)
    )

# Modelos de regresión
regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'ANN (MLP Regressor)': MLPRegressor(random_state=42, max_iter=1000),
    'XGBoost Regressor': XGBRegressor(random_state=42)
}

# Resultados
regression_results = {
    'Model': [],
    'R2 (Avg)': [],
    'MAE (Avg)': [],
    'MSE (Avg)': [],
    'RMSE (Avg)': []
}

# Entrenar y evaluar modelos de regresión
for model_name, model in regression_models.items():
    r2, mae, mse, rmse = train_and_evaluate_regression_model(model, X_train, y_train, X_test, y_test)
    regression_results['Model'].append(model_name)
    regression_results['R2 (Avg)'].append(r2)
    regression_results['MAE (Avg)'].append(mae)
    regression_results['MSE (Avg)'].append(mse)
    regression_results['RMSE (Avg)'].append(rmse)

# Mostrar resultados
regression_results_df = pd.DataFrame(regression_results)

print("\nTabla de Desempeño Promedio de Modelos de Regresión (10 Mediciones)")
print(regression_results_df.to_string(index=False))

#####################test################


coleoptera_completa_colombia = pd.read_csv('coleoptera_completa.txt', encoding = "utf8", sep="\t")

coleoptera_completa_colombia = coleoptera_completa_colombia.drop(columns=[ 'Unnamed: 0'])


# Seleccionar solamente las variables categóricas
categorical_columns = coleoptera_completa_colombia.select_dtypes(include=['object']).columns
# Seleccionar solamente variables las numéricas
numerical_columns = coleoptera_completa_colombia.select_dtypes(include=['float64']).columns

############ Quitar los NA ###############
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Loop through each column and apply imputation based on the column type
for column in coleoptera_completa_colombia.columns:
    if coleoptera_completa_colombia[column].dtype in ['float64', 'int64']:  # Numeric columns
        coleoptera_completa_colombia[column] = numeric_imputer.fit_transform(coleoptera_completa_colombia[[column]]).flatten()  # Impute with median
    else:  # Non-numeric columns (assume categorical)
        coleoptera_completa_colombia[column] = categorical_imputer.fit_transform(coleoptera_completa_colombia[[column]]).flatten()  # Impute with most frequent
############ Quitar los NA ###############


##### Crear un transformador para codificar las características categóricas
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()  # Create a new LabelEncoder for each column
    coleoptera_completa_colombia[column] = le.fit_transform(coleoptera_completa_colombia[column])
    label_encoders[column] = le 
  

coleoptera_completa_colombia.info()


# Se eliminan las columnas que se identificaron como redundantes en el conjunto de datos

coleoptera_completa_colombia = coleoptera_completa_colombia.drop(columns=['Coordinates','gbifID','occurrenceID','kingdom','phylum','class', 'order', 'stateProvince','infraspecificEpithet','license'])

####  Se divide en los conjuntos de entrenamiento y prueba.
coleoptera_completa_colombia = coleoptera_completa_colombia.drop(columns=['Velocidad_viento_10_mtrs_altura_Mensual_2000_2010.1','NumeroEspeciesFamilia'])
