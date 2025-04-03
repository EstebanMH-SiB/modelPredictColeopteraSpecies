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
import rasterio
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, ndcg_score, mean_squared_error, r2_score
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.optimizers import Adam
from docx import Document
from io import StringIO


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
coleoptera_completa_mundo_especie_colombia.to_csv( 'coleoptera_completa'+'.txt', sep="\t", encoding = "utf8")


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

dejar un solo punto en geopandas con formato point
luego lo que podemos es estimar o tener todos los datos de colombia excepto el shape de colombia y luego con todo Colombia 

'''

'''
siempre tener presente los objetivos específicos, para poder irlos checando

dejar explicito todo el procesamiento de los datos, paso por paso
con la informacion de donde provienen
y al final decir cual es el conjunto de datos fianl, con las columnas

el enconder


A partir de eso empezar a entrenar los modelos


'''


## Etapa 2: Análisis exploratorio de los datos.



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


# Seleccionar solamente las variables categóricas
categorical_columns = coleoptera_completa_mundo_especie_colombia.select_dtypes(include=['object']).columns
# Seleccionar solamente variables las numéricas
numerical_columns = coleoptera_completa_mundo_especie_colombia.select_dtypes(include=['float64']).columns
### Crear un transformador para codificar las características categóricas
le = LabelEncoder()
# Pasar el encoder en todos las columnas categóricas
for column in categorical_columns:
    coleoptera_completa_mundo_especie_colombia[column] = le.fit_transform(coleoptera_completa_mundo_especie_colombia[column])


coleoptera_completa_mundo_especie_colombia.info()



####  Se divide en los conjuntos de entrenamiento y prueba.
coleoptera_completa_mundo_especie_colombia = coleoptera_completa_mundo_especie_colombia.drop(columns=['Coordinates'])
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



''' 

Describir nu

dependiendo de lo que nos de, podemos ahcer un algoritmo augmentation

'''


coleoptera_completa_mundo_especie_colombia.head()



# Se generan mini-lotes de la base de datos
buffer_size = 10000
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)





## Etapa 3: Entrenamiento de los modelos.

#### Modelo número 0, regresión linear múltiple como base para las comparaciones

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)






#### Modelo número 1 MAE como métrica
model = Sequential()
model.add(Dense(64, input_dim=47, activation='relu'))
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer, single unit (no activation, linear by default)



model.summary()
tf.keras.utils.plot_model(model)

# Compile the model


optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
# Print the R-squared value
print("R-squared:", r2)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")


#### Modelo número 2 CET como métrica
model = models.Sequential([
    layers.Dense(64, activation='relu', input_dim=47),  # Input layer with 1 feature
    layers.Dense(32, activation='relu'),              # Hidden layer
    layers.Dense(1)                                   # Output layer with 1 unit (for numeric output)
])


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







model = Sequential([
    Dense(16, input_dim=47, activation='relu'),  # 10 features in input data
    Dense(1)  # Sigmoid output for binary classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), loss='mean_squared_error', metrics=['mae'])

# Summary of the model
model.summary()


early_stopping = EarlyStopping(monitor='loss', patience=3)
model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[early_stopping])

# Example: Fit the model (replace `X_train` and `y_train` with your data)
model.fit(X_train, y_train, epochs=10, batch_size=32)


with tf.device('/CPU:0'):
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    
    
    
X_train_subset = X_train[:100]
y_train_subset = y_train[:100]

model.fit(X_train_subset, y_train_subset, epochs=10, batch_size=32)



with tf.device('/CPU:0'):
    model.fit(X_train_subset, y_train_subset, epochs=10, batch_size=32)
