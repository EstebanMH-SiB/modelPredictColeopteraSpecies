# Citación e información asociada a los datos y capas utilizados en el modelo de predicción

Los datos de los coleopteros fueron descargados de [GBIF]() con el siguiente llamado al API : {
  "type": "and",
  "predicates": [
    {
      "type": "equals",
      "key": "COUNTRY",
      "value": "CO",
      "matchCase": false
    },
    {
      "type": "equals",
      "key": "TAXON_KEY",
      "value": "1470",
      "matchCase": false
    }
  ]
}

GBIF.org (21 October 2024) GBIF Occurrence Download https://doi.org/10.15468/dl.9jxk9n

Las capas provienen de: 


se realizó una búsqueda de los datos abiertos disponibles en repositorios nacionales, en especial en el portal del IDEAM (https://visualizador.ideam.gov.co/CatalogoObjetos/). Las capas de información encontradas para estas variables fueron:

Velocidad del viento mensual a 10 metros de altura de 2000 a 2010. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, clasificando la velocidad en 12 rangos significativos que oscilan entre 0 hasta superior a 11 m/s [25]. 
Mapa de ecosistemas continentales y marinos-costeros de Colombia a escala 1:100.000. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia y clasifican todos los ecosistemas del país en un total de 48 categorías, adicionalmente cada uno de los polígonos tiene información adicional como tipo de clima, relieve, suelo, unidad biótica y número de anfibios, reptiles, mamíferos aves y magnólias presentes en el ecosistema [26].  
Humedad relativa anual promedio multianual desde 1981 hasta 2010. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, clasificando la humedad en 6 rangos significativos que oscilan entre 65 y 95 porciento (%) [27].
Precipitación para Colombia (mm) 1976-2005. Escala 1:100,000. Los datos son un archivo raster tipo TIFF con un tamaño de celda (X,Y): 0.009, 0.009 Radianes por cada grado sexagesimal, clasificando la precipitación en 6 rangos significativos medidos en mm[28].
Radiación solar global promedio 2005. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, correspondientes al valor agregado de los kWh que en promedio inciden durante el día sobre un metro cuadrado, expresado en KWh/m2 [29].
Temperatura mínima mensual promedio durante el periodo 1981-2010. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, clasificando la temperatura en 8 rangos significativos que oscilan entre inferior a 8 y superior a 24 C° [30].
Temperatura media mensual promedio durante el periodo 1981-2010. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, clasificando la temperatura en 9 rangos significativos que oscilan entre inferior a 8 y superior a 28 C°  [31].
Temperatura máxima mensual promedio durante el periodo 1981-2010. Los datos son un archivo tipo ESRI shapefile con cobertura para Colombia, clasificando la temperatura en 9 rangos significativos que oscilan entre inferior a 8 y superior a 34 C°   [32].
Elevación NASA. Los datos son un archivo raster tipo TIFF con un tamaño de celda (X,Y) de 1 arco de segundo, dando un valor de elevación sobre el nivel del mar medido en metros usando el geoide EGM96 como referencia [33].
Adicionalmente a estas variables identificadas, se utilizó el mapa oficial de departamentos de Colombia para realizar los cruces geográficos [34].

La información de las especies de plantas asociadas a los coleópteros, se obtuvo descargando todos los registros biológicos del reino Plantae para Colombia directamente de GBIF con corte a febrero del 2025 [35]. Este conjunto de datos está en formato tabla, tiene 5’653.070 de filas y 50 columnas, que tienen información de la taxonomía, localidad, coordenadas, fecha, entidad publicadora, colector y tipo de registro.

Los datos referentes al orden Orden Coleoptera fueron descargados directamente de GBIF con corte a marzo de 2025 [36], utilizando el filtro para el orden Coleoptera en todo el mundo. Este conjunto de datos está en formato tabla, tiene 32’356.574 de filas y 50 columnas, que tienen información de la taxonomía, localidad, coordenadas, fecha, entidad publicadora, colector y tipo de registro.



Bibliografía:

[25]	R. R. Angela Maria, “Velocidad del Viento más Probable a 10 metros de altura Mensual durante el periodo 2000-2010. República de Colombia. Año 2015.” Accessed: Apr. 03, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/6b916fb1-67cb-4c42-a7c6-6080a92e592b
[26]	IDEAM et al., “Ecosistemas Continentales, Costeros y Marinos de Colombia. Escala 1:100.000. versión 2.1.Año 2017.” Accessed: Apr. 03, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/0684d637-5b6a-40e8-80f4-bdf915b3e3da
[27]	A. M. Ruiz Rotta, “Humedad Relativa Anual promedio Multianual durante el periodo 1981-2010. República de Colombia. Año 2014.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/048126d7-0ba7-4bd6-bd1c-3e475385e377
[28]	IDEAM, “Precipitación para Colombia (mm) 1976-2005. Escala 1:100,000. Año 2015.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/b2771c78-47eb-4b25-ad64-20a6f992e410
[29]	J. Albarracin, “Radiación global media recibida en una superficie horizontal durante el día promedio anual multianual (KWH/M2). República de Colombia.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/c2d36ff5-41de-47ff-a866-8a60932b8a31
[30]	A. M. Ruiz Rotta, “Temperatura Mínima Mensual Promedio Multianual durante el periodo 1981-2010. República de Colombia. Año 2014.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/7123a846-33bf-4831-b51c-2d11f629905f
[31]	R. R. Angela Maria, “Temperatura Media Mensual Promedio Multianual durante el periodo 1981-2010. República de Colombia. Año 2014.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/5f504388-2ad8-404c-a36f-e2b023357589
[32]	R. R. Angela Maria, “Temperatura Máxima Mensual Promedio Multianual durante el periodo 1981-2010.” Accessed: Apr. 10, 2025. [Online]. Available: https://visualizador.ideam.gov.co/geonetwork/srv/spa/catalog.search#/metadata/378ab856-167f-4d01-813a-f8b9e74934a5
[33]	NASA/METI/AIST/Japan Spacesystems And U.S./Japan ASTER Science Team, “ASTER Global Digital Elevation Model V003.” NASA Land Processes Distributed Active Archive Center, 2019. doi: 10.5067/ASTER/ASTGTM.003.
[34]	DANE, “Nivel geográfico de Departamentos del Marco Geoestadístico Nacional (MGN) versión 2020.” Accessed: Apr. 04, 2025. [Online]. Available: https://www.arcgis.com/home/webmap/viewer.html?url=https%3A%2F%2Fportalgis.dane.gov.co%2Fmparcgis%2Frest%2Fservices%2FMGN2020%2FServ_CapaDepartamentos_2020%2FMapServer&source=sd
[35]	GBIF.org, “GBIF Occurrence Download, Plantae Colombia,” https://www.gbif.org/. Accessed: Apr. 03, 2025. [Online]. Available: https://doi.org/10.15468/dl.j6xumw
[36]	GBIF.org, “GBIF Occurrence Download.” Accessed: Apr. 03, 2025. [Online]. Available: https://doi.org/10.15468/dl.retzxm
