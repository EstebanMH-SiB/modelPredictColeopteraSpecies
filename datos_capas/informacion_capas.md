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

Para los datos climáticos y de hábitat, se realizó una descarga desde la página del IDEAM (http://dhime.ideam.gov.co/atencionciudadano/) de los siguientes mapas:

- Velocidad_viento_10_mtrs_altura_Mensual_2000_2010 [69]
- Mapa_ecosistemas_Continentales_Marinos_Costeros_100K_V2.1_2017 [70]
- Humedad_Relativa_Anual_Promedio_Multianual_1981_2010 [71]
- Escenario_Precipitacion_1976_2005 [72]
- Radiacion_solar_global_promedio_multianual [73]
- Temperatura_Minima_Media_Mensual_Promedio_Multianual_1981_2010 [74]
- Temperatura_Media_Mensual_Promedio_Multianual_1981_2010 [75]
- Temperatura_Maxima_Media_Mensual_Promedio_Multianual_1981_2010 [76]
- Elevación [77]
