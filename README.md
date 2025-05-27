# Modelo de predicción para el número de especies de Coleoptera en el Departamento de Antioquia

Este repositorio contiene el script utilizado en el Proyecto Aplicado para optar al título de Magister en Ciencia de Datos de Esteban Marentes.

El proceso se llevó a cabo en la IDE [Spyder v6](https://www.spyder-ide.org/) al interior de Anaconda. Ejecutar todo el script toma aproximadamente 5-6 horas usando un computador con un procesador 2.3 GHz intel Core i9 de 8 núcleos, una memoria RAM de 16 GB y una tarjeta gráfica AMD Radep Pro 5500M 4 GB.

Pasos para ejecutar el script:

1. Instalar el ambiente usando el archivo 'requirements.yml', se recomienda hacerlo en [Anaconda o Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) siguiendo la guía [Environments](https://www.anaconda.com/docs/tools/working-with-conda/environments#creating-an-environment-from-a-yml-file).
    > Es posible que el ambiente no installe las librerías listadas bajo pip, en ese caso deberá instalarlas manualmente luego de ingresar a Spyder.
2. Descargar todos los archivos indicados en el archivo [`informacion_capas.md`](./datos_capas/informacion_capas.md)
3. Abrir el archivo '[emh_prediccionnumeroespeciescoleopteraantioquia.py](https://github.com/EstebanMH-SiB/modelPredictColeopteraSpecies/blob/main/emh_prediccionnumeroespeciescoleopteraantioquia.py)' en Spyder
    > Asegúrese de que el intérprete de spider corresponda al ambiente de conda creado a partir del archivo del punto 1.
    >
    >Para configurar el intérprete en spider ver `Q: How do I get Spyder to work with my existing Python packages/environment?` en el [faq de spider](https://docs.spyder-ide.org/current/faq.html).
    >
    >Para determinar el intérprete del ambiente de conda [ver aquí](https://www.anaconda.com/docs/tools/working-with-conda/ide-tutorials/python-path).

4. Modificar la [línea 39](./emh_prediccionnumeroespeciescoleopteraantioquia.py#L39-L40) con el directorio donde están guardados los documentos del paso 2.
5. Ejecutar la importación de librerías para verificar que el ambiente está correcto.
6. Correr el script, se recomienda hacerlo por etapas para ir verificando los resultado y evitar errores en el código. Esto puede tomar al menos 5 horas en total, dependiendo de la capacidad del equipo. No se recomienda correr todo el script de una vez, aunque posible podrían encontrarse errores inesperados que creen conflictos.
7. En la fila 874 puede cambiar el modelo con el que desea hacer la predicción, por defecto está el que dió un mejor resultado en el trabajo de grado best_model_dnn_escalado.
8. Una vez se corra la totalidad del script, en la carpeta que definió en el paso 4 encontrará 25 archivos adicionales, entre los que se encuentran:
   - Imágenes exportadas (dnn_escalado_mae.png, dnn_escalado_mse.png, dnn_mae.png, dnn_mse.png, errores_linea.png, fold_separado.png, mapaColeoptera.png, random_forest_tree_0.png)
   - Archivos de texto plano de trabajo intermedio (coleoptera_completa.txt, coleoptera_completa_final.txt, coleoptera_completa_le.txt, coleoptera_completa_mundo_especie_colombia_etiquetados_final.txt, coleoptera_completa_mundo_especie_colombia_etiquetados.txt, coleoptera_completa_sinna.txt)
   - Una carpeta my_dir con todos los modelos intermedios generados en el proceso de optimización.
   - Archivos con la información de los modelos finales (best_model_escalado.keras, best_model_randomforest_v2.joblib, best_model_sinescalar.keras, dnn_simple.keras, modelo_lineal_entrenado.pkl, model_architecture.json).
   - Dos archivos de Excel con los resultados finales de la predicción del número de especies de Coleoptera para el modelo: pivot_table_predictions_colombia_deparmentos.xlsx y pivot_table_predicciones_Antioquia.xlsx
9. Realice un ajuste manual de los archivos exportados para que sigan la taxonomía de J. F. Lawrence and A. F. Newton, “Families and subfamilies of Coleoptera (with selected genera, notes, references and data on family-group names). Esto debe darle un archivo similar al presente en el repositorio [EstimacionNumeroEspecies_TesisEMH.xlsx](https://github.com/EstebanMH-SiB/modelPredictColeopteraSpecies/blob/main/EstimacionNumeroEspecies_TesisEMH.xlsx) aunque no exactamente igual, teniendo en cuenta que el resultado del mejor modelo puede variar al ser un proceso estocástico.
