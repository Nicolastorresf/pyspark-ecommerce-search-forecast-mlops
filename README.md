# pyspark-ecommerce-search-forecast-mlops
-->Un proyecto de prueba de concepto que demuestra el análisis de términos de búsqueda de un sitio de comercio electrónico y el uso de un modelo preentrenado de pronóstico de ventas utilizando PySpark. Este repositorio explora elementos básicos de MLOps como el guardado y carga de modelos. <3

# Proyecto PySpark: Análisis de Búsquedas y Pronóstico de Ventas con MLOps

## Descripción General

Este repositorio documenta un proyecto realizado con PySpark dentro de un entorno Jupyter Notebook. El objetivo principal es demostrar un flujo de trabajo que incluye el análisis de datos de términos de búsqueda de un sitio de comercio electrónico y la utilización de modelos de Machine Learning (SparkML) para realizar pronósticos. Se exploran conceptos básicos de MLOps como la persistencia (guardado y carga) de modelos y la inferencia con modelos preentrenados.

El notebook principal es `Spark_MLOps (1).ipynb`.

## Contenido y Flujo del Código del Notebook

El notebook se divide en varias etapas clave, desde la configuración del entorno hasta el análisis de datos y la aplicación de modelos de Machine Learning.

### 1. Configuración del Entorno Spark
* **Instalación de Dependencias (Celda [1]):**
    ```python
    !pip install pyspark
    !pip install findspark
    ```
    Se instalan las bibliotecas `pyspark` y `findspark` para permitir la interacción con Apache Spark desde Python y facilitar la localización de la instalación de Spark.

* **Importación e Inicialización (Celda [5]):**
    ```python
    import findspark
    findspark.init()
    from pyspark import SparkContext, SparkConf
    from pyspark.sql import SparkSession
    ```
    Se inicializa `findspark` y se importan los componentes esenciales de PySpark: `SparkContext` para la funcionalidad central de Spark y `SparkSession` como el punto de entrada para la API de DataFrame y SQL.

* **Creación de la Sesión Spark (Celda [6]):**
    ```python
    spark = SparkSession.builder.appName("ECommerceSearchAnalysis").getOrCreate()
    sc = spark.sparkContext
    ```
    Se crea o se obtiene una instancia de `SparkSession` con el nombre de aplicación "ECommerceSearchAnalysis". También se obtiene el `SparkContext` asociado. Se imprimen las versiones para verificación.

### 2. Análisis de Términos de Búsqueda

* **Descarga de Datos (Celda [7]):**
    ```python
    file_name = "searchterms.csv"
    url = "https://..." # URL del archivo searchterms.csv
    !wget -O {file_name} {url}
    ```
    Se descarga el conjunto de datos `searchterms.csv` desde una URL pública utilizando el comando `wget`.

* **Carga de Datos en DataFrame (Celda [8]):**
    ```python
    search_df = spark.read.csv(file_name, header=True, inferSchema=True)
    search_df.show(5)
    search_df.printSchema()
    ```
    El archivo CSV descargado se carga en un DataFrame de Spark (`search_df`). Se utiliza `header=True` para indicar que la primera fila contiene los nombres de las columnas e `inferSchema=True` para que Spark determine automáticamente los tipos de datos. Se muestra una muestra y el esquema para verificar la carga. El DataFrame resultante contiene las columnas: `day` (integer), `month` (integer), `year` (integer), y `searchterm` (string).

* **Exploración Básica del DataFrame (Celdas [9], [10], [11]):**
    * Se calcula y muestra el número de filas y columnas (10000 filas, 4 columnas).
    * Se muestran las primeras 5 filas nuevamente.
    * Se confirma que el tipo de dato de la columna `searchterm` es `StringType`.

* **Consultas Analíticas (Celdas [13], [14]):**
    * **Conteo de Término Específico:** Se cuenta cuántas veces se buscó el término "gaming laptop". El nombre de la columna inferido por Spark fue `searchterm` (minúsculas, sin espacio).
        ```python
        from pyspark.sql.functions import col
        count_gaming_laptop = search_df.filter(col("searchterm") == "gaming laptop").count()
        # Salida: El término 'gaming laptop' fue buscado 499 veces.
        ```
    * **Top 5 Términos Más Frecuentes:** Se identifican los 5 términos de búsqueda más utilizados.
        ```python
        from pyspark.sql.functions import desc
        top_5_search_terms = search_df.groupBy("searchterm").count().orderBy(desc("count")).limit(5)
        top_5_search_terms.show(truncate=False)
        # Salida (ejemplo):
        # +-------------+-----+
        # |searchterm   |count|
        # +-------------+-----+
        # |mobile 6 inch|2312 |
        # |mobile 5g    |2301 |
        # |mobile latest|1327 |
        # |laptop       |935  |
        # |tablet wifi  |896  |
        # +-------------+-----+
        ```

### 3. Uso de un Modelo de Pronóstico de Ventas Preentrenado

* **Descarga y Extracción del Modelo (Celda [15]):**
    ```python
    model_file_name = "model.tar.gz"
    model_url = "https://..." # URL del modelo .tar.gz
    !wget -O {model_file_name} {model_url}
    model_dir_name = "sales_forecast_model_dir"
    !mkdir -p {model_dir_name}
    !tar -xzf {model_file_name} -C {model_dir_name}
    ```
    Se descarga un archivo `model.tar.gz` que contiene un modelo preentrenado y se extrae en el directorio `sales_forecast_model_dir`. La inspección del contenido extraído reveló que el modelo real residía en una subcarpeta llamada `sales_prediction.model`.

* **Carga del Modelo SparkML (Celda [18]):**
    ```python
    from pyspark.ml.regression import LinearRegressionModel
    path_to_model_files = f"{model_dir_name}/sales_prediction.model"
    sales_model = LinearRegressionModel.load(path_to_model_files)
    # ... (prints de coeficientes e intercepto)
    ```
    Se carga el modelo. Inicialmente se intentó cargar como `PipelineModel`, pero los metadatos indicaron que en realidad era un `LinearRegressionModel`. La carga fue exitosa utilizando la clase correcta.

* **Predicción de Ventas para 2023 (Celda [20]):**
    ```python
    from pyspark.ml.feature import VectorAssembler
    year_to_predict_data = [(2023,)]
    predict_df_raw = spark.createDataFrame(year_to_predict_data, ["year_feature"])
    assembler_pred = VectorAssembler(inputCols=["year_feature"], outputCol="features") # Corregido outputCol a "features"
    predict_df_assembled = assembler_pred.transform(predict_df_raw)
    # Se seleccionó solo la columna 'features' para la predicción
    predict_df_final = predict_df_assembled.select("features") 
    forecast = sales_model.transform(predict_df_final)
    forecast.select("prediction").show()
    # Salida:
    # +------------------+
    # |        prediction|
    # +------------------+
    # |175.16564294006457|
    # +------------------+
    ```
    Se prepara un DataFrame con el año 2023 como característica de entrada. Se utiliza `VectorAssembler` para crear la columna `features` requerida por el `LinearRegressionModel`. Finalmente, se realiza la predicción de ventas.

## Conclusiones del Proyecto

1.  **Análisis de Datos con PySpark:** PySpark demostró ser una herramienta eficaz para cargar, procesar y analizar el conjunto de datos de términos de búsqueda. Operaciones como `filter`, `groupBy`, `count`, y `orderBy` son intuitivas y potentes para extraer información valiosa.
2.  **Manejo de Nombres de Columna:** Se observó que Spark, al inferir el esquema de un CSV con encabezados como "Search Term", lo convierte a un formato más manejable programáticamente como "searchterm". Es crucial verificar los nombres de columna reales en el DataFrame antes de realizar operaciones sobre ellos.
3.  **Persistencia y Carga de Modelos (MLOps Básico):**
    * El proceso de guardar y cargar modelos es fundamental. El notebook (aunque no mostraba el guardado de este modelo específico, sino en un ejercicio previo del laboratorio) y la carga del modelo preentrenado ilustran este concepto.
    * La correcta identificación del **tipo de modelo guardado** (ej. `LinearRegressionModel` vs. `PipelineModel`) es esencial para una carga exitosa. Los metadatos del modelo son la fuente de esta información.
    * La **ruta correcta** al directorio del modelo (después de la extracción de archivos comprimidos) también es un detalle crítico.
4.  **Preparación de Datos para Inferencia:** Para utilizar un modelo cargado, los datos de entrada deben ser preprocesados exactamente de la misma manera que los datos con los que el modelo fue entrenado. En este caso, se requirió el uso de `VectorAssembler` para crear la columna `features` a partir del año.
5.  **Utilidad de Modelos Preentrenados:** Poder cargar y utilizar modelos preentrenados ahorra un tiempo y recursos considerables, permitiendo aplicar rápidamente soluciones de ML a nuevos datos.

## Aplicaciones en la Vida Real

Las técnicas y procesos demostrados en este notebook tienen numerosas aplicaciones en escenarios del mundo real, especialmente en el contexto de MLOps:

1.  **Optimización de Marketing y SEO:** Analizar los términos de búsqueda más frecuentes ayuda a las empresas de comercio electrónico a entender qué buscan sus clientes, optimizar sus listados de productos para motores de búsqueda (SEO), y dirigir campañas de marketing más efectivas (SEM).
2.  **Mejora de la Experiencia del Usuario (UX):** Conocer los términos de búsqueda populares puede guiar el diseño de la navegación del sitio web, la categorización de productos y la funcionalidad de búsqueda interna para hacerla más intuitiva y eficiente.
3.  **Gestión de Inventario y Demanda:** Identificar tendencias en las búsquedas (ej. "gaming laptop") puede ayudar a predecir la demanda de ciertos productos y optimizar los niveles de inventario.
4.  **Pronóstico de Ventas:** El uso de modelos de pronóstico de ventas, como el cargado en el notebook, es vital para la planificación financiera, la asignación de recursos, la fijación de objetivos y la toma de decisiones estratégicas en cualquier negocio.
5.  **Operacionalización de Modelos (MLOps):**
    * **Automatización:** La capacidad de guardar, cargar y ejecutar modelos programáticamente es la base para automatizar pipelines de ML, donde los modelos se reentrenan y despliegan continuamente.
    * **Escalabilidad:** PySpark permite procesar grandes volúmenes de datos para el análisis y el entrenamiento/inferencia de modelos, lo cual es crucial para aplicaciones a gran escala.
    * **Monitoreo y Reentrenamiento:** Aunque no se implementó aquí, en un sistema MLOps completo, el rendimiento del modelo de pronóstico de ventas se monitorearía continuamente. Si su precisión disminuye (deriva del modelo), se activaría un proceso de reentrenamiento con datos más recientes.
    * **Reproducibilidad:** Guardar modelos y versionar el código de preparación de datos y entrenamiento asegura que los resultados puedan ser reproducidos y auditados.

En resumen, este proyecto, aunque básico, toca muchos puntos clave del análisis de datos moderno y las prácticas de MLOps que son esenciales para que las empresas aprovechen el poder de sus datos y modelos de Machine Learning de manera eficiente y escalable.

## Cómo Ejecutar

1.  **Entorno:**
    * Jupyter Notebook o JupyterLab con Python.
    * Apache Spark y Java instalados y configurados.
2.  **Instalación:**
    * Ejecutar la primera celda del notebook para instalar `pyspark` y `findspark` si es necesario.
3.  **Ejecución:**
    * Ejecutar las celdas del notebook `Spark_MLOps (1).ipynb` en orden. Los archivos de datos (`searchterms.csv`) y el modelo (`model.tar.gz`) se descargan y procesan dentro del notebook.
