# pyspark-ecommerce-search-forecast-mlops
-->Un proyecto de prueba de concepto que demuestra el análisis de términos de búsqueda de un sitio de comercio electrónico y el uso de un modelo preentrenado de pronóstico de ventas utilizando PySpark. Este repositorio explora elementos básicos de MLOps como el guardado y carga de modelos. <3

# Análisis de Términos de Búsqueda y Pronóstico de Ventas con PySpark (PoC MLOps)

Este repositorio contiene un notebook de Jupyter (`Spark_MLOps (1).ipynb`) que demuestra un flujo de trabajo básico de Machine Learning y análisis de datos utilizando PySpark. El proyecto cubre:
1.  Análisis de un conjunto de datos de términos de búsqueda de un servidor web de comercio electrónico.
2.  Creación, guardado y carga de un modelo simple de Regresión Lineal con SparkML.
3.  Carga y uso de un modelo preentrenado de pronóstico de ventas.

Este proyecto sirve como una introducción práctica a algunas operaciones fundamentales en el ciclo de vida de MLOps dentro del ecosistema Spark.

## Contenido del Notebook (`Spark_MLOps (1).ipynb`)

El notebook está dividido en dos ejercicios principales:

### Ejercicio 1: Guardado y Carga de un Modelo SparkML (Regresión Lineal)

Este ejercicio ilustra los pasos básicos para construir, entrenar, guardar y cargar un modelo de Machine Learning con SparkML.

**Pasos Clave:**
1.  **Instalación e Importación de Librerías:**
    * Se instalan `pyspark` y `findspark`.
    * Se importan las librerías necesarias incluyendo `SparkContext`, `SparkSession`, `VectorAssembler` y `LinearRegression`.
2.  **Creación de Sesión Spark:** Se inicializa `SparkContext` y `SparkSession`.
3.  **Creación del DataFrame:** Se utiliza un conjunto de datos de muestra (altura y peso de infantes) para crear un DataFrame de Spark.
    * `height`: Característica de entrada.
    * `weight`: Etiqueta a predecir.
4.  **Preparación de Características:** Se utiliza `VectorAssembler` para transformar la columna `height` en una columna de vector de características (`features`), requerida por los algoritmos de SparkML.
5.  **Creación y Entrenamiento del Modelo:**
    * Se crea un modelo de `LinearRegression`.
    * Se entrena el modelo (`lr.fit()`) utilizando los datos preparados.
6.  **Guardado del Modelo:** El modelo entrenado (`lrModel`) se guarda en disco utilizando `lrModel.save('infantheight2.model')`. Esto crea un directorio con los metadatos y datos del modelo.
7.  **Carga del Modelo:** Se carga el modelo previamente guardado utilizando `LinearRegressionModel.load('infantheight2.model')`.
8.  **Realización de Predicciones:** Se define una función `predict(height)` que toma una altura, la transforma en el formato esperado por el modelo, y utiliza el modelo cargado para predecir el peso. Se prueba con una altura de 70 cm.
9.  **Ejercicios de Práctica:**
    * Guardar el modelo con un nuevo nombre (`babyweightprediction.model`).
    * Cargar el modelo recién guardado.
    * Predecir el peso para una altura de 50 cm usando el modelo cargado.

**Análisis MLOps (Ejercicio 1):**
* **Control de Versiones de Modelos (Básico):** El guardado de modelos (`lrModel.save()`) es un primer paso fundamental en MLOps. Permite persistir un estado entrenado del modelo para su uso posterior o para desplegarlo. Nombrar los directorios de los modelos (ej. `infantheight2.model`, `babyweightprediction.model`) puede ser una forma rudimentaria de versionado.
* **Reutilización de Modelos:** La capacidad de cargar (`LinearRegressionModel.load()`) un modelo previamente guardado es crucial para MLOps, ya que separa el entrenamiento de la inferencia y permite que los modelos se usen en diferentes entornos o momentos.
* **Reproducibilidad (Parcial):** Al guardar el modelo, se captura el estado aprendido. Sin embargo, para una reproducibilidad completa en MLOps, también se necesitaría versionar el código de entrenamiento, los datos y el entorno.

---

### Ejercicio 2: Análisis de Términos de Búsqueda y Uso de Modelo Preentrenado

Este ejercicio se enfoca en el análisis de datos con Spark SQL/DataFrames y luego en la carga y uso de un modelo de ML preentrenado.

**Pasos Clave:**
1.  **Descarga del Conjunto de Datos:** Se descarga el archivo `searchterms.csv` que contiene datos de términos de búsqueda.
2.  **Carga de Datos en DataFrame:** El archivo CSV se carga en un DataFrame de Spark (`search_df`), infiriendo el esquema y utilizando la primera línea como encabezado. Las columnas son `day`, `month`, `year`, y `searchterm`.
3.  **Análisis Exploratorio Básico:**
    * Se imprime el número de filas y columnas del DataFrame.
    * Se muestran las primeras 5 filas.
    * Se determina el tipo de dato de la columna `searchterm` (resulta ser `StringType`).
4.  **Consultas Analíticas:**
    * Se cuenta cuántas veces se buscó el término específico "gaming laptop".
    * Se identifican y muestran los 5 términos de búsqueda más frecuentes utilizando `groupBy()`, `count()`, `orderBy()`, y `limit()`.
5.  **Descarga y Extracción de Modelo Preentrenado:**
    * Se descarga un modelo de pronóstico de ventas preentrenado (`model.tar.gz`).
    * El archivo `.tar.gz` se extrae, revelando que el modelo real está en una subcarpeta (ej. `sales_prediction.model`).
6.  **Carga del Modelo Preentrenado:**
    * Se identifica que el modelo guardado es un `LinearRegressionModel` (y no un `PipelineModel` como se podría haber intentado inicialmente).
    * Se carga el modelo utilizando `LinearRegressionModel.load()` apuntando a la ruta correcta de los archivos del modelo extraído.
7.  **Realización de Predicciones con el Modelo Cargado:**
    * Se prepara un DataFrame de entrada con el año para el cual se desea una predicción (2023).
    * Dado que es un `LinearRegressionModel`, se utiliza `VectorAssembler` para transformar el año en una columna de características vectoriales (`features`).
    * Se utiliza el modelo cargado (`sales_model.transform()`) para predecir las ventas.

**Análisis MLOps (Ejercicio 2):**
* **Uso de Modelos Preentrenados:** Cargar y utilizar modelos que han sido entrenados y validados previamente es una práctica común en MLOps. Esto ahorra tiempo de reentrenamiento y permite aprovechar modelos bien establecidos.
* **Gestión de Artefactos del Modelo:** El modelo se descarga como un archivo `.tar.gz`, que es una forma de empaquetar artefactos. La extracción y el apuntar a la ruta correcta del modelo son pasos importantes en el manejo de estos artefactos.
* **Inferencia del Esquema de Entrada del Modelo:** Un desafío clave al usar modelos preentrenados es entender el esquema de entrada que esperan. En este ejercicio, se infirió que el modelo de regresión lineal probablemente esperaba una característica de año, que luego se vectorizó. En escenarios reales, esta información vendría de la documentación del modelo o del equipo que lo entrenó.
* **Monitoreo (Implícito):** Aunque no se implementa un monitoreo activo, el acto de cargar un modelo y hacer predicciones es un precursor. En MLOps maduros, se monitorearía el rendimiento del modelo en producción y se reentrenaría según fuera necesario.
* **Desafíos de Serialización de Modelos:** El error inicial al intentar cargar el modelo como `PipelineModel` cuando en realidad era un `LinearRegressionModel` resalta la importancia de conocer el tipo exacto de objeto que fue serializado (guardado) para poder deserializarlo (cargarlo) correctamente. Los metadatos del modelo son cruciales aquí.

## Cómo Ejecutar el Notebook

1.  **Entorno:**
    * Este notebook está diseñado para ejecutarse en un entorno que tenga Apache Spark y PySpark instalados y configurados.
    * Se asume un entorno tipo Jupyter Notebook o JupyterLab.
2.  **Instalación de Dependencias:**
    * La primera celda del notebook instala `pyspark` y `findspark` si aún no están presentes:
        ```python
        !pip install pyspark
        !pip install findspark
        ```
3.  **Ejecución de Celdas:**
    * Ejecuta las celdas del notebook en orden secuencial.
    * Los archivos de datos y modelos se descargan automáticamente por el notebook en el directorio de trabajo actual.
4.  **Ajustes Potenciales:**
    * **Nombres de Columna:** Presta atención a los nombres de columna inferidos por Spark al cargar `searchterms.csv`. El código intenta manejar la diferencia entre "Search Term" (en el CSV) y "searchterm" (como lo carga Spark), pero podría necesitar ajustes si tu versión de Spark lo maneja diferente.
    * **Rutas de Modelo:** La extracción del `model.tar.gz` y la carga del modelo dependen de la estructura del archivo tar. El código asume que el modelo real está en una subcarpeta llamada `sales_prediction.model` dentro del directorio donde se extrae el tar.

## Conclusión

Este notebook proporciona una visión práctica de cómo se pueden usar PySpark y SparkML para análisis de datos básicos y operaciones de Machine Learning, tocando aspectos fundamentales relevantes para MLOps como la persistencia, carga y utilización de modelos.
