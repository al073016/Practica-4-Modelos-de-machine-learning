# Proyecto de redes neuronales: tres problemas independientes

Este proyecto contiene tres notebooks de Python preparados para Google Colab. Cada uno descarga su dataset al ejecutar la primera celda y no requiere que el usuario cargue archivos. Para empezar, pulsa uno de los enlaces **Ejecutar en Colab** de la tabla y en Colab selecciona **Entorno de ejecución → Ejecutar todo**. Puedes ejecutar las celdas una por una para seguir el proceso. Es necesario tener conexión a internet para la descarga y un navegador compatible con Colab. No hace falta GPU; en CPU el entrenamiento puede tardar varios minutos.

| Notebook | Ejecutar | Dataset | Objetivo | Red | Métricas principales |
|---|---|---|---|---|---|
| `01_Clasificacion_Bancaria.ipynb` | [Ejecutar en Colab](https://colab.research.google.com/github/al073016/Practica-4-Modelos-de-machine-learning/blob/main/01_Clasificacion_Bancaria.ipynb) | [Bank Marketing, UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) | Predecir suscripción a depósito antes de la llamada | MLP | ROC-AUC, PR-AUC, precisión, recall, F1 |
| `02_Regresion_Energetica.ipynb` | [Ejecutar en Colab](https://colab.research.google.com/github/al073016/Practica-4-Modelos-de-machine-learning/blob/main/02_Regresion_Energetica.ipynb) | [Energy Efficiency, UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | Estimar la carga de calefacción de un diseño de edificio | MLP | MAE, RMSE, R² |
| `03_Series_Tiempo_FordA.ipynb` | [Ejecutar en Colab](https://colab.research.google.com/github/al073016/Practica-4-Modelos-de-machine-learning/blob/main/03_Series_Tiempo_FordA.ipynb) | [FordA, UCR/UEA](https://www.timeseriesclassification.com/description.php?Dataset=FordA) | Clasificar señales temporales de ruido de motor | CNN 1D | Accuracy, F1, matriz de confusión |

## Cómo se usan

1. Abre el notebook correspondiente mediante el enlace **Ejecutar en Colab** de la tabla.
2. Elige **Entorno de ejecución → Ejecutar todo**. Acepta la conexión al entorno si Colab la solicita. La descarga se hace automáticamente desde UCI o desde los archivos FordA usados por la guía oficial de Keras.
3. Lee las tablas y gráficas de la sección **Evaluación final en prueba**. Las métricas se calculan al ejecutar el notebook; no se presentan como cifras garantizadas.
4. Cambia `indice` en la última celda y ejecútala para ver otra predicción de ejemplo. Si reinicias el entorno, ejecuta todas las celdas de nuevo.

Cada notebook hace una división de entrenamiento, validación y prueba. Compara dos configuraciones de red y selecciona la mejor mediante validación. Los datos de prueba se usan una sola vez, para la evaluación final. En clasificación bancaria, `duration` se excluye porque no está disponible antes de realizar la llamada. En regresión, los datos son simulaciones de edificios. En FordA, se conservan los archivos originales TRAIN y TEST; la correspondencia semántica de sus etiquetas −1/+1 no se da por supuesta.

## Requisitos y fuentes

Google Colab suele traer TensorFlow, NumPy, pandas, scikit-learn y Matplotlib. La primera celda instala automáticamente cualquier biblioteca ausente, incluida `openpyxl`. Los datasets Bank Marketing y Energy Efficiency están publicados por UCI bajo CC BY 4.0; las páginas enlazadas contienen la atribución y las condiciones. El notebook FordA usa los archivos [TRAIN](https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv) y [TEST](https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv) de la [guía de Keras](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/).

Consulta `RESULTADOS.md` para ver las métricas medidas, y `PROMPTS_IA.md` para conocer el origen del código y los prompts utilizados. El informe académico se elaborará después según el formato solicitado por el usuario.
