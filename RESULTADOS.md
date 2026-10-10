# Resultados verificados

Los tres notebooks se diseñaron para descargar sus datos y recalcular las métricas al ejecutar **Entorno de ejecución → Ejecutar todo**. Los valores de esta página proceden de una ejecución local en CPU con TensorFlow 2.21.0 y semilla 42, realizada el 9 de octubre de 2026. Puede haber pequeñas variaciones en Colab por diferencias de versión y hardware.

| Tarea | Arquitectura elegida por validación | Prueba | Referencia sencilla |
|---|---|---|---|
| Clasificación bancaria | MLP 64/32, umbral 0.63 | ROC-AUC **0.7922**, PR-AUC **0.4379**, F1 clase positiva **0.48**, recall **0.50** | PR-AUC por prevalencia: 0.117 |
| Regresión energética | MLP 64/32 | MAE **0.493**, RMSE **0.656**, R² **0.9959** | Predecir la media: MAE 9.278, RMSE 10.258 |
| FordA | CNN 1D, filtros 64/64 | Accuracy **0.8992**, F1 **0.8947** | Accuracy de clase mayoritaria: 0.5159 |

## Lectura de los resultados

En Bank Marketing, la clase positiva representa alrededor del 11.7 % de los registros. Por eso se usaron PR-AUC, F1, precisión y recall además de ROC-AUC. El ROC-AUC de prueba quedó ligeramente por debajo de 0.80, que en el PDF de planeación se proponía como meta orientativa; la PR-AUC supera con claridad la prevalencia de positivos. La columna `duration` se excluyó porque no estaría disponible antes de llamar al cliente.

En Energy Efficiency, los 768 casos son simulaciones de diseños de edificios. El R² alto demuestra buen ajuste dentro de este conjunto, pero no valida automáticamente el desempeño en edificios reales.

FordA utiliza los archivos TRAIN y TEST originales. Sus dos etiquetas se presentan como clase 0 y clase 1; no se infiere cuál representa una avería concreta.
