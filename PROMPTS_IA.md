# Origen del código y uso de IA generativa

**Herramienta:** Codex de OpenAI, modelo GPT-6 (según la información de esta sesión).  
**Fecha:** 9 de octubre de 2026.  
**Autoría:** los tres notebooks, este README y el presente registro fueron generados con asistencia de Codex. La estructura y la selección de datasets partieron de la planeación facilitada por el usuario; se ajustaron criterios de evaluación para evitar fuga de datos y se documentaron las fuentes originales.

## Prompts del usuario que guiaron el trabajo

1. «Implementar en c++, python o java, 1 ejemplo de redes neuronales de tipo pattern recognition (clasificación) para un problema de clasificación real, comercial o industrial [...] con sus parámetros ajustados y con resultados aceptables. Implementar [...] fitting (regresión) [...] Implementar [...] series de tiempo para un problema de clasificación real o industrial [...] Se permite usar frameworks, generar los códigos con IA generativa (poner su origen y los prompts usados). Los enlaces deben permitir correr dicho programa y contener las instrucciones de su uso.»
2. «En el archivo que te dí describe la planeación del proyecto que estoy trabajando puedes realizarlo [...]»
3. «Python con notebooks ejecutables en Google Colab [...] siempre y cuando solo con link se puede ejecutar sin necesidad de descargar nada».
4. «Revisa la planeación».
5. «El asistente es secundario, si es posible perfecto [...] empieza por favor con la implementación».

## Instrucciones técnicas derivadas de esos prompts

Codex preparó tres notebooks independientes con descarga automática de Bank Marketing, Energy Efficiency y FordA; implementó MLP para los dos primeros y CNN 1D para FordA; añadió preparación de datos, comparación de dos arquitecturas con validación, parada temprana, métricas sobre prueba, gráficas, instrucciones de uso y una predicción de ejemplo. Las métricas que aparezcan tras ejecutar los notebooks son resultados computados, no cifras aportadas por la IA.

## Fuentes técnicas consultadas

- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- [Keras: clasificación de series temporales desde cero con FordA](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)

El código se escribió para este proyecto; las fuentes anteriores sustentan los datos y el enfoque, y no se copiaron notebooks completos de terceros.
