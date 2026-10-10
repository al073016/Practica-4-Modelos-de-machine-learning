# Aplicación interactiva de la Práctica 4

Esta aplicación **sí ejecuta Python**. El botón **Entrenar red neuronal** descarga el dataset público, prepara entrenamiento/validación/prueba, entrena dos arquitecturas y elige la mejor por validación. A continuación muestra métricas calculadas en prueba. El botón **Predecir este caso** llama al modelo de la sesión para el índice elegido, incluso si se cambia muchas veces.

Los tres modelos son los de los notebooks del repositorio: MLP para Bank Marketing, MLP para Energy Efficiency y CNN 1D para FordA. Las semillas, divisiones, arquitecturas, criterios de selección y métricas se mantuvieron. Las cifras pueden variar ligeramente entre versiones de TensorFlow o plataformas.

## Publicar para entregar un enlace

1. Sube la carpeta `app_web` completa a la raíz del repositorio público `al073016/Practica-4-Modelos-de-machine-learning`. Deben quedar `app_web/app.py`, `app_web/modelos.py`, `app_web/requirements.txt` y `app_web/README.md`. Si usas el ZIP entregado, extráelo primero y sube su contenido; no subas únicamente el ZIP. Actualiza también los tres notebooks, `README.md`, `RESULTADOS.md` y `PROMPTS_IA.md` de la raíz.
2. Entra a [Streamlit Community Cloud](https://share.streamlit.io/) con tu cuenta de GitHub. Conecta el repositorio público cuando el servicio lo solicite.
3. Crea una app desde ese repositorio, en la rama `main`, usando `app_web/app.py` como archivo principal. Elige **Python 3.11** en las opciones avanzadas si aparece la opción.
4. Espera la instalación de dependencias. Copia la dirección pública `https://...streamlit.app` que muestre el servicio. Ese será el enlace único que permite **entrenar y probar en la misma página**, sin descargas locales.

GitHub Pages no puede ejecutar esta aplicación Python; el sitio estático anterior puede quedar como material complementario. El entrenamiento en un servidor compartido puede tardar varios minutos, especialmente FordA, y la app podría reiniciarse si alcanza los límites de recursos de la plataforma. Al reiniciarse, basta con pulsar **Entrenar** de nuevo.

## Uso

1. Elige **Clasificación bancaria**, **Regresión energética** o **Series de tiempo FordA**.
2. Pulsa **Entrenar red neuronal** y espera a que aparezca **Entrenamiento y evaluación completados**.
3. Revisa las métricas, la pérdida y la matriz de confusión o la gráfica real/estimado.
4. Introduce un índice válido del conjunto de prueba y pulsa **Predecir este caso**. Puedes repetir este paso sin volver a entrenar.

Los notebooks de Google Colab siguen disponibles para revisar y ejecutar cada paso del código. Las fuentes originales y los prompts usados con IA están documentados en el README principal y en `PROMPTS_IA.md`.

## Ejecutar localmente (opcional)

Con Python 3.11 instalado, desde la raíz del repositorio:

```bash
pip install -r app_web/requirements.txt
streamlit run app_web/app.py
```
