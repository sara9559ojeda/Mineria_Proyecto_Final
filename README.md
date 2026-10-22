# Proyecto final · Minería de datos
## Segmentación de clientes con K-Means (Mall Customers)

Modelo de segmentación de clientes con K-Means, evaluado con el método del codo, el coeficiente de silueta y el índice de Davies-Bouldin, con interpretación de negocio y despliegue en Streamlit.

## Resultados principales

| Métrica | Valor |
|---|---|
| Variables de segmentación | Ingreso anual (k$) y puntaje de gasto (1-100), estandarizados |
| Número de clusters (K) | **5** (coinciden el codo, la silueta y Davies-Bouldin) |
| Coeficiente de silueta | **0.555** (estructura razonable) |
| Davies-Bouldin | **0.572** |

| Perfil | Clientes | Ingreso medio | Gasto medio |
|---|---|---|---|
| Clientes de alto valor | 39 (19.5 %) | 86.5 k$ | 82.1 |
| Clientes promedio | 81 (40.5 %) | 55.3 k$ | 49.5 |
| Clientes con potencial | 35 (17.5 %) | 88.2 k$ | 17.1 |
| Compradores impulsivos | 22 (11.0 %) | 25.7 k$ | 79.4 |
| Clientes de bajo valor | 23 (11.5 %) | 26.3 k$ | 20.9 |

**Mayor oportunidad de crecimiento:** los *clientes con potencial*. Tienen el mismo ingreso que los de alto valor, pero gastan 65 puntos menos.

## Estructura

```
├── Mall_Customers.csv                  # Dataset original (Kaggle)
├── proyecto_final_kmeans.ipynb         # Partes 1-7 del taller, con las respuestas en celdas separadas
├── app.py                              # Aplicación Streamlit (Parte 7)
├── models/kmeans_segmentation.joblib   # Pipeline entrenado (StandardScaler + K-Means) + métricas + perfiles
├── outputs/mall_customers_segmented.csv# Clientes con su cluster y perfil
├── DOCUMENTACIÓN_TÉCNICA.txt           # Documentación técnica (plantilla diligenciada + preguntas y respuestas)
├── requirements.txt                    # Dependencias de la app (las usa Streamlit Community Cloud)
├── requirements-notebook.txt           # Dependencias adicionales para ejecutar el cuaderno
└── .streamlit/config.toml              # Tema de la aplicación
```

## Cómo ejecutarlo

1. Instalar las dependencias. Basta con `requirements.txt` para la app; para el cuaderno, usar `requirements-notebook.txt`:

   ```bash
   pip install -r requirements-notebook.txt
   ```

2. (Opcional) Volver a ejecutar el cuaderno para regenerar el modelo y las salidas:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace proyecto_final_kmeans.ipynb
   ```

3. Abrir la aplicación:

   ```bash
   streamlit run app.py
   ```

   La aplicación abre en `http://localhost:8501` y tiene tres secciones:
   - **Método del codo:** inercia y silueta por K, con el K elegido marcado.
   - **Clusters:** mapa interactivo de los segmentos con sus centroides, resumen por perfil y estrategias de marketing.
   - **Clasificar cliente:** se ingresan el ingreso anual y el puntaje de gasto, y el modelo asigna automáticamente el cluster, el perfil y la estrategia recomendada.

> El modelo se guardó con scikit-learn 1.9.1. Si se usa otra versión, conviene volver a ejecutar el cuaderno (paso 2) antes de abrir la app.

## Despliegue en Streamlit Community Cloud

1. Entrar a [share.streamlit.io](https://share.streamlit.io) e iniciar sesión con la cuenta de GitHub dueña de este repositorio.
2. **Create app** → **Deploy a public app from GitHub** y completar:
   - **Repository:** `sara9559ojeda/Mineria_Proyecto_Final`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** un subdominio libre, por ejemplo `segmentacion-clientes-kmeans`
3. En **Advanced settings**, elegir **Python 3.12**. Las versiones fijadas en `requirements.txt` no existen para Python 3.11.
4. Presionar **Deploy**. Streamlit instala `requirements.txt`, aplica el tema de `.streamlit/config.toml` y publica la app en `https://<subdominio>.streamlit.app`.

Cada `git push` a `main` actualiza la app desplegada automáticamente.
