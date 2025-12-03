#  Proyecto Cancelaciones Olist – MAE XIII

##  Predicción de Cancelaciones de Pedidos  
Proyecto Final – Maestría en Data Science  

**Autor:** Carlos Fernando Mamani Quispe  
**Arquitectura:** Medallion (Bronze → Silver → Gold)  
**Tecnologías:** Python, Pandas, Scikit-Learn, XGBoost, Streamlit

---

#  1. Descripción General

Este proyecto implementa un *pipeline* completo y modular para analizar, procesar y predecir cancelaciones de pedidos utilizando el dataset público de **Olist**.  

El flujo sigue la arquitectura **Medallion** (Bronze/Silver/Gold) e incluye:

- Limpieza y estandarización de datos  
- Ingeniería de características  
- Selección de variables  
- Entrenamiento de modelos (XGBoost, Logistic Regression, Random Forest)  
- Simulación de nuevos meses  
- MVP funcional con **Streamlit**  
- Archivo de configuración con parámetros `.env`

---

#  2. Estructura del Proyecto

<img width="216" height="658" alt="image" src="https://github.com/user-attachments/assets/3a5db139-5097-4729-be6e-4145ca5a4ebb" />


#  3. Configuración – Archivo `.env`

<img width="411" height="194" alt="image" src="https://github.com/user-attachments/assets/d0add1ca-04a5-4279-98b6-11ed32f2201d" />


#  4. Pipeline de Procesamiento

## **Bronze Layer**
- Limpieza general  
- Tipificación y parseo de fechas  
- Normalización de nombres  
- Eliminación de duplicados  

## **Silver Layer**
- Feature Engineering  
- Variables temporales (week, month, lag, delays)  
- Variables categóricas y numéricas  
- Agregaciones por cliente y vendedor  

## **Gold Layer**
- Selección de variables  
- Preparación X e y  
- Entrenamiento con modelos ML  
- Métricas finales  
- Exportación del modelo en `.pkl`  

---

#  5. Ejecución del Pipeline Completo

```bash
python main_pipeline.py


