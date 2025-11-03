# Deteccion-de-Noticias-Falsas-con-PLN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Aprendizaje%20Automático-orange)
![NLTK](https://img.shields.io/badge/NLTK-Tokenización-green)
![Estado](https://img.shields.io/badge/Estado-Completado-success)

Proyecto personal que implementa un sistema completo de **detección de noticias falsas en español**, utilizando técnicas clásicas de **Procesamiento de Lenguaje Natural (PLN)** y **aprendizaje automático supervisado**.  
El objetivo es construir un pipeline reproducible capaz de clasificar noticias como verdaderas o falsas a partir de su texto.

---

## Características principales

- Pipeline completo: desde la recolección de datos hasta la evaluación de modelos  
- Limpieza, normalización y tokenización del texto en español  
- Vectorización con **Bag-of-Words (BoW)** y **TF-IDF**  
- Comparación de modelos clásicos de machine learning  
- Evaluación con métricas de precisión, recall y F1-score  
- Estructura modular y reproducible  

## Descripción general del proyecto

### 1. Integración del conjunto de datos

Se combinaron dos fuentes heterogéneas para construir un corpus balanceado de **7.200 noticias** (50 % verdaderas y 50 % falsas):

- Noticias recolectadas mediante **web scraping** de portales verificados sobre temas de salud.  
- Dataset público de Kaggle:  
  [`Spanish Political Fake News`](https://www.kaggle.com/datasets/javieroterovizoso/spanish-political-fake-news)

Cada registro del corpus tiene la siguiente estructura:
```json
{
  "id": "string",
  "label": 0,      // 0 = falsa, 1 = verdadera
  "title": "string",
  "body": "string"
}
```

## 2. Preprocesamiento

Se aplicaron diversas etapas de limpieza y normalización:

- Conversión a minúsculas y normalización de acentos (`áéíóúñ → aeioun`)
- Eliminación de números, URLs, etiquetas HTML y signos de puntuación  
- Tokenización con **NLTK**  
- Eliminación de *stopwords* en español  
- *Stemming* con **SnowballStemmer**

El texto limpio se almacena en la columna `text_clean` y se exporta como `corpus_preprocesado.csv`.

---

## 3. Extracción y ponderación de características

Los textos se vectorizan mediante dos enfoques principales:

- **BoW / TO (Term Occurrence)** – conteo de términos  
- **TF-IDF (Term Frequency–Inverse Document Frequency)**  

Para evaluar el impacto de las técnicas de reducción léxica, se probaron **ocho configuraciones** diferentes que combinan el uso de *stopwords*, *stemming* y el tipo de ponderación (BoW o TF-IDF):

1. Sin eliminación de stopwords ni stemming, usando BoW.  
2. Eliminando stopwords, sin stemming, usando BoW.  
3. Sin stopwords pero aplicando stemming, usando BoW.  
4. Combinando eliminación de stopwords y stemming, usando BoW.  
5. Sin eliminación de stopwords ni stemming, usando TF-IDF.  
6. Eliminando stopwords, sin stemming, usando TF-IDF.  
7. Sin stopwords pero aplicando stemming, usando TF-IDF.  
8. Combinando eliminación de stopwords y stemming, usando TF-IDF.  

Cada configuración generó su propia matriz esparsa y vocabulario asociado, lo que permitió analizar el efecto de cada técnica de preprocesamiento sobre el rendimiento de los modelos.

---

## 4. Entrenamiento y evaluación de modelos

Se entrenan y comparan cuatro algoritmos supervisados:

- **Regresión Logística**
- **Árboles de Decisión**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**

### Protocolo

- División de datos: **80 %** entrenamiento / **20 %** prueba  
- Validación cruzada: **k=10** (métrica principal: F1 ponderado)

Los resultados mostraron que:

- La **Regresión Logística** fue el modelo con mejor desempeño general, alcanzando un **F1-score de 0.885** cuando se utilizó **TF-IDF** y eliminación de *stopwords*.  
- El **SVM** obtuvo resultados similares, con un **F1-score cercano a 0.867**, también con TF-IDF.  
- Los **Árboles de Decisión** lograron un rendimiento aceptable (**F1 ≈ 0.86**) usando BoW.  
- El modelo **KNN** tuvo el rendimiento más bajo (**F1 ≈ 0.78**), aunque mejoró al usar TF-IDF.  

---

## Principales hallazgos

- La representación **TF-IDF** mejora la estabilidad y el rendimiento general.  
- La **eliminación de stopwords** incrementa el desempeño de forma consistente.  
- El **stemming** aporta mejoras limitadas debido a la morfología del español.  
- El mejor equilibrio entre precisión y eficiencia se obtuvo con **Regresión Logística + TF-IDF + stopwords**.  

**F1-score máximo alcanzado:** 0.885

---

