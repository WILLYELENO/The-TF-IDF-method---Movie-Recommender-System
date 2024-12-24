# 📽️ Librería para Graficar Similitudes de Películas

Este proyecto proporciona herramientas para analizar y graficar similitudes entre películas utilizando la técnica TF-IDF para vectorización del texto y cálculos de similitud.

## 🚀 Características

- Vectorización de texto usando TF-IDF.
- Cálculo de similitud entre películas usando similitud del coseno.
- Visualización gráfica de similitudes.
- Recomendación de películas basadas en similitud.

## 📋 Requisitos

- `pandas`
- `scikit-learn`
- `matplotlib`

Instalamos los requisitos utilizando pip:

```bash
pip install pandas scikit-learn matplotlib


📁 Estructura del Proyecto

movie_metadata.csv: Archivo CSV con datos de películas.
main.py: Script principal para procesar datos, calcular similitudes y generar gráficos.
functions.py: Archivo que contiene funciones auxiliares para el cálculo de valores TF-IDF y similitudes.

🛠️ Uso
1. Carga y Preprocesamiento de Datos

import pandas as pd

# Leemos el archivo CSV y lo convertimos en un DataFrame
df = pd.read_csv('db/movie_metadata.csv')

# Reemplazamos caracteres innecesarios
df['genero'] = df['genero'].str.replace('|', ' ')
df['plot_keywords'] = df['plot_keywords'].str.replace('|', ' ')

# Creamos una nueva columna combinando 'genero' y 'plot_keywords'
df['texto'] = df[['genero', 'plot_keywords']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

2. Vectorización y Cálculo de Similitud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicializamos el vectorizador TF-IDF
tfidf = TfidfVectorizer(max_features=2000)

# Vectorizamos el texto
X = tfidf.fit_transform(df['texto'])

# Consultamos la similitud del coseno
similitud = cosine_similarity(X[indice], X)

3. Visualización de Similitudes

import matplotlib.pyplot as plt

# Graficamos la similitud
plt.figure()
plt.plot(similitud.flatten())
plt.xlabel('Película')
plt.ylabel('Similitud')
plt.title('Similitud de películas con The Dark Knight Rises')
plt.savefig('similitud.png')
plt.show()

🔧 Funciones
get_map_vocabulary_and_idfVaules(model)
Obtiene un diccionario que mapea términos a sus valores IDF desde un modelo TfidfVectorizer.

get_terms_from_vector(vector, model)
Genera un diccionario con términos presentes en un vector TF-IDF y sus respectivos valores.


🧩 Ejemplo de Uso

import functions

# Obtener términos y valores IDF
idf_dict = functions.get_map_vocabulary_and_idfVaules(tfidf)

# Obtener términos y valores TF-IDF desde un vector
term_values = functions.get_terms_from_vector(consulta, tfidf)


¡Claro! Aquí tienes un archivo README.md completo que incluye todo el contenido, diseñado para proporcionar una visión completa de tu proyecto y sus características:

markdown
Copiar código
# 📽️ Librería para Graficar Similitudes de Películas

Este proyecto proporciona herramientas para analizar y graficar similitudes entre películas utilizando la técnica TF-IDF para vectorización del texto y cálculos de similitud.

## 🚀 Características

- Vectorización de texto usando TF-IDF.
- Cálculo de similitud entre películas usando similitud del coseno.
- Visualización gráfica de similitudes.
- Recomendación de películas basadas en similitud.

## 📋 Requisitos

- `pandas`
- `scikit-learn`
- `matplotlib`

Instala los requisitos utilizando pip:

```bash
pip install pandas scikit-learn matplotlib
📁 Estructura del Proyecto
movie_metadata.csv: Archivo CSV con datos de películas.
main.py: Script principal para procesar datos, calcular similitudes y generar gráficos.
functions.py: Archivo que contiene funciones auxiliares para el cálculo de valores TF-IDF y similitudes.
🛠️ Uso
Carga y Preprocesamiento de Datos

python
Copiar código
import pandas as pd

# Leemos el archivo CSV y lo convertimos en un DataFrame
df = pd.read_csv('db/movie_metadata.csv')

# Reemplazamos caracteres innecesarios
df['genero'] = df['genero'].str.replace('|', ' ')
df['plot_keywords'] = df['plot_keywords'].str.replace('|', ' ')

# Creamos una nueva columna combinando 'genero' y 'plot_keywords'
df['texto'] = df[['genero', 'plot_keywords']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
Vectorización y Cálculo de Similitud

python
Copiar código
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicializamos el vectorizador TF-IDF
tfidf = TfidfVectorizer(max_features=2000)

# Vectorizamos el texto
X = tfidf.fit_transform(df['texto'])

# Consultamos la similitud del coseno
similitud = cosine_similarity(X[indice], X)
Visualización de Similitudes

python
Copiar código
import matplotlib.pyplot as plt

# Graficamos la similitud
plt.figure()
plt.plot(similitud.flatten())
plt.xlabel('Película')
plt.ylabel('Similitud')
plt.title('Similitud de películas con The Dark Knight Rises')
plt.savefig('similitud.png')
plt.show()
🔧 Funciones
get_map_vocabulary_and_idfVaules(model)
Obtiene un diccionario que mapea términos a sus valores IDF desde un modelo TfidfVectorizer.

get_terms_from_vector(vector, model)
Genera un diccionario con términos presentes en un vector TF-IDF y sus respectivos valores.

🧩 Ejemplo de Uso
python
Copiar código
import functions

# Obtener términos y valores IDF
idf_dict = functions.get_map_vocabulary_and_idfVaules(tfidf)

# Obtener términos y valores TF-IDF desde un vector
term_values = functions.get_terms_from_vector(consulta, tfidf)

📊 Análisis de Resultados
Términos y Valores IDF: Proporciona una visión de la importancia de cada término en el corpus de texto.
Similitud del Coseno: Calcula qué tan similares son las películas entre sí en función de su contenido textual.
Visualización: Genera gráficos para entender la distribución de similitudes y ayuda en la recomendación de películas.

Resultados Esperados
Valores TF-IDF: Muestra la importancia relativa de cada término para un documento específico.
Similitud: Los valores indican la similitud entre la película consultada y las demás.


🤝 Contribuir
Haz un fork del proyecto.
Crea una rama (git checkout -b feature/mi-nueva-caracteristica).
Realiza tus cambios y realiza un commit (git commit -am 'Añadir nueva característica').
Envía tus cambios a la rama principal (git push origin feature/mi-nueva-caracteristica).
Abre una pull request.