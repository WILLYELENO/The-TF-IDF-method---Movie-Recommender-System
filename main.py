
import pandas as pd

#Leemos el archivo csv y lo convertimos en un dataframe
df = pd.read_csv('db/movie_metadata.csv')

print (df)


#Reemplazamos algunos caracteres por espacios
df['genero'] = df['genero'].str.replace('|', ' ')

print (df)


df['plot_keywords'] = df['plot_keywords'].str.replace('|', ' ')

print (df)


"""Ahora vamos a juntar las columnas de genero y plot_keywords y la guardaremos en una nueva columna llamada texto

Para ello utilizamos una funcion lambda en donde por cada fila haremos un join de ambas columnas que estaran separadas por un espacio.
"""

df['texto'] = df[['genero', 'plot_keywords']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

print (df)

#Enumero todo lo que tiene
print(df['texto'].iloc[0])

#Action Adventure Fantasy Sci-Fi avatar future marine native paraplegic



row = df[['genero', 'plot_keywords', 'texto']].iloc[0]
print("ROW",row)

"""
Ya tenemos nuestra columna hecha y podremos convertir esto en una vectorizacion del tipo TF e IDF.
(VER CALCULOS DE EL EPISODIO 0)

Para representarlas,haremos una funcion...

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
"""
TfidfVectorizer = PARA VECTORIZAR
cosine_similarity =  METODO PARA SACAR SIMILITUD DEL COSENO
euclidean_distances = METODO PARA SACAR LA DISTANCIA EUCLIDIANA

"""


tfidf = TfidfVectorizer(max_features=2000) #max_features = CANTIDAD MAXIMA DE TOKEN (NO DEBE HABER UNA PELI PARA MAS DE 2000 TOKEN. Esto lo averiguamos con print(df['texto'].iloc[0]) )

# 1)Vectorizamos
#Creamos nuestra matriz de vectores sobre la columna texto
X = tfidf.fit_transform(df['texto'])

print(X)


"""NUestra matriz es de 5043 x 2000, si le poniamos mas en el max_features teniamos una matriz mas grande"""

# 2) Generamos un mapeo de las películas

"""Ahora vamos a generar un mapeo de las peliculas, un nuevo dataframe que sera una serie en realidad en donde el indice de esta serie va a ser el titulo de la pelicula (movie_tittle)"""
peliculas = pd.Series(df.index, index=df['movie_title'])

print (peliculas)

"""El error dice que no reconoce la clave, y ebe ser porque existe un espacio de mas, lo mas probable al principio o al final.

Para solucionarlo, podemos usar un strip para que saque los espacios entre los nombres de las peliculas y los indices"""

peliculas.index = peliculas.index.str.strip()
print (peliculas)


indice = peliculas['The Dark Knight Rises']

print ("Indice:",indice)

# Indice: 5039

"""
AHora haremos una consulta y averiguaremos su vector
"""

consulta = X[indice]

print ("COnsulta del vector de la pelicula The Following:", consulta)


"""Nos dice que es un vector de 2000 dimensiones"""



"""
Se ve extraño porque hizo una compresion del vector, mostrando unicamente las dimensiones que no tenian 0

Por ello, para mostrar el vector completo (la mayoria seran 0) lo convertimos en array
"""

print (consulta.toarray())
#[[0. 0. 0. ... 0. 0. 0.]]

""" Ahora vamos a calcular la similitud del coseno sobre la consulta (es el indice  de la pelicula 'The Dark Knight Rises' ) y el vector 'X' 
que es el vector con todas nuestras peliculas..."""

#3) Calculamos la similitud del coseno

similitud = cosine_similarity(consulta, X)

print("Vector X (QUE CONTIENE TODAS NUESTRAS PELICULAS):", consulta)


print (similitud)

#[[0.02767798 0.02180518 0.18356101 ... 0.05069847 0.         0.        ]]

"""Nos da el arreglo con todas las similitudes. En donde dice solo 'o', implica que no son nada parecidos

Vemos que en el elemento 4 del arreglo de similitudes, es decir  The Dark Knight Rises tiene similitud 1 con la pelicula The Dark Knight Rises"""

print (similitud[0][3])

#1.0000000000000002

"""Para evitar poniendo como matriz 0 y 3, vamos a transformar en un array la similitud"""


similitud = similitud.flatten()

print (similitud[3])

#1.0000000000000002


"""AHora vamos a graficar las similitudes para ver como se ven... para ello instalaremos la libreria matplotlib..."""

import matplotlib
matplotlib.use('Agg')  # Usa el backend Agg para guardar la imagen en un archivo
import matplotlib.pyplot as plt





# Graficamos la similitud
plt.figure()  # Crea una nueva figura
plt.plot(similitud) #Crea el grafico de la similitud
plt.xlabel('Película') #Añade una etiqueta al eje x
plt.ylabel('Similitud') #Añade una etiqueta al eje y
plt.title('Similitud de películas con The Dark Knight Rises') #Añade un titulo al grafico
plt.savefig('similitud.png')  # Guarda el gráfico en un archivo

plt.show() #Muestra el grafico en una ventana


"""UNa vez guardado el grafico, vamos a ordenar las similitudes en orden descendente para que otorgue un array con elementos (indices de peliculas)que van desde el mas parecido al menos
con la pelicula que pasamos como indice. JUstamente esta primero el 3 porque es el indice de la pelicula."""

print("Orden descendente de similitudes:",(-similitud).argsort())

orden_similitud = (-similitud).argsort()

#Orden descendente de similitudes: [   3 4139 3647 ... 2380 2363 5042]


"""Ahora graficaremos la similitud pero con este orden"""

# Graficamos la similitud en orden descendente
plt.figure()  # Crea una nueva figura
plt.plot(similitud[orden_similitud])
plt.xlabel('Película (ordenada por similitud)')
plt.ylabel('Similitud')
plt.title('Similitud de películas con The Dark Knight Rises (ordenada)')
plt.savefig('similitud_ordenada.png')  # Guarda el gráfico en un archivo

print("Gráfico guardado como similitud_ordenada.png")


"""Ahora para obtener la recomendacion de pelicula podriamos indicarles que nos arroje las 10 mas parecidas. Para ello, creamos una variable recomendacion y hacemos un slising
con los diez primeros elementos, sin seleccionar el primero (el 0), porque es el que es igual, pero necesitamos solo los similares."""

recomendacion = (-similitud).argsort()[1:11]

print (recomendacion)

#[4139 3647 1034  387 3426 3841 2489 1402 2558 3678] ----> Me arroja los indices de las peliculas

"""AHora, voy a seleccionar de esos indices los titulos de las peliculas...."""

titulos_pelis_recomendadas = df['movie_title'].iloc[recomendacion]


print(titulos_pelis_recomendadas)


