"""Librería para graficar las similitudes de películas"""
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

"""Librería para trabajar con dataframe nuestra db de películas"""
import pandas as pd

"""Librerías y métodos para vectorizar y obtener distancia euclidiana y similitud del coseno """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

"""Funciones"""
import functions


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

#Enumeramos todo lo que tiene
print(df['texto'].iloc[0])

row = df[['genero', 'plot_keywords', 'texto']].iloc[0]
print("ROW",row)

"""
Ya tenemos nuestra columna hecha y podremos convertir esto en una vectorizacion del tipo TF e IDF.

- TF: Se calcula como el número de veces que aparece una palabra en un documento, dividido por el total de palabras de ese doc.
Aquí cada texto será un documento.

- IDF: Logaritmo del total de documentos dividido por el número de documentos que contienen la palabra

- TF-IDF: Se calcula multiplicando dos componentes: TF e IDF

Para representarlas,haremos una funcion...
"""

tfidf = TfidfVectorizer(max_features=2000) #max_features = CANTIDAD MAXIMA DE TOKEN (NO DEBE HABER UNA PELI PARA MAS DE 2000 TOKEN. Esto lo averiguamos con print(df['texto'].iloc[0]) )

# 1)Vectorizamos
#Creamos nuestra matriz de vectores de características TF-IDF sobre la columna texto.
X = tfidf.fit_transform(df['texto'])
print(X)

#Utilizamos nuestra funcion para generar un mapeo del total de vocabulario creado en la variable 'tfidf' con los valores idf
idf_dict = functions.get_map_vocabulary_and_idfVaules(tfidf)

# Imprimir el diccionario
# for term, idf in list(idf_dict.items())[:10]:  # Imprime los primeros 10 términos y sus IDF
#     print(f"{term}: {idf}")

"""
La variable tfidf.idf_ contiene los valores de IDF (Inverse Document Frequency) para cada término en el vocabulario del 
modelo TfidfVectorizer que cree.
Nuestra matriz es de 5043 x 2000, si le poniamos mas en el max_features teniamos una matriz mas grande

El array tiene una longitud de 2000, lo que indica que el TfidfVectorizer ha creado un vocabulario con 2000 términos. si le poniamos 
mas en el max_features teniamos una matriz mas grande


Los valores en tfidf.idf_ reflejan la importancia de cada término en el vocabulario de las películas en función de su presencia en el corpus.

Valores más altos indican que el término es menos frecuente en el corpus, por lo tanto, es más distintivo.
Valores más bajos indican que el término aparece en muchos documentos, por lo que tiene menos poder discriminativo.

"""

# 2) Generamos un mapeo de las películas

"""Ahora vamos a generar un mapeo de las peliculas, un nuevo dataframe que será una serie en realidad en donde el 
    indice de esta serie va a ser el titulo de la pelicula (movie_tittle)"""
peliculas = pd.Series(df.index, index=df['movie_title'])

print (peliculas)

"""El error dice que no reconoce la clave, y debe ser porque existe un espacio de mas, lo mas probable al principio o al final.
Para solucionarlo, podemos usar un strip para que saque los espacios entre los nombres de las peliculas y los indices"""

peliculas.index = peliculas.index.str.strip()
print (peliculas)

indice = peliculas['The Dark Knight Rises']
print ("Indice:",indice)


"""
Ahora haremos una consulta y averiguaremos su vector. Nos dirá que es un vector de  2000 dimensiones (1 fila, 2000 columnas)
No deja de ser una matriz dispersa, es decir que contiene principalmente muchos ceros.

Esta variable contendrá los valores TF-IDF para la película que la hemos asignado a la variable índice.
"""

consulta = X[indice]
print ("Consulta del vector de la pelicula The Following:", consulta)

"""
Se ve extraño porque hizo una compresion del vector, mostrando unicamente las dimensiones que no tenian 0
Por ello, para mostrar el vector completo (la mayoria seran 0) lo convertimos en array.
"""

print (consulta.toarray())
#[[0. 0. 0. ... 0. 0. 0.]]


"""Tambien podemos ver en 'consulta.indices' los indices de las columnas donde los valores no son nulos

A su vez con 'consulta.data' podemos ver los valores TF-IDF correspondientes a los índices.
Es decir, si en la posición de 'consulta.indices' tenemos un valor de 0.15 en 'consulta.data', quiere decir que la 
posicion 32, tiene una importancia TF-IDF de 0.15 para la pelicula indicada. En otras palabras, tiene un valor especifico
que indica la importancia de esos términos en el contexto de la película guardada en la variable 'indice'
"""

# Llamar a la función para obtener términos y valores TF-IDF
term_values = functions.get_terms_from_vector(consulta, tfidf)


""" Ahora vamos a calcular la similitud del coseno sobre la consulta (es el indice  de la pelicula 'The Dark Knight Rises' ) y el vector 'X' 
que es el vector con todas nuestras peliculas..."""

#3) Calculamos la similitud del coseno
similitud = cosine_similarity(consulta, X)
print("Vector X (QUE CONTIENE TODAS NUESTRAS PELICULAS):", consulta)
print (similitud)


"""Nos da el arreglo con todas las similitudes. En donde dice solo '0', implica que no son nada parecidos
Vemos que en el elemento 4 del arreglo de similitudes, es decir  The Dark Knight Rises tiene similitud 1 con la pelicula The Dark Knight Rises"""

print (similitud[0][3])


"""Para evitar poniendo como matriz 0 y 3, vamos a transformar en un array la similitud"""
similitud = similitud.flatten()
print (similitud[3])

#4) Graficamos la similitud
plt.figure()  # Crea una nueva figura
plt.plot(similitud) #Crea el grafico de la similitud
plt.xlabel('Película') #Añade una etiqueta al eje x
plt.ylabel('Similitud') #Añade una etiqueta al eje y
plt.title('Similitud de películas con The Dark Knight Rises') #Añade un titulo al grafico
plt.savefig('similitud.png')  # Guarda el gráfico en un archivo
plt.show() #Muestra el grafico en una ventana

"""Una vez guardado el grafico, vamos a ordenar las similitudes en orden descendente para que otorgue un array con elementos (indices de 
peliculas)que van desde el mas parecido al menos con la pelicula que pasamos como indice. 
Justamente esta primero el 3 porque es el indice de la pelicula."""

print("Orden descendente de similitudes:",(-similitud).argsort())
orden_similitud = (-similitud).argsort()


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

"""Ahora, voy a seleccionar de esos indices los titulos de las peliculas...."""

titulos_pelis_recomendadas = df['movie_title'].iloc[recomendacion]
print(titulos_pelis_recomendadas)


