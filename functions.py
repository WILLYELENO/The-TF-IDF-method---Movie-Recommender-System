import numpy as np


def get_map_vocabulary_and_idfVaules(model):
    """ 
    Toma un objeto TfidfVectorizer y devuelve un diccionario
    que mapea términos a sus valores IDF.
    
    Args:
    tfidf_vectorizer (TfidfVectorizer): Objeto TfidfVectorizer entrenado.

    Returns:
    dict: Diccionario con términos como claves y valores IDF como valores.
    """


    # Obtener el vocabulario
    """
    Las claves: Son los términos (palabras) que aparecen en el corpus de texto.
    Los valores: Cada valor es un entero que indica la columna en la matriz donde se encuentra la información del término.
    """
    vocabulario = model.vocabulary_

    # Obtener los valores IDF
    idf_values = model.idf_

     # Crear un diccionario para asociar términos con sus valores IDF
    term_idf_mapping = {term: idf_values[idx] for term, idx in vocabulario.items()}

    return term_idf_mapping


def get_terms_from_vector(vector, model):
    """
    Toma un vector disperso de TF-IDF y un objeto TfidfVectorizer, y devuelve un diccionario
    que mapea términos a sus valores TF-IDF en ese vector.

    Args:
    vector (scipy.sparse.csr_matrix): Vector TF-IDF disperso para el cual se quieren obtener los términos y valores.
    model (TfidfVectorizer): Objeto TfidfVectorizer entrenado.

    Returns:
    dict: Diccionario con términos como claves y valores TF-IDF como valores.
    """
    # Obtener el vocabulario
    vocabulario = model.vocabulary_

    # Invertir el vocabulario para mapear índices a términos
    vocabulario_invertido = {idx: term for term, idx in vocabulario.items()}

    # Crear un diccionario para asociar términos con sus valores TF-IDF
    term_values = {vocabulario_invertido[idx]: vector.data[i] for i, idx in enumerate(vector.indices)}

    return term_values


"""
PRIMER FUNCION:

- Propósito: Esta función toma un modelo TfidfVectorizer y devuelve un diccionario que mapea cada término a su valor IDF 
(Inverse Document Frequency).

- Resultado: El diccionario contiene todos los términos del vocabulario y sus valores IDF, lo que refleja la importancia de cada 
término en el corpus de documentos en general. No está relacionado con un vector específico de documento o consulta, sino que se
 basa en el modelo entrenado.

- Valor IDF: El valor IDF indica cuán importante es el término en el corpus completo. Un valor alto significa que el término es 
raro en el corpus.

SEGUNDA FUNCION:

- Propósito: Esta función toma un vector TF-IDF disperso (como el resultado de una consulta para una película) y el modelo TfidfVectorizer,
 y devuelve un diccionario que mapea cada término presente en ese vector a su valor TF-IDF en el contexto de ese vector.

- Resultado: El diccionario contiene solo los términos que están presentes en el vector TF-IDF para el documento específico (consulta) y sus
 respectivos valores TF-IDF.

- Valor TF-IDF: El valor TF-IDF indica la importancia del término en el documento específico en comparación con el corpus. Un valor alto
 significa que el término es significativo en ese documento.

Cada función sirve para propósitos diferentes en el análisis de texto usando TF-IDF. La primera es útil para entender la importancia general
 de los términos en el corpus, mientras que la segunda es útil para analizar y entender la importancia de los términos en un documento o 
 consulta específica."""