'''
The file define hamming, l1, l2, innerproduct distance.
Refer to repository: https://github.com/JKnighten/k-nearest-neighbors
'''

from math import sqrt

import numpy as np


######################
# Euclidean Distance #
######################
def euclidean(vector_a, vector_b):
    """
    Finds the euclidean distance between two vectors.
    Args:
            vector_a (ndarray): the first 1D array must be of type np.float
            vector_b (ndarray): the second 1D array must be of type np.float

        Returns:
            (np.float): the euclidean distance between the two supplied vectors.
    """
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        temp = vector_a[i] - vector_b[i]
        distance += (temp*temp)

    return sqrt(distance)


def euclidean_pairwise(vectors_a, vectors_b):
    """
    Finds the euclidean distance between all pairs of vectors in the two supplied matrices.
    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float
    Returns:
        (ndarray): A 2D array containing the euclidean distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.
    """

    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]

    distances = np.zeros([num_vectors_b, num_vectors_a])

    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            distance = 0.0
            for k in range(num_dims):
                temp = vectors_a[j, k] - vectors_b[i, k]
                distance += (temp*temp)

            distances[i, j] = sqrt(distance)

    return distances

######################
# Manhattan Distance #
######################
def manhattan(vector_a, vector_b):
    """
    Finds the manhattan (l1) distance between two vectors.
    Args:
            vector_a (ndarray): the first 1D array must be of type np.float
            vector_b (ndarray): the second 1D array must be of type np.float

        Returns:
            (np.float): the manhattan distance between the two supplied vectors.
    """
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        temp = abs(vector_a[i] - vector_b[i])
        distance += temp

    return distance

def manhattan_pairwise(vectors_a, vectors_b):
    """
    Finds the manhattan distance between all pairs of vectors in the two supplied matrices.
    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float
    Returns:
        (ndarray): A 2D array containing the manhattan distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.
    """

    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]

    distances = np.zeros([num_vectors_b, num_vectors_a])

    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                 distances[i, j] += abs(vectors_a[j, k] - vectors_b[i, k])
    return distances

####################
# Hamming Distance #
####################
def hamming(vector_a, vector_b):
    """
    Finds the hamming between two vectors.
    Args:
            vector_a (ndarray): the first 1D array must be of type np.float
            vector_b (ndarray): the second 1D array must be of type np.float

        Returns:
            (np.float): the hamming distance between the two supplied vectors.
    """
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        if vector_a[i] != vector_b[i]:
            distance += 1.0
    return distance


def hamming_pairwise(vectors_a, vectors_b):
    """
    Finds the hamming distance between all pairs of vectors in the two supplied matrices.
    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float
    Returns:
        (ndarray): A 2D array containing the hamming distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.
    """

    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]

    distances = np.zeros([num_vectors_b, num_vectors_a])

    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                if vectors_a[j, k] != vectors_b[i, k]:
                    distances[i, j] += 1.0
    return distances

##########################
# Inner Product Distance #
##########################
def innerproduct(vector_a, vector_b):
    """
    Finds the inner product between two vectors.
    Args:
            vector_a (ndarray): the first 1D array must be of type np.float
            vector_b (ndarray): the second 1D array must be of type np.float

        Returns:
            (np.float): the hamming distance between the two supplied vectors.
    """
    dims = vector_a.shape[0]
    distance = 0
    for i in range(dims):
        distance += (vector_a[i]*vector_b[i])
    return distance

def innerproduct_pairwise(vectors_a, vectors_b):
    """
    Finds the IP distance between all pairs of vectors in the two supplied matrices.
    Args:
        vectors_a (ndarray): the first 2D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float
    Returns:
        (ndarray): A 2D array containing the IP distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.
    """

    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]

    distances = np.zeros([num_vectors_b, num_vectors_a])

    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            for k in range(num_dims):
                 distances[i, j] += vectors_a[j, k] * vectors_b[i, k]
    return distances



######################
# Range query #
######################
def rangequery(vector_a, vector_b):
    """
    Finds the range query between two vectors.
    Args:
            vector_a (ndarray): the first 1D array must be of type np.float
            vector_b (ndarray): the second 1D array must be of type np.float

        Returns:
            (np.float): the euclidean distance between the two supplied vectors.
    """
    dims = vector_a.shape[0]
    distance = 0

    for i in range(dims):
        temp = 0 if (vector_a[i] >= vector_b[i, 0]) & (vector_a[i] <= vector_b[i, 1]) else 1
        distance += temp

    return (distance)


def rangequery_pairwise(vectors_a, vectors_b):
    """
    Finds the range query between all pairs of vectors in the two supplied matrices.
    Args:
        vectors_a (ndarray): the first D array of vectors must be of type np.float
        vectors_b (ndarray): the second 2D array of vectors must be of type np.float
    Returns:
        (ndarray): A 2D array containing the euclidean distances between the vectors in the matrices provided. Rows
            correspond to the vectors in vectors_b and the columns correspond to vectors_a.
    """

    num_vectors_a = vectors_a.shape[0]
    num_vectors_b = vectors_b.shape[0]
    num_dims = vectors_a.shape[1]

    distances = np.zeros([num_vectors_b, num_vectors_a])

    for i in range(num_vectors_b):
        for j in range(num_vectors_a):
            distance = 0.0
            for k in range(num_dims):
                temp = 0 if (vectors_b[i, k] >= vectors_a[j, k, 0]) & (vectors_b[i, k] <= vectors_a[j, k, 1]) else 1
                distance += temp
            distances[i, j] = distance

    return distances
