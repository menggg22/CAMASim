import random

import numpy as np


def get_array_best_results(distance_matrix):
    """
    For each query, find the best match index and distance from the distance matrix.
    """
    k = 1
    indices = np.argpartition(distance_matrix, k - 1)[:, :k]
    distances = np.array([distance_matrix[i, x] for i, x in enumerate(indices)])
    return indices, distances


def get_array_best_results_sensing(distance_matrix, sensing_limit):
    """
    For each query, find the best match index and distance from the distance matrix.
    """

    k = 1
    indices = np.argpartition(distance_matrix, k - 1)[:, :k]
    for rowIndex in range(distance_matrix.shape[0]):
        # find all indices of distances that is within the sensing limit
        allRowIndices = np.where(
            (
                distance_matrix[rowIndex]
                <= distance_matrix[rowIndex, indices[rowIndex]] + sensing_limit
            )
            & (
                distance_matrix[rowIndex]
                >= distance_matrix[rowIndex, indices[rowIndex]] - sensing_limit
            )
        )[0]
        # randomly chose one result since hardware limit cannot tell the difference from these distances
        indices[rowIndex] = random.choice(allRowIndices)
    distances = np.array([distance_matrix[i, x] for i, x in enumerate(indices)])
    return indices, distances


def get_array_exact_results(distance_matrix):
    """
    For each query, find indices with a distance of 0 (exact match) and their distances from the distance matrix.
    """
    indices = []
    distances = []
    for i in range(distance_matrix.shape[0]):
        indice = np.where(distance_matrix[i] == 0)[0]
        indices.append(indice)
        distances.append(
            np.array([distance_matrix[i, x] for p, x in enumerate(indice)])
        )
    return indices, distances


def get_array_threshold_results(distance_matrix, threshold):
    """
    For each query, find indices with distances less than or equal to the threshold and their distances from the distance matrix.
    """
    indices = []
    distances = []
    for i in range(distance_matrix.shape[0]):
        indice = np.where(distance_matrix[i] <= threshold)[0]
        indices.append(indice)
        distances.append(
            np.array([distance_matrix[i, x] for p, x in enumerate(indice)])
        )
    return indices, distances
