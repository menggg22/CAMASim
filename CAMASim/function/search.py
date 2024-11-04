from collections import defaultdict

from CAMASim.function.module.distance import *
from CAMASim.function.module.merge import *
from CAMASim.function.module.sensing import *


class CAMSearch:
    def __init__(self, query_config, array_config):
        """
        Initializes the CAMSearch class with configuration settings.

        Args:
            query_config (dict): Configuration settings for query operations.
            array_config (dict): Configuration settings for the CAM array.

        Initializes variables and settings for CAM array search operations.
        """
        self.query_config = query_config
        self.array_config = array_config
        self.metric = self.define_distance_metrics()

        self.searchScheme = query_config['searchScheme']  # "exact"
        self.searchParameter = query_config['parameter']  # 20
        self.sensing = array_config['sensing']  # exact, best, threshold
        self.sensinglimit = array_config.get('sensingLimit', 0)

    def define_search_area(self, numRowCAMs, numColCAMs):
        """
        Define the search area based on the number of row and column CAMs.

        Args:
            numRowCAMs (int): Number of row-wise CAM arrays.
            numColCAMs (int): Number of column-wise CAM arrays.

        Sets the number of row and column CAMs for the search operation.
        """
        self.numRowCAMs = numRowCAMs
        self.numColCAMs = numColCAMs

    def search(self, cam_data, query_data):
        """
        Perform a search operation in CAM arrays.

        Args:
            cam_data (array): Data stored in the CAM arrays.
            query_data (array): Query data for the search.

        Returns:
            results (list): List of search results.

        Searches in multiple CAM arrays, merges results, and returns a list of search results.
        """
        matchInd = defaultdict(list)  # Matched indices from each array
        matchIndDist = defaultdict(list)  # Matched indices distance from each array
        rowSize = self.array_config['row']

        # 1. Search in multiple arrays
        for i in range(self.numRowCAMs):
            for j in range(self.numColCAMs):
                indices, distances = self.array_search(cam_data[i, j], query_data[:, j])
                for m in range(query_data.shape[0]):
                    curr_indices = [row + i * rowSize for row in indices[m]]
                    matchInd[m] += curr_indices
                    matchIndDist[m] += distances

        # 2. Merge results from multiple arrays
        results = []
        for m in range(query_data.shape[0]):
            Ind = matchInd[m]
            result = self.merge_indices(Ind, matchIndDist)
            results.append(result)
        return results

    def array_search(self, data, query):
        """
        Search operation within a single CAM array.

        Args:
            data (array): Data stored in a CAM array.
            query (array): Query data for the search.

        Returns:
            indices (list): Matched indices.
            distances (list): Distances to matched indices.

        Calculates the distance matrix and performs search in a CAM array.
        """
        # The function performs a search in a CAM array.
        # 1. Calculate the distance matrix in the array
        distance_matrix = self.array_distance(data, query)
        # 2. Find the output IDs of the array
        indices, distances = self.array_sensing(distance_matrix)
        return indices, distances

    def array_distance(self, data, query):
        """
        Calculate the distance matrix within a CAM array.

        Args:
            data (array): Data stored in a CAM array.
            query (array): Query data for the search.

        Returns:
            distance_matrix (array): Distance matrix within the CAM array.
        """
        return self.metric(data, query)

    def array_sensing(self, distance_matrix):
        """
        Perform sensing within a CAM array.

        Args:
            distance_matrix (array): Distance matrix within the CAM array.

        Returns:
            indices (list): Matched indices.
            distances (list): Distances to matched indices.

        Sensing operation based on the configured sensing method (exact, best, threshold).
        """


        if self.sensing == 'exact':
            indices, distances = get_array_exact_results(distance_matrix)
        elif self.sensing == 'best':
            if self.sensinglimit != 0:
                indices, distances = get_array_best_results_sensing(distance_matrix, self.sensinglimit)
            else:
                indices, distances = get_array_best_results(distance_matrix)
        elif self.sensing == 'threshold':
            indices, distances = get_array_threshold_results(distance_matrix, self.searchParameter)
        else:
            raise NotImplementedError
        return indices, distances

    def merge_indices(self, matchInd, matchIndDist):
        """
        Merge search results from multiple CAM arrays.

        Args:
            matchInd (list): List of matched indices from multiple arrays.
            matchIndDist (list): List of distances to matched indices from multiple arrays.

        Returns:
            result (list): Merged search result.

        Merges search results based on the configured search scheme (exact, knn, threshold).
        """
        if self.searchScheme == 'exact':
            result = exact_merge(matchInd, self.numRowCAMs, self.numColCAMs)
        elif self.searchScheme == 'knn':
            result = knn_merge(matchInd, self.numRowCAMs, self.numColCAMs, self.searchParameter)
        elif self.searchScheme == 'threshold':
            result = threshold_merge(matchInd, self.numRowCAMs, self.numColCAMs)
        else:
            print("Please choose a search scheme.")
            raise NotImplementedError
        return result

    def define_distance_metrics(self):
        """
        Define the distance metric for search operations.

        Returns:
            metric (function): Distance metric function to be used for search.

        The user can configure the distance metric for search operations. The function returns the appropriate metric function.
        """
        metric = self.query_config['distance']  # "hamming"

        if not callable(metric):
            metrics = {
                "euclidean": euclidean_pairwise,
                "manhattan": manhattan_pairwise,
                "hamming": hamming_pairwise,
                "innerproduct": innerproduct_pairwise,
                "rangequery": rangequery_pairwise
            }

            metric = metrics.get(metric, euclidean_pairwise)

        # Use hamming distance to define exact match (distance = 0)
        if self.query_config['searchScheme'] == 'exact':
            metric = metrics.get(metric, hamming_pairwise)

        if self.array_config['cell'] == "ACAM":
            metric = metrics.get(metric, rangequery_pairwise)
        return metric
