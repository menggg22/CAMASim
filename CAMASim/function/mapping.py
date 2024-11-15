import math

import numpy as np


class mapping:
    def __init__(self, cam_size, map_rule, array_config):
        """
        Initializes the mapping class with configuration settings.

        Args:
            cam_size (dict): Size configuration of the CAM array.
            map_rule (dict): Mapping rule configuration (to be implemented).
            array_config (dict): Configuration settings for the CAM array.

        Initializes variables and data structures for mapping data to CAM arrays.
        """
        self.rowSize = array_config['row']  # Number of rows in each array
        self.colSize = array_config['col']  # Number of columns in each array
        self.rowCams = 0  # Number of row-wise arrays
        self.colCams = 0  # Number of column-wise arrays
        self.map_rule = map_rule  # Mapping rule (to be implemented)
        self.cam_size = None
        self.CamData = []  # CAM arrays storing the data
        self.QueryData = []  # Query data

    def check_size(self, data):
        """
        Check if the size of the write data is within the maximum size of the CAM.

        Args:
            data (array): Input data to be checked.

        Returns:
            cam_usage (float): Fraction of CAM array usage.

        Raises:
            NotImplementedError: If the CAM size is smaller than the dataset size.
        """
        self.data_size = np.shape(data)[0] * np.shape(data)[1]

        if not self.cam_size:
            self.cam_size = self.data_size
            cam_usage = self.data_size / self.cam_size
        elif self.cam_size < self.data_size:
            print('CAM Size is smaller than dataset size. Not supported now.')
            raise NotImplementedError
        else:
            print('CAM size is larger than dataset size. Write all data into CAM.')
            cam_usage = self.data_size / self.cam_size
        return cam_usage

    def write(self, data):
        """
        Map the data to CAM arrays for write operations.

        Args:
            data (array): Input data to be mapped.

        Returns:
            cam_usage (float): Fraction of CAM array usage.

        Slices the data into rowCams x colCams arrays, each with a size of rowSize x colSize.
        """
        shape = np.shape(data)

        num_col = shape[1]
        num_row = shape[0]

        self.rowCams = math.ceil(num_row / self.rowSize)
        self.colCams = math.ceil(num_col / self.colSize)

        cam_usage = self.check_size(data)

        # Padding zeros to the end
        if len(data.shape) == 3:  # Determine whether ACAM
            cam_array = np.zeros((self.rowCams * self.rowSize, self.colCams * self.colSize, 2))
            cam_array[:num_row, :num_col, :] = data
            self.CamData = np.zeros((self.rowCams, self.colCams, self.rowSize, self.colSize, 2))
            for i in range(self.rowCams):
                for j in range(self.colCams):
                    self.CamData[i, j] = cam_array[i * self.rowSize:(i + 1) * self.rowSize,
                                                  j * self.colSize:(j + 1) * self.colSize, :]
        else:
            cam_array = np.zeros((self.rowCams * self.rowSize, self.colCams * self.colSize))
            cam_array[:num_row, :num_col] = data
            self.CamData = np.zeros((self.rowCams, self.colCams, self.rowSize, self.colSize))
            for i in range(self.rowCams):
                for j in range(self.colCams):
                    self.CamData[i, j] = cam_array[i * self.rowSize:(i + 1) * self.rowSize,
                                                j * self.colSize:(j + 1) * self.colSize]
        return cam_usage

    def query(self, data):
        """
        Map the query data to multiple CAM arrays.

        Args:
            data (array): Query data to be mapped.

        Slices the query data into rows and columns for CAM query operations.
        """
        shape = np.shape(data)  # shape[0]: num_input, shape[1]: input_dim
        padded_input = np.zeros((shape[0], self.colCams * self.colSize))
        padded_input[:shape[0], :shape[1]] = data

        self.QueryData = np.zeros((shape[0], self.colCams, self.colSize))
        for i in range(shape[0]):
            for j in range(self.colCams):
                self.QueryData[i, j] = padded_input[i, j * self.colSize: (j + 1) * self.colSize]
        print("Mapping the query..., # Query, # COL", shape[0], self.colCams, '\n')

    def update(self, data):
        """
        Update the data in CAM arrays.
        To be implemented.
        """
        return None
