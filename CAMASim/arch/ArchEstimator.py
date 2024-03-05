import math

class ArchEstimator:
    def __init__(self, arch_config, array_config):
        """
        Initialize the CAM architecture estimator with configuration parameters.

        Args:
            arch_config (dict): Configuration parameters for CAM architecture.
            array_config (dict): Configuration parameters for the CAM array.

        Attributes:
            arch_config (dict): Configuration parameters for CAM architecture.
            row (int): Number of rows in the CAM array.
            col (int): Number of columns in the CAM array.
        """
        self.arch_config = arch_config
        self.row = array_config['row']
        self.col = array_config['col']

    def estimate_cam_architecture(self, data):
        """
        Estimate the CAM architecture based on the provided data.

        Args:
            data (numpy.ndarray): Input data for estimation.

        Returns:
            dict: Estimated CAM architecture parameters.

        This method estimates the size and peripherals of CAM arrays, mats, and banks based on the provided data dimensions.
        It calculates the number of arrays, mats, and banks required and the peripherals for each level.
        """
        self.num_entries = data.shape[0]
        self.data_dim = data.shape[1]

        num_array = self.estimate_size('array')
        num_mat = self.estimate_size('mat')
        num_bank = self.estimate_size('bank')

        print("--> Arch: #Arrays", num_array, ", #Mats", num_mat, ", #Banks", num_bank)

        array_peripherals = self.estimate_peripherals('array')
        mat_peripherals = self.estimate_peripherals('mat')
        bank_peripherals = self.estimate_peripherals('bank')

        cam_arch = {
            "array": {"size": num_array, "peripherals": array_peripherals},
            "mat": {"size": num_mat, "peripherals": mat_peripherals},
            "bank": {"size": num_bank, "peripherals": bank_peripherals}
        }
        return cam_arch

    def estimate_size(self, level):
        """
        Estimate the size (number of units) at the specified architectural level.

        Args:
            level (str): The architectural level ('array', 'mat', or 'bank').

        Returns:
            int: Estimated size (number of units) at the specified level.

        This method estimates the size at the specified architectural level based on the data dimensions and architectural parameters.
        """
        if self.data_dim <= self.col and self.num_entries <= self.row:
            num_subarrays = 1
        elif self.data_dim > self.col and self.num_entries <= self.row:
            num_subarrays = math.ceil(self.data_dim / self.col)
        elif self.data_dim <= self.col and self.num_entries >= self.row:
            num_subarrays = math.ceil(self.num_entries / self.row)
        else:
            print('Only support one of R and C > 1')
            raise NotImplementedError

        if level == 'array':
            num_array = math.ceil(num_subarrays / self.arch_config['SubarraysPerArray'])
            return num_array
        elif level == 'mat':
            num_mat = math.ceil(self.estimate_size('array') / self.arch_config['ArraysPerMat'])
            return num_mat
        elif level == 'bank':
            num_bank = math.ceil(self.estimate_size('mat') / self.arch_config['MatsPerBank'])
            return num_bank

    def estimate_peripherals(self, level):
        """
        Estimate the peripherals (decoder, encoder, adder, register) at the specified architectural level.

        Args:
            level (str): The architectural level ('array', 'mat', or 'bank').

        Returns:
            dict: Estimated peripherals at the specified level.

        This method estimates the peripherals (decoder, encoder, adder, register) for the specified architectural level.
        """
        if level == 'array':
            subarrays_per_array = self.arch_config['SubarraysPerArray']
            peripherals = {
                "decoder": {"num": subarrays_per_array, "size": 1},
                "encoder": {"num": subarrays_per_array, "size": 1},
                "adder": {"num": 1, "size": self.row},
                "register": {"num": self.row, "size": math.ceil(math.log2(self.row))}
            }
            return peripherals

        elif level == 'mat':
            num_entries_per_mat = self.arch_config['ArraysPerMat'] * self.row
            peripherals = {
                "adder": {"num": self.row, "size": math.ceil(math.log2(num_entries_per_mat)) - 1},
                "register": {"num": self.row, "size": math.ceil(math.log2(num_entries_per_mat)) - 1}
            }
            return peripherals

        elif level == 'bank':
            num_entries_per_bank = self.arch_config['ArraysPerMat'] * self.arch_config['MatsPerBank'] * self.row
            peripherals = {
                "adder": {"num": self.row, "size": math.ceil(math.log2(num_entries_per_bank)) - 1},
                "register": {"num": self.row, "size": math.ceil(math.log2(num_entries_per_bank)) - 1}
            }
            return peripherals
