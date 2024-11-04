
import numpy as np
from scipy.optimize import newton

conduct2VbdFunc = {
    "6T2M": lambda x: -0.18858359 * np.exp(-0.16350861 * x)
    + 0.00518336 * x
    + 0.56900874,
    "8T2M": lambda x: -2.79080037e-01 * np.exp(-1.24915981e-01 * x)
    + 6.36010747e-04 * x
    + 1.00910243,
}

N2V_linear_range_margin = 0.05

def acam_N2V(
    data,
    Nmin,
    Nmax,
    VbdMin: float,
    VbdMax: float,
):
    """
    This function takes an array and converts it into a voltage array while also providing sparsity statistics.
    It is designed for ACAM threshold arrays.
    """
    dontCareCnt = 0

    assert (
        len(data.shape) == 3 and data.shape[2] == 2
    ), "ERROR: the input array shape is not correct. This function is only for ACAM threshold array!"

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j, 0]) and np.isnan(data[i, j, 1]):
                dontCareCnt += 1
            if np.isnan(data[i, j, 0]):
                data[i, j, 0] = VbdMin
            else:
                data[i, j, 0] = num2Vbd(data[i, j, 0], Nmin, Nmax, VbdMin, VbdMax)
            if np.isnan(data[i, j, 1]):
                data[i, j, 1] = VbdMax
            else:
                data[i, j, 1] = num2Vbd(data[i, j, 1], Nmin, Nmax, VbdMin, VbdMax)

    return data, dontCareCnt / (data.shape[0] * data.shape[1])


def num2Vbd(
    num: float | np.ndarray,
    numMin: float,
    numMax: float,
    VbdMin: float,
    VbdMax: float,
) -> float | np.ndarray:
    assert numMin < numMax and VbdMin < VbdMax
    percentage = (num - numMin) / (numMax - numMin)
    return VbdMin + (VbdMax - VbdMin) * percentage


class convertToPhys:
    def __init__(self, cell_config):
        """
        Initializes the convertToPhys class with cell configuration.

        Args:
            cell_config (dict): Configuration settings for CAM cells.

        The class is designed to convert numerical data to a physical representation used in CAM for search operations.
        """
        self.physical_rep = cell_config["representation"]
        self.cell = cell_config["type"]  # ACAM/TCAM/MCAM
        self.device = cell_config["device"]  # RRAM, FeFET...
        self.design = cell_config["design"]  # 6T2M
        self.config = cell_config

        self.Nmin = None
        self.Nmax = None

        self.VbdMin = conduct2VbdFunc[self.design](self.config.get("minConductance", 0))
        self.VbdMax = conduct2VbdFunc[self.design](self.config.get("maxConductance",150))

    def write(self, data):
        """
        Converts data to a physical representation suitable for write operations.

        Args:
            data (array): Input data to be converted.

        Depending on the CAM cell type (e.g., ACAM), it converts data to a physical voltage representation.
        """
        if self.cell == "ACAM":
            # Check for NaN values to determine Nmin and Nmax
            has_nan = np.isnan(data).any()

            if has_nan:
                self.Nmin = np.nanmin(data)
                self.Nmax = np.nanmax(data)
            else:
                # Find the minimum and maximum values
                self.Nmin = np.min(data)
                self.Nmax = np.max(data)
            Nrange = self.Nmax - self.Nmin
            global N2V_linear_range_margin
            self.Nmin -= Nrange * N2V_linear_range_margin
            self.Nmax += Nrange * N2V_linear_range_margin
            data, sparsity = acam_N2V(data, self.Nmin, self.Nmax, self.VbdMin, self.VbdMax)
        else:
            data = data
            print(
                "For now, only ACAM converter to voltage is done. For other CAM types, use numerical values"
            )
            # Check for NaN values to determine Nmin and Nmax
            has_nan = np.isnan(data).any()

            if has_nan:
                self.Nmin = np.nanmin(data)
                self.Nmax = np.nanmax(data)
            else:
                # Find the minimum and maximum values
                self.Nmin = np.min(data)
                self.Nmax = np.max(data)

        return data

    def query(self, input):
        """
        Converts numerical input vector or value to the correct voltage used in CAM for search.

        Args:
            input (Union[float, np.ndarray]): Numerical input data for conversion.

        Returns the input converted to the appropriate voltage for CAM search operations (Vsl).
        """
        if self.cell == "ACAM":
            input = num2Vbd(input, self.Nmin, self.Nmax, self.VbdMin, self.VbdMax)
        return input

    def __RRAM_V2C(VbdArray: np.ndarray, cellType: str) -> np.ndarray:
        """
        RRAM: Convert boundary voltage to memristor conductance.
        """

        global conduct2VbdFunc

        assert len(VbdArray.shape) == 2 or (
            len(VbdArray.shape) == 3 and VbdArray.shape[2] == 2
        ), "ERROR: VbdArray is not an array with valid shape"
        result = np.zeros(VbdArray.shape)

        # Solving transcendental equations using the bisection approximation
        guessX = 1
        maxIterations = 10000
        tolerance = 1e-9
        for i in range(VbdArray.shape[0]):
            for j in range(VbdArray.shape[1]):
                if len(VbdArray.shape) == 2:  # TCAM
                    func_temp = lambda x: conduct2VbdFunc[cellType](x) - VbdArray[i, j]
                    # Perform Newton's method iteration
                    result[i, j] = float(newton(func_temp, guessX))
                else:
                    for k in range(VbdArray.shape[2]):
                        func_temp = (
                            lambda x: conduct2VbdFunc[cellType](x) - VbdArray[i, j, k]
                        )
                        # Perform Newton's method iteration
                        result[i, j, k] = float(newton(func_temp, guessX))

        return result

    def __RRAM_C2V(self, conductanceArray: np.ndarray, cellType: str) -> np.ndarray:
        """
        Convert RRAM conductance to boundary voltage.
        This function can handle Vbd Array for both ACAM and TCAM
        """

        # if this func needs to be changed, func in Vbd2Res() should also be changed！
        return conduct2VbdFunc[cellType](conductanceArray)
