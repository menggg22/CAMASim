import numpy as np
from scipy.optimize import newton
import pickle
from CAMASim.function.quantize import quantize_tensor
import random
from CAMASim.function.convert import conduct2VbdFunc

__conductVarDataPath = "./data/ACAM_variation_MLcurrent/Gbox.pkl"

def addVbdQuantVar(VbdArray: np.ndarray, cellType='6T2M') -> np.ndarray:
    """
    This function takes the ideal Vbd array for the CAM and returns the Vbd array after considering memristor quantization and variances.
    cellType：CAM cell type. Could be either "6T2M" or "8T2M"
    """
    raise DeprecationWarning("This function is deprecated after using a global config.")
    conductArray = Vbd2conduct(VbdArray, cellType)
    if __configAddQuant:
        conductArray = __addConductQuantVar(conductArray)
    else:
        conductArray = addConduDependentVar(conductArray)
    return conduct2Vbd(conductArray, cellType)


def addConduDependentVar(conductArray: np.ndarray) -> np.ndarray:
    assert len(conductArray.shape) == 2 or (
        len(conductArray.shape) == 3 and conductArray.shape[2] == 2
    ), "ERROR: VbdArray should be a 2-D or 3-D array"

    for i in range(conductArray.shape[0]):
        for j in range(conductArray.shape[1]):
            if len(conductArray.shape) == 2:
                conductArray[i, j] = __addSingleConduDependVar(conductArray[i, j])
            else:
                for k in range(conductArray.shape[2]):
                    conductArray[i, j, k] = __addSingleConduDependVar(
                        conductArray[i, j, k]
                    )

    return conductArray


def __addConductQuantVar(conductArray: np.ndarray) -> np.ndarray:
    raise DeprecationWarning("This function is deprecated after using a global config.")
    global __conductVarDataPath
    assert len(conductArray.shape) == 2 or (
        len(conductArray.shape) == 3 and conductArray.shape[2] == 2
    ), "ERROR: VbdArray should be a 2-D or 3-D array"

    with open(__conductVarDataPath, "rb") as f:
        gBox = pickle.load(f)

    quant4Bit, ___, __ = quantize_tensor(
        conductArray, num_bits=4, min_val=0, max_val=150
    )
    for i in range(conductArray.shape[0]):
        for j in range(conductArray.shape[1]):
            if len(conductArray.shape) == 2:
                conductArray[i, j] = random.choice(
                    gBox[int(quant4Bit[i, j])]
                )  # quant first, then add rand variation
            else:
                for k in range(conductArray.shape[2]):
                    conductArray[i, j, k] = random.choice(
                        gBox[int(quant4Bit[i, j, k])]
                    )  # quant first, then add rand variation

    return conductArray


def Vbd2conduct(VbdArray: np.ndarray, cellType: str) -> np.ndarray:
    """
    Convert boundary voltage to memristor conductance.
    """

    global conduct2VbdFunc

    assert len(VbdArray.shape) == 2 or (
        len(VbdArray.shape) == 3 and VbdArray.shape[2] == 2
    ), "ERROR: VbdArray is not an array with valid shape"
    result = np.zeros(VbdArray.shape)

    # Solving transcendental equations using the bisection approximation
    guessX = 1
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


def conduct2Vbd(conductanceArray: np.ndarray, cellType: str) -> np.ndarray:
    """
    Convert memristor conductance to boundary voltage.
    This function can handle Vbd Array for both ACAM and TCAM
    """

    # if this func needs to be changed, func in Vbd2Res() should also be changed！
    return conduct2VbdFunc[cellType](conductanceArray)


def __addSingleConduDependVar(conductance: float) -> float:
    """
    This function implements the continuous conductance variation funtion presented in paper https://ieeexplore.ieee.org/abstract/document/9737569.
    """
    assert conductance >= 0, "ERROR: conductance out of range!"

    if conductance >= 0 and conductance <= 30:
        Grelax = (0.317 * conductance + 1) * np.random.normal(0, 1)
    else:  # conductance > 30
        Grelax = (412 * (conductance**-1.11) + 0.927) * np.random.normal(0, 1)

    return max(0, conductance + Grelax)


def __test_conductance2Vbd():
    import matplotlib.pyplot as plt
    import pandas as pd

    a1 = pd.read_csv("./data/ACAM_variation_MLcurrent/6T2M_G_Vbd_20230822.csv")
    a2 = pd.read_csv("./data/ACAM_variation_MLcurrent/8T2M_G_Vbd_20230822.csv")

    # plot the fitting curve
    plt.subplot(2, 2, 3)
    plt.plot(
        a1.iloc[:, 0] * 1e6,
        conduct2Vbd(a1.iloc[:, 0].to_numpy() * 1e6, "6T2M"),
        "r-",
    )
    plt.plot(a1.iloc[:, 0] * 1e6, a1.iloc[:, 1], "b.")
    plt.xlabel(r"Conductance ($\mu$S)")
    plt.ylabel(r"Threshold voltage (V)")
    plt.subplot(2, 2, 4)
    plt.plot(
        a1.iloc[:, 0] * 1e6,
        conduct2Vbd(a1.iloc[:, 0].to_numpy() * 1e6, "8T2M"),
        "r-",
    )
    plt.plot(a2.iloc[:, 0] * 1e6, a2.iloc[:, 1], "b.")
    plt.xlabel(r"Conductance ($\mu$S)")
    plt.tight_layout()

    plt.savefig("my_plot.png")


def __test_Vbd2Conductance():
    import matplotlib.pyplot as plt
    import pandas as pd

    a1 = pd.read_csv("./data/ACAM_variation_MLcurrent/6T2M_G_Vbd_20230822.csv")
    a2 = pd.read_csv("./data/ACAM_variation_MLcurrent/8T2M_G_Vbd_20230822.csv")

    p1 = np.random.uniform(0.4, 1.5, (300, 1))
    p2 = np.random.uniform(0.7, 1.12, (300, 1))

    # plot the fitting curve
    plt.subplot(2, 2, 3)
    plt.plot(p1.flatten(), Vbd2conduct(p1, "6T2M").flatten() * 1e6, "r.")
    plt.plot(a1.iloc[:, 1], a1.iloc[:, 0] * 1e6, "b.")
    plt.ylabel(r"Conductance ($\mu$S)")
    plt.xlabel(r"Threshold voltage (V)")
    plt.subplot(2, 2, 4)
    plt.plot(p2, Vbd2conduct(p2, "8T2M").flatten() * 1e6, "r.")
    plt.plot(a2.iloc[:, 1], a2.iloc[:, 0] * 1e6, "b.")
    plt.ylabel(r"Conductance ($\mu$S)")
    plt.xlabel(r"Threshold voltage (V)")
    plt.tight_layout()

    plt.savefig("my_plot.png")


def __test_addVar():
    import matplotlib.pyplot as plt

    VbdArray = np.random.uniform(0.5, 1.3, (100, 100))
    VbdArrayVar = addVbdQuantVar(VbdArray)
    diff = VbdArray - VbdArrayVar

    plt.figure(figsize=(17, 5))
    VbdArray = list(VbdArray.flatten())
    plt.subplot(1, 3, 1)
    plt.hist(VbdArray, histtype="stepfilled", alpha=0.3, density=False, bins=16)
    plt.xlabel("Vbd (V)")
    plt.tight_layout()

    VbdArrayVar = list(VbdArrayVar.flatten())
    plt.subplot(1, 3, 2)
    plt.hist(VbdArrayVar, histtype="stepfilled", alpha=0.3, density=False, bins=16)
    plt.xlabel("Vbd with quant & var (V)")
    plt.tight_layout()

    diff = list(diff.flatten())
    plt.subplot(1, 3, 3)
    plt.hist(diff, histtype="stepfilled", alpha=0.3, density=False, bins=16)
    plt.xlabel("diff (V)")
    plt.tight_layout()

    plt.savefig("my_plot.png")


def __test_addContinuousVar():
    import pandas as pd

    __temp = np.zeros((400, 3))
    condu = np.linspace(0, 150, 400, endpoint=True)
    __temp[:, 0] = condu.flatten()
    condu = condu.reshape(10, 40)
    conduVar = addConduDependentVar(condu)

    __temp[:, 1] = conduVar.flatten()
    __temp[:, 2] = __temp[:, 0] - __temp[:, 1]
    __temp = pd.DataFrame(__temp)
    __temp.to_csv("./conductance.csv")

def __test_addVbdQuantVar():
    addVbdQuantVar(np.array([1,2,3]))

if __name__ == "__main__":
    __test_addVbdQuantVar()
