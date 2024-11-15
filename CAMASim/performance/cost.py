import json
import math
import os
import re
from pathlib import Path

"""
The File will connect to EvaCAM tool
"""

def get_component_cost():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    with open(script_dir + "/cost_config.json") as f:
        config = json.load(f)
    peripheral_cost = config["peripheral_cost"]
    array_cost = config["array_cost"]
    return peripheral_cost, array_cost


def get_EVACAM_cost(array_config: dict, cell_config: dict):
    """
    This function uses the CAMASim config to run EVACAM performance valuation and update the cost_config.json file
    The cost in cost_config.json is now for a single CAM array.
    """
    scriptPath = Path(__file__)
    workingDir = scriptPath.parent.joinpath("module/EVACAM")
    assert (
        workingDir.is_dir()
    ), "ERROR: EvaCAM submodule not loaded. Try `git submodule update --init --recursive`"

    assert os.system(f"make -C {workingDir} > /dev/null") == 0, "Failed to build EvaCAM"
    assert workingDir.joinpath(
        "Eva-CAM"
    ).exists(), 'ERROR: EvaCAM executable not exist! Check the "make" process.'

    if cell_config["type"] != "TCAM":
        # raise NotImplementedError(
        #     "EVACAM currently does not support to evaluate configs other than 'TCAM'."
        # )
        pass

    if cell_config["device"] == "RRAM":
        evacam_config_path = workingDir.joinpath("configs/ReRAM-2T2R_TCAM.cfg")
    elif cell_config["device"] == "FeFET":
        evacam_config_path = workingDir.joinpath("configs/2FeFET_TCAM.cfg")
    else:
        raise NotImplementedError

    __modify_EvaCAM_config(array_config, cell_config, evacam_config_path)

    assert (
        os.system(f"cd {workingDir};./Eva-CAM {evacam_config_path} > run_output.log")
        == 0
    ), "ERROR: EvaCAM returned non-zero value."

    array_cost = __update_cost_config()

    return array_cost


def __modify_EvaCAM_config(
    array_config: dict, cell_config: dict, evacam_config_path: Path
):
    with open(evacam_config_path) as cfgFile:
        cfgLines = cfgFile.readlines()

    modifiedItem = {
        "capacity": False,
        "wordwidth": False,
    }
    for i in range(len(cfgLines)):
        if re.search(r"-Capacity \(B\):", cfgLines[i]):
            assert (
                modifiedItem["capacity"] == False
            ), "double modification to the same config"
            col = 2 ** (int(math.log(array_config["col"] - 1, 2)) + 1)
            row = 2 ** (int(math.log(array_config["row"] - 1, 2)) + 1)
            cfgLines[i] = "-Capacity (B): %d\n" % (
                col * row / 8
            )
            modifiedItem["capacity"] = True
        elif re.search(r"-WordWidth \(bit\):", cfgLines[i]):
            assert (
                modifiedItem["wordwidth"] == False
            ), "double modification to the same config"
            cfgLines[i] = "-WordWidth (bit): %d\n" % (array_config["col"])
            modifiedItem["wordwidth"] = True

    for item in modifiedItem.keys():
        assert modifiedItem[item] == True, f"config '{item}' not changed!"

    with open(evacam_config_path, mode="w") as cfgFile:
        for line in cfgLines:
            line = line.strip()
            cfgFile.write(line + "\r\n")


def __update_cost_config() -> dict:
    scriptPath = Path(__file__)
    workingDir = scriptPath.parent.joinpath("module/EVACAM")
    assert (
        workingDir.is_dir()
    ), "ERROR: EvaCAM submodule not loaded. Please load all git submodule."

    costs = {
        "total area": {
            "lineMatchPattern": r"- Total Area =",
            "numberMatchPattern": r"[0-9]+.[0-9]+um\^2",
            "unitMatchPattern": r"um\^2",
            "value": None,
            "unit": None,
        },
        "mat area": {
            "lineMatchPattern": r"\|--- Mat Area",
            "numberMatchPattern": r"[0-9]+.[0-9]+um\^2",
            "unitMatchPattern": r"um\^2",
            "value": None,
            "unit": None,
        },
        "subarray area": {
            "lineMatchPattern": r"\|--- Subarray Area",
            "numberMatchPattern": r"[0-9]+.[0-9]+um\^2",
            "unitMatchPattern": r"um\^2",
            "value": None,
            "unit": None,
        },
        "search latency": {
            "lineMatchPattern": "-  Search Latency =",
            "numberMatchPattern": r"[0-9]+.[0-9]+[pn]s",
            "unitMatchPattern": r"[pn]s",
            "value": None,
            "unit": None,
        },
        "RESET latency": {
            "lineMatchPattern": "- RESET Latency",
            "numberMatchPattern": r"[0-9]+.[0-9]+[pn]s",
            "unitMatchPattern": r"[pn]s",
            "value": None,
            "unit": None,
        },
        "read dynamic energy": {
            "lineMatchPattern": "-  Read Dynamic Energy =",
            "numberMatchPattern": r"[0-9]+.[0-9]+pJ",
            "unitMatchPattern": r"pJ",
            "value": None,
            "unit": None,
        },
        "write dynamic energy": {
            "lineMatchPattern": "- Write Dynamic Energy =",
            "numberMatchPattern": r"[0-9]+.[0-9]+pJ",
            "unitMatchPattern": r"pJ",
            "value": None,
            "unit": None,
        },
        "leakage power": {
            "lineMatchPattern": "- Leakage Power =",
            "numberMatchPattern": r"[0-9]+.[0-9]+[um]W",
            "unitMatchPattern": r"[um]W",
            "value": None,
            "unit": None,
        },
    }

    assert workingDir.joinpath(
        "run_output.log"
    ).is_file(), "EVACAM output does not exist!"
    with open(workingDir.joinpath("run_output.log")) as fin:
        for _lineID, line in enumerate(fin):
            line = line.strip()
            assert not re.search(
                "No valid solutions", line
            ), "the CAM configuration is impractical for EVACAM to evaluate its performances!"
            for item in costs.keys():
                if re.search(costs[item]["lineMatchPattern"], line):
                    numberWithUnit = re.search(costs[item]["numberMatchPattern"], line)
                    assert numberWithUnit, "did not got number from the line."
                    number = re.search("[0-9]+.[0-9]+", numberWithUnit.group())
                    number = float(number.group())
                    assert costs[item]["value"] is None, "double match occurred!"
                    costs[item]["value"] = number
                    unit = re.search(
                        costs[item]["unitMatchPattern"], numberWithUnit.group()
                    )
                    assert unit, "failed to extract unit form line!"
                    costs[item]["unit"] = unit.group()

    for item in costs.keys():
        assert (
            costs[item]["value"] is not None and costs[item]["unit"] is not None
        ), f"Failed to extract performance value for {item}."

    costs = {
        item: {"value": costs[item]["value"], "unit": costs[item]["unit"]}
        for item in costs.keys()
    }

    costs = __convert2CAMASimFormatCost(costs)

    workingDir = scriptPath.parent
    with open(workingDir.joinpath("EVACAM_cost.json"), mode="w+") as fout:
        json.dump(costs, fout)

    return costs


def __convert2CAMASimFormatCost(cost: dict) -> dict:
    latencyConvertDict = {
        "search latency": "subarray",
        # "?": "interconnect",
        # "??": "peripheral",
        "RESET latency": "write",
    }
    energyConvertDict = {
        "read dynamic energy": "subarray",
        # "?": "interconnect",
        # "??": "peripheral",
        "write dynamic energy": "write",
    }
    convertedCost = {
        "subarray": {"latency": None, "energy": None},
        "interconnect": {"latency": 0, "energy": 0},
        "peripheral": {"latency": 0, "energy": 0},
        "write": {"latency": None, "energy": None},
    }

    for item in latencyConvertDict:
        latency = cost[item]["value"]
        unit = cost[item]["unit"]

        if unit == "ns":
            latency *= 1000
        elif unit == "ps":
            pass
        else:
            raise NotImplementedError(f"the convertion from {unit} to ns is undefined!")

        convertedCost[latencyConvertDict[item]]["latency"] = latency

    for item in energyConvertDict:
        energy = cost[item]["value"]
        unit = cost[item]["unit"]

        if unit == "pJ":
            energy *= 1e-12
        else:
            raise NotImplementedError(f"the conversion from {unit} to J is undefined!")

        convertedCost[energyConvertDict[item]]["energy"] = energy

    for item in convertedCost:
        assert (
            convertedCost[item]["energy"] is not None
            and convertedCost[item]["latency"] is not None
        ), f"item '{item}' has invalid data!"

    return convertedCost


def __test_get_EVACAM_cost():
    import yaml

    with open("test/cam_config.yml") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    get_EVACAM_cost(config["array"], config["cell"])


if __name__ == "__main__":
    __test_get_EVACAM_cost()
