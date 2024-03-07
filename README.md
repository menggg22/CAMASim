# 🛠️ CAMASim
A comprehensive tool for evaluating CAM-based CIM accelerators for both chip-level performance and application accuracy with hardware constraints.

CAMASim is a Easy-to-use, Modular and Extendible package of simulating CAM-based CIM accelerators along with lots of core components layers which can be used to easily build custom search schemes. You can use it with any CAM-compatible applications with cam.write() and cam.query().

## Quick Start
### Get the tool from GitHub
```
git clone https://github.com/menggg22/CAMASim.git
```
### Dependencies
The dependencies required by CAMASim is listed in `requirements.txt`. Please run `pip install -r ./requirements.txt` to install these dependencies.

### Run examples
#### Decision Tree Inference on CAM
The example/DecisionTree directory contains scripts for converting a decision tree into CAM format and performing inference with CAM, using the iris dataset. Run the command below to see performance metrics and simulation results from CAMASim:
```sh
python example/DecisionTree/example.py
```

## Integrate in your own applications

- Step 1: Prepare stored data *CAM_Data* and query data *CAM_Query* in your application

- Step 2: modify config file *cam_config* and load config
- Step 3: Initialize *CAMASim* class with config
- Step 4: Simulate **write** opeartion on *CAM_Data* 
- Step 5: simulate **query** opeartion *CAM_Query* 

```
# Step 2 Example
import yaml

def load_config():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]

    with open(script_dir + "/cam_config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

## Step 3-5 Example
from CAMASim.CAMASim import CAMASim

def simCAM(CAM_Data, CAM_Query):
    cam_config = load_config()
    cam = CAMASim(cam_config)
    cam.write(CAM_Data)
    CAM_pred_ids, _, _ = cam.query(CAM_Query) # Accuracy Evaluation
```

## File Description
- CAMASim/ 	Library source files with function simulator, layer, and device definition
- CAMASim/function	Function simulator and basic modules.
- CAMASim/performance	Performance evaluator.
- CAMASim/arch	Architecture estimator.
- examples/	Examples for CAM-related applications

## EVACAM Integration
### What is EVACAM
Eva-CAM is a circuit/architecture-level modeling and evaluation tool for content addressable memories (CAMs) that can project the area, timing, dynamic energy, and leakage power of NVM-based CAMs. It is developed by Liu Liu at University of Notre Dame. Please go to [EVACAM github repository](https://github.com/eva-cam/EvaCAM) for more detailed information.

### Why CAMASim and EVACAM can Work Together?
CAMASim is an chip-level performance and application accuracy simulator while EVACAM gives detailed hardware costs, therefore the two simulators are perfectly  complementary to each other and can work together to give better results. When EVACAM is enabled, CAMASim will pass the CAM configuration it received to EVACAM, and read from EVACAM's performance evaluation on power and latency for whole-chip performance evaluation. If EVACAM is not enabled, CAMASim will read from pre-defined performance data for limited CAM configurations. The pre-defined performance data is stored in `CAMASim/performance/cost_config.json`

### How to enable EVACAM?
EVACAM is integrated as a git submodule within CAMASim to enhance functionality. To align with CAMASim's requirements, we've forked and slightly modified the original EVACAM, with our adapted version available at [this repo](https://github.com/Andyliu92/EvaCAM-for-CAMASim).

```sh
git submodule init
git submodule update
```

To activate EVACAM's features during simulations in CAMASim, include "useEVACAMCost": true within the array section of your CAM simulation's configuration file. For instance, in the decision tree example (`example/DecisionTree/cam_config.json`), adjust the array section as follows:

```json
    "array":{
        "row": 128,
        "col": 128,
        "sensing": "exact",
        "cell": "ACAM",
        "bit": 3,
        "useEVACAMCost": true
    },
```
This setup allows CAMASim to utilize EVACAM's evaluations of latency and power for more accurate simulation outcomes. Please note that EVACAM is written in C/C++. Compilation of is needed for the first run of EVACAM. We use `g++` to compile. Please make sure that `g++` is available in your environment.

## Citation
If you want to learn more about the CAM-based accelerators, please refer to the following manuscript:

- Mengyuan Li. (2024). CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators.

If you find this code useful in your work, please cite it using the following BibTeX:

```
@inproceedings{li2024CAMASim,
    title     = {CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators},
    author    = {Li, Mengyuan and Liu, Shiyi and Sharifi, Mohammad Mehdi and Hu, Xiaobo Sharon},
    journal= {arXiv preprint arXiv:2202.09433},
    year      = {2024}
}
```

## Contact
If you have questions or comments on the model, please contact
Mengyuan Li (mli22@nd.edu), University of Notre Dame, Shiyi Liu.