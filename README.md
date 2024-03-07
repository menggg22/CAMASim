# 🛠️ CAMASim
A comprehensive tool for evaluating CAM-based CIM accelerators  for both chip-level performance and application accuracy with hardware constraints.

CAMASim is a Easy-to-use, Modular and Extendible package of simulating CAM-based CIM accelerators along with lots of core components layers which can be used to easily build custom search schemes. You can use it with any CAM-compatible applications with cam.write() and cam.query().

## Quick Start
### Get the tool from GitHub
```
git clone https://github.com/menggg22/CAMASim.git
```
### Dependencies

### Run examples
The examples/ folder contains example on HDMANN.

## Integrate in your own applications

- Step 1: Prepare stored data *CAM_Data* and query data *CAM_Query* in your application

- Step 2: modify config file *cam_config* and load config
- Step 3: Initialize *CAMASIM* class with config
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
- camasim/ 	Library source files with function simulator, layer, and device definition
- camasim/function	Function simulator and basic modules.
- camsim/performance	Performance evaluator.
- camsim/arch	Architecture estimator.
- examples/	Example for


## Citation
If you want to learn more about the CAM-based accelerators, please refer to the following manuscript:

- Mengyuan Li. (2024). CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators.

If you find this code useful in your work, please cite it using the following BibTeX:

```
@inproceedings{li2024camasim,
    title     = {CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators},
    author    = {Li, Mengyuan and Liu, Shiyi and Sharifi, Mohammad Mehdi and Hu, Xiaobo Sharon},
    journal= {arXiv preprint arXiv:2202.09433},
    year      = {2024}
}
```

## Contact
If you have questions or comments on the model, please contact
Mengyuan Li (mli22@nd.edu), University of Notre Dame, Shiyi Liu.