# 🛠️ CAMASim
Content addressable memory (CAM) [wiki](https://en.wikipedia.org/wiki/Content-addressable_memory) stands out as an efficient hardware solution for memory-intensive search operations by supporting parallel computation in memory [](https://ieeexplore.ieee.org/abstract/document/9720562). 

CAMASim is a comprehensive simulation framework for evaluating content-addressable memory based accelerators in different memory-intensive CAM-compatiable application for both
application **accuracy** with hardware constraints and **hardware performance**.
We provide a Easy-to-use, Modular and Extendible package of simulating CAM-based accelerators. It comes with function simulator and performance evaluator which can be used to easily build custom search schemes. You can use it with any CAM-compatible applications with cam.write() and cam.query().

If you want to learn more info about the simulator, please refer to the following manuscript:

- Mengyuan Li. (2024). CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators. [link](https://arxiv.org/abs/2403.03442)

## Quick Start
### Get the tool from GitHub
```
git clone https://github.com/menggg22/CAMASim.git
```
### Dependencies

### Run examples
The examples/ folder contains example on HDMANN.

## Integrate in your own applications
- Step 0: Prepare user-defined config. Modify cam simulation config file *cam_config* and prepare user-defined cricuit-level data in *CAMASim/performance/cost_config.json* if needed.
- Step 1: Extract stored data *CAM_Data* and query data *CAM_Query* in user-defined application.
- Step 2: Load CAM simulation config.
- Step 3: Initialize *CAMASIM* class with config
- Step 4: Simulate **write** opeartion on *CAM_Data* 
- Step 5: simulate **query** opeartion *CAM_Query* 

```
# Step 2 Example
import json

def load_config():
    script_path = os.path.abspath(__file__) 
    script_dir = os.path.split(script_path)[0] 
    with open(script_dir+ '/cam_config.json', 'r') as f:
        config = json.load(f)
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
- camsim/arch	Architecture estimator. (Default arch based on majority voting type arch proposed in [link](https://www.nature.com/articles/s41598-022-23116-w))
- examples/	Example for decision tree. More coming.


## Citation
If you find this code useful in your work, please cite it using the following BibTeX:

```
@misc{li2024camasim,
      title={CAMASim: A Comprehensive Simulation Framework for Content-Addressable Memory based Accelerators}, 
      author={Mengyuan Li and Shiyi Liu and Mohammad Mehdi Sharifi and X. Sharon Hu},
      year={2024},
      eprint={2403.03442},
      archivePrefix={arXiv}
}
```

## Contact
If you have suggestions or questions on the work, please contact
[Mengyuan Li](https://menggg22.github.io) (mli22@nd.edu), University of Notre Dame or [Shiyi Liu](andyliu.pub@outlook.com).