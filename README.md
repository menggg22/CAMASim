# EvaCAM+
A comprehensive tool for evaluating CAM-based CIM accelerators  for both chip-level performance and application accuracy with hardware constraints.

#### Author: Mengyuan Li (mli22@nd.edu), University of Notre Dame

## What is this?
- The tool provides a Python interface for easy integration with popular Python-based ML platforms.
- Application-specific data and search operations are used as input for architecture modeling and FoM evaluation
- Functional simulation is conducted to generate CAM outputs for analyzing application accuracy
- Architectural modeling is performed for chip-level FoMs and built on top of the previous circuit-level EvaCAM tools that support different underlying CAM designs.


## Supported Function (v1.0)
We model architecture-level design space for CAM architecture. The following aspects are considered:
- CAM Type: BCAM, TCAM, MCAM, ACAM 
- Match scheme function: best/exact/threshold
- Distance function: Hamming, L2...
- Mapping: map data based on array row, column
- Merge function 

Updates planned in the next version:
- Sensing scheme / device variations 
- More merge schemes
- A complete simulator EvaCAM+ including previou EvaCAM tool


## Requirements
- python 3.7+

## How to run CAM simulation

## **File descriptions**
- `example.py`: a simple example presents how to use the 'CAMArch' class for nearest neighbor search.
- `CAMArch.py`: 'CAMArch' class 
- `QuerySimulation.py`: 
- `PerformanceEvaluation.py`: 
- `CAMArchEstimator.py`:

- `distance.py`: support array-level distance modelling.
- `query.py`: support different match function for each array(best/exact/threshold/actual distance)
- `quantize.py`: implement basic quantization function.
- `example.py`: a simple example presents how to use the 'camnn' class for searching.
- `variation.py`: add write variation to each CAM cell after mapping.
- `*.ipynb`: debug file (can ignore).


## **Key configuration parameters** 
    * metric *: string value. We support 'hamming', 'euclidean','manhattan','innerproduct'.
    * hknn *: boolean. Default 'True' we use knn on the algorithm level. 
    * hthnn *: boolean. Use threhold NN.
    * hexact *: boolean. Use exact NN.
    * hk *: integer. Specify how many nearest neighbor found in the whole dataset.
    * hth *: fp. Specity the threhold for the dataset.


    # CAM Parameters
    * col_size *: integer. Cam array column.
    * row_size *: integer. cam array row.
    * cknn *: boolean value. Default means we find best match in each cam. 
    * cthnn *: boolean value. Default means we find threshold match in each cam.
    * cexact *: boolean value. Default means we find exact match in each cam. 
    * ck *: integer. Specify how many nearest neighbor found in each CAM array.
    * cth *: fp. Specity the threhold for each cam array.

