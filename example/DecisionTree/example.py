
import json
import os
import random
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)

# Import necessary libraries for decison tree
from dt2cam import DT2Array
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from CAMASim.CAMASim import CAMASim

def load_config():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.split(script_path)[0]
    with open(script_dir+ '/cam_config.json') as f:
        config = json.load(f)
    return config

def simCAM(CAM_Data, CAM_Query):
    cam_config = load_config()
    cam = CAMASim(cam_config)
    cam.write(CAM_Data)
    CAM_pred_ids, _, _ = cam.query(CAM_Query) # Accuracy Evaluation
    print('CAM Simulation Done')
    return CAM_pred_ids

def load_dataset():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Target variable (species)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_accuray(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def convert2cam(clf, X_test):
    (
        thresholdArray,
        col2featureID,
        row2classID,
        thresholdMin,
        thresholdMax,
    ) = DT2Array(clf)

    X_feature = X_test[:, col2featureID]

    queryArray = np.clip(X_feature, thresholdMin, thresholdMax)

    return thresholdArray, queryArray, row2classID


def main():
    X_train, X_test, y_train, y_test = load_dataset()

   # Create a Decision Tree classifier
   # (Todo): Support random forest
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    # Prediction (Original)
    y_pred = clf.predict(X_test)
    accuracy_original = evaluate_accuray(y_test, y_pred)
    print('DT Accuracy (original): ', accuracy_original)

    # CAM Prediction
    CAM_Array, CAM_Query, row2classID  = convert2cam(clf, X_test)
    cam_pred_row = simCAM(CAM_Array, CAM_Query)
    #cam_pred_class = np.take(row2classID, np.array(cam_pred_row).ravel()
    for i in range(len(cam_pred_row)):
        if len(cam_pred_row[i]) > 1:
            cam_pred_row[i] = [random.choice(cam_pred_row[i])]
        if len(cam_pred_row[i]) == 0:
            cam_pred_row[i] = [0]

    cam_pred_row = np.array(cam_pred_row).ravel()
    cam_pred_class = np.take(row2classID, cam_pred_row)
    accuracy_cam = evaluate_accuray(y_test, cam_pred_class)
    print('DT Accuracy (CAM): ',accuracy_cam)

if __name__ == '__main__':
    main()
