import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import re
from pathlib import Path
from typing import Union

import numpy as np
from sklearn import tree

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)


class DTLoader:
    def __init__(self, treeTextPath: Path) -> None:
        with open(treeTextPath) as fin:
            self.__treeText = fin.read()

    def treeText(self):
        return self.__treeText

    def predict(self, X: np.ndarray) -> np.ndarray:
        # convert tree text to dict
        treeDict, leafNodes, featureIDs, classIDs, thresholds = parseTreeStructure(
            self.treeText()
        )

        result = []
        for i in range(X.shape[0]):
            result.append(self.__predRow(X[i, :], treeDict))

        return np.array(result)

    def __predRow(self, row: np.ndarray, node: dict) -> int:
        try:
            label = node["class"]
            assert isinstance(label, int)
            return label
        except KeyError:
            if row[node["featureID"]] <= node["threshold"]:
                return self.__predRow(row, node["leNode"])
            else:
                return self.__predRow(row, node["gtNode"])


class RFLoader:
    def __init__(self, treeTextPathList: list[Path]) -> None:
        self.estimators_ = [DTLoader(path) for path in treeTextPathList]

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(self.estimators_), X.shape[0]))
        for i in range(len(self.estimators_)):
            pred[i, :] = self.estimators_[i].predict(X)

        pred = pred.T

        result = []
        for row in pred:
            values, counts = np.unique(row, return_counts=True)
            most_frequent = values[np.argmax(counts)]
            result.append(most_frequent)
        return np.array(result)


def DT2Array(
    DT: tree.DecisionTreeClassifier | DTLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    this function takes a decision tree and
    returns CAMVbdArray, col2featureID, row2classID, sparsity, thresholdMin, thresholdMax
    """
    if isinstance(DT, tree.DecisionTreeClassifier):
        treeText = tree.export_text(DT)
    elif isinstance(DT, DTLoader):
        treeText = DT.treeText()
    else:
        raise TypeError
    # print(treeText)
    # exit(-1)

    # convert tree text to dict
    treeDict, leafNodes, featureIDs, classIDs, thresholds = parseTreeStructure(treeText)
    thresholdArray, col2featureID, row2classID = __tree2CAMThresholdArray(
        leafNodes, featureIDs
    )
    # print("thresholdArray: ", thresholdArray)
    # VbdArray, sparsity = __threshold2Vbd(
    #    thresholdArray, min(thresholds), max(thresholds)
    # )

    return (
        thresholdArray,
        col2featureID,
        row2classID,
        min(thresholds),
        max(thresholds),
    )


def __tree2CAMThresholdArray(
    leafNodes: list[dict],
    featureIDs: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    featureIDs.sort()
    # the meaning of each dimension of the CAMnumArray: row, col, lowerBound/upperBound
    thresholdArray = np.full(
        (len(leafNodes), len(featureIDs), 2), np.nan, dtype=np.float64
    )
    row2classID = []

    for leafNode in leafNodes:
        node = leafNode
        row2classID.append(leafNode["class"])
        while node["parent"] != None:
            parentNode = node["parent"]
            featureID = parentNode["featureID"]
            if parentNode["leNode"] == node:
                sign = "<="
            elif parentNode["gtNode"] == node:
                sign = ">"
            else:
                sign = "?"
                assert 0, "info in parent/child does not agree"
            threshold = parentNode["threshold"]
            rowID = len(row2classID) - 1
            try:
                colID = featureIDs.index(featureID)
            except ValueError:
                print("ERROR: the feature id does not exist in featureIDs list!")
                exit(-1)
            boundaryID = 0 if sign == ">" else 1

            if thresholdArray[rowID, colID, boundaryID] == np.nan:
                thresholdArray[rowID, colID, boundaryID] = threshold
            else:
                if boundaryID == 0:
                    thresholdArray[rowID, colID, boundaryID] = max(
                        threshold, thresholdArray[rowID, colID, boundaryID]
                    )
                else:
                    thresholdArray[rowID, colID, boundaryID] = min(
                        threshold, thresholdArray[rowID, colID, boundaryID]
                    )

            node = parentNode

    return thresholdArray, np.array(featureIDs), np.array(row2classID)


# Parse the tree structure text into a nested dictionary
def parseTreeStructure(
    text: str,
) -> tuple[dict, list[dict], list[int], list[int], list[float]]:
    """
    this function takes a tree text and returns the parsed tree structure and all leaf nodes, and all feature ids, class ids, and thresholds used in the tree.
    """
    lines = text.strip().split("\n")

    leafNodes = []
    featureIDs = []
    classIDs = []
    thresholds = []
    (
        treeDict,
        subTreeEndLineID,
        leafNodes,
        featureIDs,
        classIDs,
        thresholds,
    ) = __parseSubTree(lines, 0, None, leafNodes, featureIDs, classIDs, thresholds)
    assert (
        subTreeEndLineID == len(lines) - 1
    ), "ERROR: the tree is not completely parsed!"

    return treeDict, leafNodes, featureIDs, classIDs, thresholds


__nodeUID = 0  # add an uid to make each node unique


def __parseSubTree(
    lines: list[str],
    lineID: int,
    parentNode: dict | None,
    leafNodes: list[dict],
    featureIDs: list[int],
    classIDs: list[int],
    thresholds: list[float],
) -> tuple[dict, int, list[dict], list[int], list[int], list[float]]:
    """
    this function takes all lines of a tree, the line id where the subtree to be parsed starts, and the pointer to parentNode.
    this function returns the parsed sub tree and the line id where the subtree ends(not the id where the following structure of tree starts!).
    """
    global __nodeUID
    if "class:" in lines[lineID]:  # this subtree is a leaf node
        classID = re.search("[0-9]+.?0?", lines[lineID]).group()
        classID = int(classID.strip().split(".")[0])
        leafNode = {"class": classID, "parent": parentNode, "uid": __nodeUID}
        __nodeUID += 1
        leafNodes.append(leafNode)
        if classID not in classIDs:
            classIDs.append(classID)
        return (leafNode, lineID, leafNodes, featureIDs, classIDs, thresholds)
    elif re.search(
        "feature_[0-9]+ <=", lines[lineID]
    ):  # is a stem node, parse recursively
        # the following code assumes that le and ge nodes both exists!
        stemNode = {
            "featureID": int,
            "threshold": float,
            "leNode": dict,  # less or equal. e.g. feature 3 <= 4
            "gtNode": dict,  # greater than.  e.g. feature 3 >  4
            "parent": Union[None, dict],
            "uid": __nodeUID,
        }
        __nodeUID += 1
        matchStr = re.search("feature_[0-9]+", lines[lineID]).group()
        featureID = int(matchStr.split("_")[1])
        if featureID not in featureIDs:
            featureIDs.append(featureID)
        matchStr = re.search("<=[ ]+[-]?[0-9]+.[0-9]+", lines[lineID]).group()
        threshold = float(matchStr.split(" ")[-1])
        thresholds.append(threshold)
        leNode, endLineId, leafNodes, featureIDs, classIDs, thresholds = __parseSubTree(
            lines, lineID + 1, stemNode, leafNodes, featureIDs, classIDs, thresholds
        )
        assert re.search(
            f"feature_{featureID} >", lines[endLineId + 1]
        ), "ERROR: the gt branch does not follow the end of last sub tree."
        gtNode, endLineId, leafNodes, featureIDs, classIDs, thresholds = __parseSubTree(
            lines, endLineId + 2, stemNode, leafNodes, featureIDs, classIDs, thresholds
        )
        stemNode["featureID"] = featureID
        stemNode["threshold"] = threshold
        stemNode["leNode"] = leNode
        stemNode["gtNode"] = gtNode
        stemNode["parent"] = parentNode

        return stemNode, endLineId, leafNodes, featureIDs, classIDs, thresholds
    else:
        assert not re.search(
            "truncated", lines[lineID]
        ), "Tree too deep. Some branch is truncated in tree text"
        raise NotImplementedError


def __DT2TCAM(DT: tree.DecisionTreeClassifier) -> np.ndarray:
    """
    maps a Decision Tree to TCAM
    """
    flag = False
    treeText = tree.export_text(DT)
    print(treeText)
    lines = treeText.split("\n")
    depth = DT.get_depth()
    table = []
    visited = []
    dog = []
    catch = []
    for l in lines[:-1]:
        app = depth - (l.find("feature") - 1) // 4
        if app == depth + 1:
            dog.append(-1)
            this_line = [0 for _ in range(DT.n_features_in_)]
            for index, this_depth in visited:
                this_line[index] = 1
            # print(visited)
            if len(visited) != 0:
                end = visited[-1][1]
            if flag:
                end = catch[-1][1]
            for i in range(end, depth):
                for j in catch[::-1]:
                    if j[1] == i:
                        this_line[j[0]] = 1
            table.append(this_line)
        else:
            if flag:
                flag = False
            left = l.find("_") + 1
            right = l.find("<=") if l.find("<=") != -1 else l.find(">")
            index = int(l[left:right])
            if len(visited) == 0:
                visited.append((index, app))
            else:
                if visited[-1] != (index, app):
                    visited.append((index, app))
                else:
                    poped = visited.pop()
                    catch.append(poped)
                    flag = True
    table = np.array(table)
    return (table > 0).astype(np.int32)


# def findSparsity(DT: tree.DecisionTreeClassifier):
#     """
#     Calculates the sparsity of a Decision Tree when mapped to CAM.
#     """
#     table = __DT2TCAM(DT)
#     empty = 0
#     for i in range(len(table[0])):
#         if sum(table[:, i]) == 0:
#             empty += 1
#     return 1 - ((table == 1).sum() / (table.size - empty * len(table))), table


# def findSize(DT: tree.DecisionTreeClassifier):
#     """
#     Calculates the size of a Decision Tree when mapped to CAM.
#     """
#     return DT.get_n_leaves() * len(findFeatures(DT))


# def findFeatures(DT: tree.DecisionTreeClassifier) -> list:
#     """
#     Find all features used in a decision tree.
#     """
#     tree_text = tree.export_text(DT)
#     lines = tree_text.split("\n")
#     depth = DT.get_depth()
#     features = []
#     for l in lines[:-1]:
#         if l.find("class") == -1 and l.find("truncate") == -1:
#             app = depth - (l.find("feature") - 1) // 4
#             left = l.find("_") + 1
#             right = l.find("<=") if l.find("<=") != -1 else l.find(">")
#             index = int(l[left:right])
#             features.append(index)
#     return list(set(features))


# def compressTreeTable(
#     thresholdArray: np.ndarray, tileWidth: int, tileHeight: int
# ) -> float:
#     """
#     This function takes a tabular representation of a decision tree and apply Giacomo's compression algorithm to it.
#     """
#     assert (
#         len(thresholdArray.shape) == 3 and thresholdArray.shape[2] == 2
#     ), "ERROR: the dimension of the input array is not correct."

#     # emptyCells = __getEmptyCellLocation(thresholdArray)
#     # import pandas as pd
#     # __temp = pd.DataFrame(emptyCells)
#     # __temp.to_csv('./before.csv')

#     thresholdArray = __rearrangeTable(thresholdArray)

#     # emptyCells = __getEmptyCellLocation(thresholdArray)
#     # __temp = pd.DataFrame(emptyCells)
#     # __temp.to_csv('./after.csv')

#     compressionRate = __map2smallTile(thresholdArray, tileWidth, tileHeight)

#     return compressionRate


# def __map2smallTile(
#     thresholdArray: np.ndarray, tileWidth: int, tileHeight: int
# ) -> float:
#     emptyCells = __getEmptyCellLocation(thresholdArray)
#     tileRowCnt = 0
#     for tileColIndex in range(((thresholdArray.shape[1] - 1) // tileWidth) + 1):
#         for rowIndex in range(thresholdArray.shape[0]):
#             if (
#                 emptyCells[
#                     rowIndex : rowIndex + 1,
#                     tileWidth * tileColIndex : tileWidth * (tileColIndex + 1),
#                 ].min()
#                 != 1
#             ):  # has non-empty cell
#                 tileRowCnt += 1

#     tileCnt = 0 if tileRowCnt == 0 else (tileRowCnt // tileHeight)
#     compressionRate = (tileWidth * tileHeight * tileCnt) / (
#         thresholdArray.shape[0] * thresholdArray.shape[1]
#     )

#     return compressionRate


# def __getEmptyCellLocation(thresholdArray: np.ndarray) -> np.ndarray:
#     """
#     This function finds the locations that has empty cells.
#     In the returned array, the place where a '1' is presented is where we have an empty cell.
#     """
#     assert (
#         len(thresholdArray.shape) == 3 and thresholdArray.shape[2] == 2
#     ), "ERROR: the dimension of the input array is not correct."
#     result = np.zeros((thresholdArray.shape[0], thresholdArray.shape[1]), dtype=np.int8)
#     for i in range(thresholdArray.shape[0]):
#         for j in range(thresholdArray.shape[1]):
#             if np.isnan(thresholdArray[i, j, 0]) and np.isnan(thresholdArray[i, j, 1]):
#                 result[i, j] = 1

#     return result


# def __rearrangeTable(thresholdArray: np.ndarray) -> np.ndarray:
#     assert (
#         len(thresholdArray.shape) == 3 and thresholdArray.shape[2] == 2
#     ), "ERROR: the dimension of the input array is not correct."
#     colEmptyCellCnts = np.zeros((thresholdArray.shape[1]))
#     for colIndex in range(thresholdArray.shape[1]):
#         colEmptyCellCnt = 0
#         for rowIndex in range(thresholdArray.shape[0]):
#             if np.isnan(thresholdArray[rowIndex, colIndex, 0]) and np.isnan(
#                 thresholdArray[rowIndex, colIndex, 1]
#             ):
#                 colEmptyCellCnt += 1
#         colEmptyCellCnts[colIndex] = colEmptyCellCnt
#     sortedColIndices = np.argsort(colEmptyCellCnts)
#     thresholdArray = thresholdArray[:, sortedColIndices, :]

#     rowEmptyCellCnts = np.zeros((thresholdArray.shape[0]))
#     for rowIndex in range(thresholdArray.shape[0]):
#         rowEmptyCellCnt = 0
#         for colIndex in range(thresholdArray.shape[1]):
#             if np.isnan(thresholdArray[rowIndex, colIndex, 0]) and np.isnan(
#                 thresholdArray[rowIndex, colIndex, 1]
#             ):
#                 rowEmptyCellCnt += 1
#         rowEmptyCellCnts[rowIndex] = rowEmptyCellCnt

#     sortedRowIndices = np.argsort(rowEmptyCellCnts)
#     thresholdArray = thresholdArray[sortedRowIndices, :, :]

#     return thresholdArray


# def __test_compressTreeTable():
#     import data.datasets.gas_concentrations as gas
#     from sklearn.ensemble import RandomForestClassifier
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import accuracy_score
#     from CAMASim.function.convert import acam_N2V

#     # Import dataset loader
#     import data.datasets.iris as iris
#     import data.datasets.BelgiumTSC.BTSC as BelgiumTSC
#     import data.datasets.gas_concentrations as gas
#     import data.datasets.BelgiumTSC.BTSC_adapted as BTSC_adapted

#     # X_train, y_train = BelgiumTSC.loadData("/home/andyliu/Researches/FuncCAM/data/datasets/BelgiumTSC/Training")
#     # X_train, X_test, y_train, y_test = gas.load_dataset()
#     X_train, X_test, y_train, y_test = BTSC_adapted.load_data()

#     # Create a Random Forest classifier
#     # clf = RandomForestClassifier(random_state=42, n_estimators=1356, max_features=129, max_leaf_nodes=217)
#     clf = RandomForestClassifier(
#         random_state=42,
#         n_estimators=15,
#         max_depth=10,
#     )
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     accuracy_original = accuracy_score(y_test, y_pred)
#     print("DT Accuracy (original): ", accuracy_original)

#     for treeNumber, treeEstimator in enumerate(clf.estimators_):
#         treeText = tree.export_text(treeEstimator, max_depth=100)
#         treeDict, leafNodes, featureIDs, classIDs, thresholds = parseTreeStructure(
#             treeText
#         )
#         thresholdArray, col2featureID, row2classID = __tree2CAMThresholdArray(
#             leafNodes, featureIDs
#         )

#         compressionRates = np.zeros(
#             (thresholdArray.shape[0], thresholdArray.shape[1])
#         )  # width, height
#         for i in range(0, compressionRates.shape[0], 1):
#             for j in range(0, compressionRates.shape[1], 1):
#                 compressionRate = compressTreeTable(thresholdArray, i + 1, j + 1)
#                 print("compressionRate: ", compressionRate)
#                 compressionRates[i, j] = compressionRate

#         # _, compressionRate = compressTreeTable(thresholdArray, thresholdArray.shape[1], thresholdArray.shape[0])
#         # print(compressionRate)

#         VbdArray, sparsity = acam_N2V(
#             thresholdArray, min(thresholds), max(thresholds), 0.3, 1.3
#         )

#         print("sparsity: ", sparsity)
#         # exit(0)
#         # Create a heatmap
#         compressionRates = pd.DataFrame(compressionRates)
#         # compressionRates.columns = np.array(compressionRates.columns) + 1
#         # print(compressionRates.columns)
#         compressionRates.to_csv("./compressionRate.csv")
#         plt.imshow(
#             compressionRates, cmap="viridis"
#         )  # Use the 'viridis' colormap, you can choose others
#         plt.colorbar()  # Add a color bar to show the scale

#         # Adjust the x-axis and y-axis tick labels to start at 1
#         plt.xticks(
#             range(0, compressionRates.shape[1], 4),
#             np.array(range(0, compressionRates.shape[1], 4)) + 1,
#         )
#         plt.yticks(
#             range(0, compressionRates.shape[0], 6),
#             np.array(range(0, compressionRates.shape[0], 6)) + 1,
#         )

#         # Add labels to the axes if needed
#         plt.ylabel("Tile Height")
#         plt.xlabel("Tile Width")

#         # Show the heatmap
#         plt.savefig("./compressionRate.png", dpi=250)
#         exit(0)


# def __test_DT2CAM():
#     from sklearn.datasets import load_iris
#     from matplotlib import pyplot as plt
#     import numpy as np

#     iris = load_iris()
#     X, y = iris.data, iris.target
#     clf = tree.DecisionTreeClassifier(max_depth=8)
#     clf = clf.fit(X, y)
#     (
#         CAMVbdArray,
#         col2featureID,
#         row2classID,
#         sparsity,
#         thresholdMin,
#         thresholdMax,
#     ) = DT2Array(clf)
#     print("col2featureID: ", col2featureID)
#     print("row2classID: ", row2classID)
#     print("sparsity: ", sparsity)
#     print("thresholdMin: ", thresholdMin)
#     print("thresholdMax: ", thresholdMax)
#     print("CAMVbdArray: ", CAMVbdArray)


if __name__ == "__main__":
    # __test_compressTreeTable()
    pass
