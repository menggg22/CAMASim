from collections import Counter


def getKeyEqual(dct, value):
    """
    Get keys from a dictionary where the corresponding values are equal to the specified value.

    Args:
        dct (dict): Input dictionary.
        value: The value to match.

    Returns:
        list: List of keys in the dictionary with values equal to the specified value.
    """
    return [key for key in dct if (dct[key] == value)]

def getKeyLarge(dct, value):
    """
    Get keys from a dictionary where the corresponding values are greater than or equal to the specified value.

    Args:
        dct (dict): Input dictionary.
        value: The threshold value.

    Returns:
        list: List of keys in the dictionary with values greater than or equal to the specified value.
    """
    return [key for key in dct if (dct[key] >= value)]

def sortDic(dct):
    """
    Sort a dictionary by values in descending order.

    Args:
        dct (dict): Input dictionary.

    Returns:
        dict: Dictionary sorted by values in descending order.
    """
    return dict(sorted(dct.items(), key=lambda item: item[1], reverse=True))

def exact_merge(matchInd, rowCams, colCams):
    """
    Find indices that have matches in all CAM arrays, indicating an exact match.
    """
    count = Counter(matchInd)
    results = getKeyEqual(count, colCams)
    return results

def knn_merge(matchInd, rowCams, colCams, topk):
    """
    Sort the matches by frequency and return the top-k most frequent indices.
    """
    count = Counter(matchInd)
    if rowCams == 1:
        sortedCount = sortDic(count)
        results = list(sortedCount.keys())[:topk]
    else:
        raise NotImplementedError  # Functionality for row-wise CAM arrays is not implemented.
    return results

def threshold_merge(matchInd, rowCams, colCams):
    """
    Find indices that have matches in a specified number of CAM arrays, based on the column-wise CAM count.
    """
    count = Counter(matchInd)
    if colCams == 1:
        results = getKeyEqual(count, colCams)
    else:
        raise NotImplementedError  # Functionality for column-wise CAM arrays is not implemented.
