from typing import List, Dict


def list_equal(list1: List, list2: List) -> bool:
    if not len(list1) == len(list2):
        return False
    for elem1, elem2 in zip(list1, list2):
        if elem1 != elem2:
            return False
    return True


def dict_equal(dict1: Dict, dict2: Dict) -> bool:
    if not len(dict1) == len(dict2):
        return False
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        if k1 != k2:
            return False
        if v1 != v2:
            return False
    return True
