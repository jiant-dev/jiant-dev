import math
import itertools
from typing import Collection, Any, Sequence, Iterable, Iterator, Union


def take_one(ls: Union[Sequence, Collection]) -> Any:
    if not len(ls) == 1:
        raise IndexError(f"has more than one element ({len(ls)})")
    return next(iter(ls))


def chain_idx_get(container: Collection, key_list: Sequence, default: Any) -> Any:
    try:
        return chain_idx(container, key_list)
    except (KeyError, IndexError, TypeError):
        return default


def chain_idx(container: Collection, key_list: Sequence) -> Any:
    curr = container
    for key in key_list:
        curr = curr[key]
    return curr


def group_by(ls, key_func):
    result = {}
    for elem in ls:
        key = key_func(elem)
        if key not in result:
            result[key] = []
        result[key].append(elem)
    return result


def combine_dicts(dict_ls, strict=True, dict_class=dict):
    new_dict = dict_class()
    for i, dictionary in enumerate(dict_ls):
        for k, v in dictionary.items():
            if strict:
                if k in new_dict:
                    raise RuntimeError(f"repeated key {k} seen in dict {i}")
            new_dict[k] = v
    return new_dict


def sort_dict(d):
    return {
        k: d[k]
        for k in sorted(list(d.keys()))
    }


def partition_list(ls, n, strict=False):
    length = len(ls)
    if strict:
        assert length % n == 0
    parts_per = math.ceil(length / n)
    print(parts_per)
    result = []
    for i in range(n):
        result.append(ls[i*parts_per: (i+1) * parts_per])
    return result


class ReusableGenerator(Iterable):
    """
    Makes a generator reusable e.g.

    ```
    for x in gen:
        pass
    for x in gen:
        pass
    ```
    """
    def __init__(self, generator_function, *args, **kwargs):
        self.generator_function = generator_function
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.generator_function(*self.args, **self.kwargs)


class InfiniteYield(Iterator):

    def __init__(self, iterable: Iterable):
        self.iterable = iterable
        self.iterator = iter(itertools.cycle(self.iterable))

    def __next__(self):
        return next(self.iterator)

    def pop(self):
        return next(self.iterator)


def has_same_keys(dict1: dict, dict2: dict) -> bool:
    return dict1.keys() == dict2.keys()


def get_all_same(ls):
    assert len(set(ls)) == 1
    return ls[0]


def zip_equal(*iterables):
    sentinel = object()
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError('Iterables have different lengths')
        yield combo
