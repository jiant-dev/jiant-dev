import math
import itertools
from typing import Mapping, Any, Sequence, Iterable, Iterator, Union, Tuple, Dict


def take_one(ls: Union[Sequence, Mapping]) -> Any:
    """Extract item from a collection containing a just one item.

    Args:
        ls (Union[Sequence, Mapping]): collection containing a single item.

    Returns:
        Single item from collection.

    """
    if not len(ls) == 1:
        raise IndexError(f"has more than one element ({len(ls)})")
    return next(iter(ls))


def chain_idx_get(container: Union[Sequence, Mapping], key_list: Sequence, default: Any) -> Any:
    """Retrieve entry at a path from a arbitrarily nested collection, return default if not found.

    Args:
        container (Union[Sequence, Mapping]): collection from which to try to retrieve element.
        key_list (Sequence): list of index and/or keys specifying the path to the requested element.
        default (Any): default value to return if no value exists at the specified path.

    Returns:
        Entry found at the specified path (or default value if no entry is found at that path).

    """
    try:
        return chain_idx(container, key_list)
    except (KeyError, IndexError, TypeError):
        return default


def chain_idx(container: Union[Sequence, Mapping], key_list: Sequence) -> Any:
    """Retrieve entry at a path from a arbitrarily nested collection.

    Args:
        container (Union[Sequence, Mapping]): collection from which to try to retrieve element.
        key_list (Sequence): list of index and/or keys specifying the path to the requested element.

    Returns:
        Entry found at the specified path.

    """
    curr = container
    for key in key_list:
        curr = curr[key]
    return curr


def group_by(ls: Sequence, key_func) -> dict:
    """Apply a function to every element of a sequence.

    Args:
        ls (Sequence): sequence to process.
        key_func: function to apply.

    Returns:
        Dict mapping the result of applying the fn to the corresponding component in the sequence.

    Examples:
        group_by([1,2,3], lambda x: x**2)
        {1: [1], 4: [2], 9: [3]}

    """
    result = {}
    for elem in ls:
        key = key_func(elem)
        if key not in result:
            result[key] = []
        result[key].append(elem)
    return result


def combine_dicts(dict_ls: Sequence[dict], strict=True, dict_class=dict):
    """Merges entries from one or more dicts into a single dict (shallow copy).

    Args:
        dict_ls (Sequence[dict]): sequence of dictionaries to combine.
        strict (bool): whether to throw an exception in the event of key collision, else overwrite.
        dict_class (dictionary): dictionary class for the destination dict.

    Returns:
        Dictionary containing the entries from the input dicts.

    """
    new_dict = dict_class()
    for i, dictionary in enumerate(dict_ls):
        for k, v in dictionary.items():
            if strict:
                if k in new_dict:
                    raise RuntimeError(f"repeated key {k} seen in dict {i}")
            new_dict[k] = v
    return new_dict


def sort_dict(d: dict):
    return {k: d[k] for k in sorted(list(d.keys()))}


def partition_list(ls, n, strict=False):
    length = len(ls)
    if strict:
        assert length % n == 0
    parts_per = math.ceil(length / n)
    print(parts_per)
    result = []
    for i in range(n):
        result.append(ls[i * parts_per : (i + 1) * parts_per])
    return result


class ReusableGenerator(Iterable):
    """Makes a generator reusable e.g.

    for x in gen:
        pass
    for x in gen:
        pass
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
            raise ValueError("Iterables have different lengths")
        yield combo


class ExtendedDataClassMixin:
    @classmethod
    def get_fields(cls):
        # noinspection PyUnresolvedReferences
        return list(cls.__dataclass_fields__)

    @classmethod
    def get_annotations(cls):
        return cls.__annotations__

    def to_dict(self):
        return {k: getattr(self, k) for k in self.get_fields()}

    @classmethod
    def from_dict(cls, kwargs):
        # noinspection PyArgumentList
        return cls(**kwargs)

    def new(self, **new_kwargs):
        kwargs = {k: v for k, v in self.to_dict().items()}
        for k, v in new_kwargs.items():
            kwargs[k] = v
        # noinspection PyArgumentList
        return self.__class__(**kwargs)


class BiMap:
    """Maintains (bijective) mappings between two sets.

    Args:
        a (Sequence): sequence of set a elements.
        b (Sequence): sequence of set b elements.

    """

    def __init__(self, a: Sequence, b: Sequence):
        self.a_to_b = {}
        self.b_to_a = {}
        for i, j in zip(a, b):
            self.a_to_b[i] = j
            self.b_to_a[j] = i
        assert len(self.a_to_b) == len(self.b_to_a) == len(a) == len(b)

    def get_maps(self) -> Tuple[Dict, Dict]:
        """Return stored mappings.

        Returns:
            Tuple[Dict, Dict]: mappings from elements of a to b, and mappings from b to a.

        """
        return self.a_to_b, self.b_to_a
