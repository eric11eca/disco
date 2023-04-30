from typing import Sequence, Tuple, Dict


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


class BiDict(dict):
    """Maintains bidirectional dict
    Example:
        bd = BiDict({'a': 1, 'b': 2})
        print(bd)                     # {'a': 1, 'b': 2}
        print(bd.inverse)             # {1: ['a'], 2: ['b']}
        bd['c'] = 1                   # Now two keys have the same value (= 1)
        print(bd)                     # {'a': 1, 'c': 1, 'b': 2}
        print(bd.inverse)             # {1: ['a', 'c'], 2: ['b']}
        del bd['c']
        print(bd)                     # {'a': 1, 'b': 2}
        print(bd.inverse)             # {1: ['a'], 2: ['b']}
        del bd['a']
        print(bd)                     # {'b': 2}
        print(bd.inverse)             # {2: ['b']}
        bd['b'] = 3
        print(bd)                     # {'b': 3}
        print(bd.inverse)             # {2: [], 3: ['b']}
    """

    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)
