import os
from numpy.random import randint
import json

class Table(dict):
    """
    Modeled after the utils.Vocabulary

    Mapping from strings to sets.
    """

    def __init__(self, random_int=None):
        self._mapping = {}   # str -> int
        self._flip = {}      # int -> str; timv: consider using array or list
        self._i = 0
        self._frozen = False
        self._growing = True
        self._random_int = random_int   # if non-zero, will randomly assign
                                        # integers (between 0 and randon_int) as
                                        # index (possibly with collisions)


    def __repr__(self):
        return 'Table(size=%s)' % (len(self))

    @classmethod
    def from_iterable(cls, s):
        "Assumes keys are strings."
        inst = cls()
        for x in s:
            inst.add(x)
#        inst.freeze()
        return inst

    @classmethod
    def exact_load(cls, exact_dict):
        new_vocab = cls()
        for k,v in exact_dict.items():
            assert isinstance(v,int)
            new_vocab._mapping[k] = v
            new_vocab._flip[v] = k
        new_vocab._i = len(new_vocab) + 1
        return new_vocab


    @classmethod
    def load(cls, filename):
        if not os.path.exists(filename):
            return cls()
        with open(filename) as fp:
            return cls.exact_load(json.load(fp))

    def save(self, filename, exactly=False):
        with file(filename, 'wb') as fp:
            json.dump(self._mapping, fp)

