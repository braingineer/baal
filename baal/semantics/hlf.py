from weakref import WeakValueDictionary
from baal.utils import loggers

try:
    range = xrange
except:
    pass

import sys
if sys.version_info[0] < 3:

    class gensym(object):
        __slots__ = ["__weakref__", "_instances", "sym_generator", "head", "symbol","number_inst", "pos_symbol"]
        _instances = WeakValueDictionary()
        sym_generator = ("g%d" % i for i in range(10**10))

        def __new__(cls, *args, **kargs):
            if 'head' not in kargs:
                raise ValueError("pass head")

            if 'address' in kargs:
                base = kargs['address'][:-1]
                backupkey = (cls, kargs['head'], tuple(base))
                key = (cls, kargs['head'], tuple(kargs['address']))
            else:
                backupkey = None
                key = (cls, kargs['head'])

            if backupkey and backupkey in cls._instances:
                ### keep pushing the representation down
                inst = cls._instances.setdefault(key, cls._instances[backupkey])
                ### update the symbol? 
            elif key in cls._instances:
                ### we've already seen this one
                inst = cls._instances.setdefault(key, cls._instances[key])
            else:
                ### new
                inst = cls._instances.setdefault(key, super(gensym, cls).__new__(cls))
            return inst

        def __init__(self, head="", address="", symbol=""):
            try:
                self.number_inst += 1
            except:
                self.head = head
                self.symbol = next(self.sym_generator)
                self.number_inst = 1
                self.pos_symbol = symbol

        def __str__(self):
            return self.symbol

        def __repr__(self):
            return repr(self.__str__())

        def __enter__(self):
            return self

        def __exit__(self):
            pass

        def __hash__(self):
            return hash(self.head)+hash(self.symbol)

        def __eq__(self, other):
            if other.symbol == self.symbol:
                assert other.head == self.head
            return other.symbol == self.symbol

        def str(self):
            return self.symbol

        def verbose(self):
            return self.symbol

        @staticmethod
        def reset():
            gensym.sym_generator = ("g%d" % i for i in range(10**10))
            gensym._instances = WeakValueDictionary()


    class unboundgensym(gensym):
        __slots__ = ["sym_generator", "head", "symbol","number_inst", "pos_symbol"]
        sym_generator = ("X%d" % i for i in range(10**10))

        def __init__(self, head="", address=None, symbol=""):
            try:
                self.number_inst += 1
            except:
                self.symbol = self.head = next(self.sym_generator)
                self.number_inst = 1
                self.pos_symbol = symbol

        def __str__(self):
            return self.symbol

        def __repr__(self):
            return repr(self.__str__())

        def __enter__(self):
            return self

        def __exit__(self):
            pass

        def __hash__(self):
            return hash(self.symbol)

        def __eq__(self, other):
            return other.symbol == self.symbol

        def str(self):
            return self.symbol

        def verbose(self):
            return self.symbol

        @staticmethod
        def reset():
            unboundgensym.sym_generator = ("X%d" % i for i in range(10**10))
            unboundgensym._instances = WeakValueDictionary()

else:

    class gensym:
        _instances = WeakValueDictionary()
        sym_generator = ("g%d" % i for i in range(10**10))

        def __new__(cls, *args, **kargs):
            if 'head' not in kargs:
                raise ValueError("pass head")

            if 'address' in kargs:
                base = kargs['address'][:-1]
                backupkey = (cls, kargs['head'], tuple(base))
                key = (cls, kargs['head'], tuple(kargs['address']))
            else:
                backupkey = None
                key = (cls, kargs['head'])

            if backupkey and backupkey in cls._instances:
                ### keep pushing the representation down
                inst = cls._instances.setdefault(key, cls._instances[backupkey])
                ### update the symbol? 
            elif key in cls._instances:
                ### we've already seen this one
                inst = cls._instances.setdefault(key, cls._instances[key])
            else:
                ### new
                inst = cls._instances.setdefault(key, super(gensym, cls).__new__(cls))
            return inst

        def __init__(self, head="", address="", symbol=""):
            try:
                self.number_inst += 1
            except:
                self.head = head
                self.symbol = next(self.sym_generator)
                self.number_inst = 1
                self.pos_symbol = symbol

        def __str__(self):
            return self.symbol

        def __repr__(self):
            return repr(self.__str__())

        def __enter__(self):
            return self

        def __exit__(self):
            pass

        def __hash__(self):
            return hash(self.head)+hash(self.symbol)

        def __eq__(self, other):
            if other.symbol == self.symbol:
                assert other.head == self.head
            return other.symbol == self.symbol

        def str(self):
            return self.symbol

        def verbose(self):
            return self.symbol

        @staticmethod
        def reset():
            gensym.sym_generator = ("g%d" % i for i in range(10**10))
            gensym._instances = WeakValueDictionary()


    class unboundgensym(gensym):
        sym_generator = ("X%d" % i for i in range(10**10))

        def __init__(self, head="", address=None, symbol=""):
            try:
                self.number_inst += 1
            except:
                self.symbol = self.head = next(self.sym_generator)
                self.number_inst = 1
                self.pos_symbol = symbol

        def __str__(self):
            return self.symbol

        def __repr__(self):
            return repr(self.__str__())

        def __enter__(self):
            return self

        def __exit__(self):
            pass

        def __hash__(self):
            return hash(self.symbol)

        def __eq__(self, other):
            return other.symbol == self.symbol

        def str(self):
            return self.symbol

        def verbose(self):
            return self.symbol

        @staticmethod
        def reset():
            unboundgensym.sym_generator = ("X%d" % i for i in range(10**10))
            unboundgensym._instances = WeakValueDictionary()    

def is_bound(x):
    return not isinstance(x, unboundgensym) and isinstance(x, gensym)

def reset():
    gensym.reset()
    unboundgensym.reset()
