from weakref import WeakValueDictionary


class Flyweight(object):
    __slots__ = ["_instances","__weakref__"]
    _instances = WeakValueDictionary()

    def __new__(cls, *args, **kargs):
        key = (cls, args, tuple(kargs.items()))
        return cls._instances.setdefault(key,
                                         super(Flyweight,cls).__new__(cls))


def test():
    class Spam(Flyweight):

        def __init__(self, a, b):
            self.a = a
            self.b = b

    class Egg(Flyweight):

        def __init__(self, x, y):
            self.x = x
            self.y = y

    assert Spam.get_instance(1, 2) is Spam.get_instance(1, 2)
    assert Egg.get_instance('a', 'b') is Egg.get_instance('a', 'b')
    assert Spam.get_instance(1, 2) is not Egg.get_instance(1, 2)

    # Subclassing a flyweight class
    class SubSpam(Spam):
        pass

    assert SubSpam.get_instance(1, 2) is SubSpam.get_instance(1, 2)
    assert Spam.get_instance(1, 2) is not SubSpam.get_instance(1, 2)


def test2():
    class gensym(Flyweight):
        genner = ("g%d" % i for i in xrange(100000))

        def __init__(self, headword):
            if len(self.__dict__) == 0:
                self.headword = headword
                self.symbol = next(self.genner)

        def __str__(self):
            return "%s => %s" % (self.headword, self.symbol)

        def __repr__(self):
            return self.__str__(self)

    one = gensym('brian')
    two = gensym('is')
    three = gensym('testing')

    oneagain = gensym('brian')


    print one
    print two
    print three
    print oneagain

    assert one == oneagain

if __name__ == "__main__":
    # test()
    test2()
