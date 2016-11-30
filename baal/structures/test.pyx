cpdef str teststring = "work =("

cdef enum enumtest:
    k_one, k_two,
    k_three, k_four, k_five, k_six

def printit():
    print(teststring)
    print(k_three)
    print(k_one)
    print(k_six)



cdef test8(str *foo):
    print(*foo.upper())

cdef class Tester(object):
    cpdef f_test3(self, bint xin):
        print(type(xin))
        print(xin)

    @classmethod
    def one(cls, str blah=None):
        blah = blah or "foo"
        print(blah)
        test8(*blah)
        return cls()
