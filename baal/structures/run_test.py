import pyximport
pyximport.install()


import test


test.printit()
#print(test.workone)


x = test.Tester.one("woo")
x = test.Tester.one()


#x.f_test2("foo")
x.f_test3(False)
