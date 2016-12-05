import pyximport
pyximport.install()

import sys
sys.path.append("../baal/structures")

import tree

tree.Tree("test")

tree.run_tests()
