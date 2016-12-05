import sys
sys.path.append("../baal/structures")

import pyximport
pyximport.install()

#sys.path.append("../baal/induce")

import tree

tree.Tree("test")

tree.run_tests()

import enrich

