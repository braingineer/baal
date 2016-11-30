import pyximport
pyximport.install()

from copy import deepcopy
from baal.structures.tree_ops import *

def lazyrun(f, s):
    try:
        print("[Pass] {:<20} --- {}".format(s, f()))
    except:
        print("[Fail] {:<20}".format(s))

def lazyassert(f, v, s):
    try:
        out = f()
        assert out == v
        print("[Pass] {:<20} --- {}".format(s, out))
    except:
        print("[Fail] {:<20} --- {}".format(s, out))


print("==tests for cython implementation of baal structures==")

print("[*] AttachmentPoint Tests")
pt = AttachmentPoint.root()
pt2 = deepcopy(pt)

lazyassert(lambda: pt.left_insert, -0.01, "Left Insertion")
lazyassert(lambda: pt.left_sub, -1.0, "Left Substitution")
lazyassert(lambda: pt.right_insert, 0.01, "Right Insertion")
lazyassert(lambda: pt.right_sub, 1.0, "Right Substitution")

lazyassert(lambda: pt2.left_sub, -1.0, "Left Substitution")
lazyassert(lambda: pt2.left_insert, -1.01, "Left Insertion")
lazyassert(lambda: pt2.right_sub, 1.0, "Right Substitution")
lazyassert(lambda: pt2.right_insert, 1.01, "Right Insertion")


# cases = [(POS.root, POINTS.substitute, OPS.substitute),
#          (POS.np, POINTS.insert, OPS.left_insert)]
# import pdb
# #pdb.set_trace()
# for pos_sym, point_type, op_type in cases:
#     print(pos_sym, point_type, op_type)
#     pt = AttachmentPoint.make(pos=pos_sym, pt_type=point_type)
#     op = MakeOp(op_type, pos_sym)
#     if pt.match(op, 1):
#         print("SUCCESS")
#     else:
#         print("FAILURE")
