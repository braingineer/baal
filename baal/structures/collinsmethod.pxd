from functools import wraps
from tree cimport Tree

cdef dict headruleset
cdef dict headrule_functions
cdef dict complementruleset

cdef is_complement(Tree tree, Tree child)
cdef mark_complements(Tree tree)
cdef ccs_are_hard(Tree tree, int c_i)

cdef cc_context(list children, tuple headlist, Tree parent)

cdef left(list children, tuple headlist, Tree parent)

cdef right(list children, tuple headlist, Tree parent)

cdef leftdis(list children, tuple headlist, Tree parent)

cdef rightdis(list children, tuple headlist, Tree parent)

cdef leftexcept(list children, tuple headlist, Tree parent)

cdef rightexcept(list children, tuple headlist, Tree parent)

cdef backward_enumeration(list vals)

