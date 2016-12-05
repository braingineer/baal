from __future__ import print_function
import re

cdef enum OPTYPE:
    INSERT_LEFT, INSERT_RIGHT, SUBSTITUTE,
    SUBSITE, LEXICAL, SPINE

cdef set tag_exceptions

cdef class TREEOP:
    cdef public tuple gorn
    cdef public str hlf
    cdef public OPTYPE optype

cdef class Tree:
    cdef public str symbol
    cdef public str head
    cdef public str hlf
    cdef public list children
    cdef public int spine_index
    cdef public Tree parent
    cdef public int head_index
    cdef public TREEOP target
    cdef int iterator_index
    cdef list iterator_object
    cdef TREEOP op
    cdef dict _addressbook
    cdef tuple _gorn
    cdef public bint adjunct
    cdef public bint complement
    cpdef set_op(self, op_name)

cdef inline int tree_starting(str x)
cdef inline int tree_ending(str x)


cdef Tree from_string(str in_str)


cdef int none_filter(Tree tree)

cdef Tree clean_tree(Tree tree)

cpdef run_tests()
