from tree cimport Tree
cimport collinsmethod as CollinsMethod
from copy import deepcopy

try:
    range = xrange
except:
    pass

cdef class HLFGenerator:
    cdef int i
    cdef str _id

    def __init__(self, str _id="g"):
        self.i = 0
        self._id = _id

    cdef reset(self):
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        out = "{}{}".format(self._id, self.i)
        self.i += 1
        return out

#hlf_gen = ("g{}".format(i) for i in range(10**10))

cdef HLFGenerator hlf_gen = HLFGenerator()

cdef _annotate(Tree tree):
    if len(tree.children) == 0:
        assert hasattr(tree, 'head')
        assert tree.head == tree.symbol
        print("{} is leaf".format(tree.symbol))

    elif len(tree.children) == 1 and tree.children[0].is_lexical:
        print("{}.{}  is lexical".format(tree.symbol, tree.children[0].symbol))
        tree.spine_index = 0
        tree.head = tree.children[0].head

    else:
        print("going into {}'s {} children".format(tree.symbol, len(tree.children)))
        map(_annotate, tree.children)

        print("leaving {}'s {} children".format(tree.symbol, len(tree.children)))
        select_head(tree)
        mark_dependencies(tree)

    return tree


cdef select_head(Tree tree):
    cdef list rules
    cdef tuple rule, argset
    cdef int i, head_index
    cdef str search_method
    cdef Tree head_child

    # if len(tree.children) == 0:
    #     return
    # elif len(tree.children) == 1 and tree.children[0].is_lexical:
    #     tree.spine_index = 0
    #     tree.head = tree.children[0].head
    #     return

    rules = CollinsMethod.headruleset[tree.symbol]

    rules = [('cc_context',"CC",)] + rules

    for i, rule in enumerate(rules):
        search_method, argset = rule[0], rule[1:]
        func = CollinsMethod.headrule_functions[search_method]
        head_index, success = func(tree.children, argset, tree)
        is_last_rule = i+1 == len(rules)
        if head_index is not None and (success or is_last_rule):
            head_child = tree.children[head_index]
            head_child.set_op("spine")

            tree.spine_index = head_index
            tree.head = head_child.head

            return

    raise Exception("should not reach end of select_head")

cdef _spine_head(Tree tree, int index):
    return tree.spine_index == index

cdef _prn_head(Tree tree, int index):
    if tree.symbol == "PRN" and (tree.spine_index == index or
                                 index + 1 == len(tree.children)):
        return True
    else:
        return False

cdef adjunct_condition(Tree parent, Tree child, int index):
    if child.complement or _spine_head(parent, index):
        return False
    else:
        return True


cdef mark_dependencies(Tree tree):
    deps = CollinsMethod.mark_complements(tree)

    for c_i, child in enumerate(tree.children):
        if adjunct_condition(child, tree, c_i):
            child.adjunct = True


    spine = tree.children[tree.spine_index]
    spine.adjunct = False
    spine.complement = False


cdef decompose(tree, hlf=None, gorn=None, is_root=True):
    tree.hlf = hlf or next(hlf_gen)
    left, right = [], []
    gorn = gorn or (0,)
    offset = 0

    saw_spine = 0

    print("| {}; #children={}".format(tree.symbol, len(tree.children)))

    for i, child in enumerate(tree.children):
        print("+ inside child: {}".format(child))
        if tree.spine_index == i:
            saw_spine = 1
            print("+ + inside spine")
            tree.spine_index -= offset
            assert not child.adjunct
            assert not child.complement
        elif child.complement:
            print("+ + inside complement")
            subtree = Tree.remake(child, "sub")
            subtree.target.gorn = gorn+(i-offset,)
            subtree.target.hlf = tree.hlf
            tree.children[i] = Tree.sub_site(child.symbol, tree)
            if i < tree.spine_index:
                print("+ + + appending left")
                left.append(subtree)
            else:
                print("+ + + appending to right")
                right.append(subtree)
        elif child.adjunct:
            print("+ + inside adjunct")
            if i < tree.spine_index:
                opname = "insert_left"
            else:
                opname = "insert_right"

            subtree = Tree.insertion_tree(tree.symbol, opname,
                                          children=[Tree.remake(child)])
            subtree.spine_index = 0
            subtree.target.gorn = gorn
            subtree.target.hlf = tree.hlf
            offset += 1

            if i < tree.spine_index:
                print("+ + + appending left")
                left.append(subtree)
            else:
                print("+ + + appending to right")
                right.append(subtree)
        else:
            print("+ + Was not anything")


    tree.children = filter(lambda c: not c.adjunct, tree.children)


    out = [tree] if is_root else []

    print("{} spine, {} in left, {} in right".format(saw_spine, len(left), len(right)))

    left.reverse()
    for subtree in left:
        subtree, more_out = decompose(subtree)
        out.extend(more_out)

    if tree.spine_index >= 0:
        subtree, more_out = decompose(tree.children[tree.spine_index],
                                               hlf=tree.hlf,
                                               gorn=gorn+(tree.spine_index,),
                                               is_root=False)
        out.extend(more_out)

    for subtree in right:
        subtree, more_out = decompose(subtree)
        out.extend(more_out)

    return tree, out


cpdef run_tests():

    cdef str test_str = """(ROOT(S(NP (NNP Man))(VP (VBD dressed)(PP (IN in)(NP(NP (NN leather) (NNS chaps))(CC and)(NP (JJ purple) (NN -NONE-) (NNS stands))))(PP (IN in)(NP(NP (NN front))(PP (IN of)(NP (NNS onlookers))))))(. .)))"""

    cdef str test_str2 = """(ROOT(S(NP(NP (DT The) (NN boy))(VP (VBG laying)(S(VP (VB face)(PRT (RP down))(PP (IN on)(NP (DT a) (NN skateboard)))))))(VP (VBZ is)(VP (VBG being)(VP (VBN pushed)(PP (IN along)(NP (DT the) (NN ground)))(PP (IN by)(NP (DT another) (NN boy))))))(. .)))"""

    tree = _annotate(Tree.from_string(test_str2))
    print(decompose(tree))
