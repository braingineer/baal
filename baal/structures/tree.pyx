from __future__ import print_function
import re


tag_exceptions = set(["-LRB-", "-RRB-", "-LSB-",
                                "-RSB-", "-LCB-", "-RCB-"])


cdef dict opmap = {"insert_left": INSERT_LEFT,
                   "insert_right": INSERT_RIGHT,
                   "spine": SPINE,
                   "sub": SUBSTITUTE,
                   "lex": LEXICAL}

cdef class TREEOP:
    def __init__(self, OPTYPE optype=SPINE, tuple gorn=None, str hlf=None):
        self.optype = optype
        self.gorn = gorn
        self.hlf = hlf

cdef class Tree:
    def __init__(self, str symbol, list children=None, Tree parent=None,
                       OPTYPE op=SUBSTITUTE):
        self.symbol = symbol
        self.parent = parent
        self.op = TREEOP(op)
        self.target = TREEOP()
        self.spine_index = -1
        self.iterator_index = -1
        if op == LEXICAL:
            self.head = symbol
        else:
            self.head = None

        if children == None:
            self.children = []
        else:
            self.children = children


    @classmethod
    def remake(cls, Tree subtree, str opname="spine", Tree parent=None):
        children = map(Tree.remake, subtree.children)
        out = cls(subtree.symbol, children, parent, opmap[opname])
        return out

    @classmethod
    def lex(cls, str symbol, Tree parent):
        return cls(symbol, [], parent, LEXICAL)

    @classmethod
    def sub_site(cls, str symbol, Tree parent):
        return cls(symbol, [], parent, SUBSITE)

    @classmethod
    def insertion_tree(cls, str symbol, str opname, list children):
        out = cls(symbol, children, None, opmap[opname])
        return out

    @classmethod
    def raw(cls, str symbol, list children, Tree parent):
        cdef OPTYPE op
        if symbol == "":
            if len(children)==1:
                symbol = "ROOT"
            else:
                raise Exception("bad tree")

        if symbol[0] == "*":
            op = INSERT_RIGHT
            symbol = symbol[1:]
        elif symbol[-1] == "*":
            op = INSERT_LEFT
            symbol = symbol[:-1]
        else:
            op = SUBSTITUTE

        if "-" in symbol and symbol not in tag_exceptions:
            symbol = symbol.split("-")[0]

        symbol = symbol.split("=")[0].split("|")[0]

        return cls(symbol=symbol, children=children, op=op, parent=parent)

    @classmethod
    def from_string(cls, str in_str):
        return from_string(in_str)

    property addressbook:
        def __get__(self):
            if self._addressbook == None:
                self.addressbook = {}
            return self._addressbook

        def __set__(self, shared):
            self._addressbook = shared
            self._addressbook[self.gorn] = self
            for child in self.children:
                child.addressbook = shared

    property gorn:
        def __get__(self):
            if self._gorn == None:
                self.gorn = (0,)
            return self._gorn

        def __set__(self, g):
            assert self._gorn == None
            self._gorn = g
            for i, child in enumerate(self.children):
                child.gorn = self._gorn + (i,)

    @property
    def is_lexical(self):
        return self.op.optype == LEXICAL

    cpdef set_op(self, op_name):
        self.op.optype = opmap[op_name]


    def __str__(self):
        if self.op.optype == LEXICAL:
            assert len(self.children) == 0
            return self.symbol
        return self.symbol + "[\t" + \
                ",\n\t".join(map(str, self.children)) + "]"


    def __iter__(self):
        self.iterator_index = 0
        self.iterator_object = sorted(self.addressbook.items())
        return self

    def __next__(self):
        if self.iterator_index < len(self.iterator_object):
            self.iterator_index += 1
            return self.iterator_object[self.iterator_index-1][1]
        else:
            raise StopIteration()

    def visitor(self, func):
        return func(self)

cdef inline int tree_starting(str x):
    if x[0] == "(":
        return 1
    else:
        return 0

cdef inline int tree_ending(str x):
    if x[0] == ")":
        return 1
    else:
        return 0


cdef Tree from_string(str in_str):
    """
    modeled from NLTK's version in their class.
    Assuming () as open and close patterns
    e.g. (S (NP (NNP John)) (VP (V runs)))

    TODO: we want parent to be insertion, foot to be foot. fix.

    """
    cdef list stack = [(None, [])]
    cdef str token;

    token_re = re.compile("\(\s*([^\s\(\)]+)?|\)|([^\s\(\)]+)")
    for match in token_re.finditer(in_str):
        token = match.group()
        #if "-" in token and token[0] != '-' and in_str[match.start()-1] != " ":
        #    token, semantic = token.split("-")[0], token.split("-")[1]

        # Case: tree/subtree starting. prepare structure
        if tree_starting(token):
            stack.append((token[1:].lstrip(), []))
        # Case: tree/subtree is ending. make sure it's buttoned up
        elif tree_ending(token):
            label, children = stack.pop()
            stack[-1][1].append(Tree.raw(symbol=label,
                                         children=children,
                                         parent=None))
        # Case: leaf node.
        else:
            stack[-1][1].append(Tree.lex(token, None))

    try:
        assert len(stack) == 1
        assert len(stack[0][1]) == 1
        assert stack[0][0] is None
    except AssertionError as e:
        print(stack)
        print(in_str)
        raise AssertionError

    resulting_tree = stack[0][1][0]
    if isinstance(resulting_tree, list):
        resulting_tree = resulting_tree[0]

    return clean_tree(resulting_tree)


cdef int none_filter(Tree tree):
    try:
        if '-NONE-' in tree.symbol:
            return 0
    except AttributeError as e:
        print(tree, e)

    try:
        if (len(tree.children) == 1 and
            '-NONE-' in tree.children[0].symbol):
            return 0
    except AttributeError as e:
        print(tree, tree.children[0], e)

    return 1

cdef Tree clean_tree(Tree tree):
    """
        Clean the tree. This is called by from_string
        From_String doesn't annotate the tree, it just makes makes the structure
        this is to annotate with the relevant information
    """
    tree.children = map(clean_tree, filter(none_filter, tree.children))

    cdef int i, n;
    n = len(tree.children)
    for i in range(n):
        tree.parent = tree
        if tree.children[i].head != None:
            tree.head = tree.children[i].head
            tree.spine_index = i
        else:
            assert tree.op == SUBSTITUTE

    return tree


cpdef run_tests():

    cdef str test_str = """(ROOT(S(NP (NNP Man))(VP (VBD dressed)(PP (IN in)(NP(NP (NN leather) (NNS chaps))(CC and)(NP (JJ purple) (NN -NONE-) (NNS stands))))(PP (IN in)(NP(NP (NN front))(PP (IN of)(NP (NNS onlookers))))))(. .)))"""

    cdef str test_str2 = """(ROOT(S(NP(NP (DT The) (NN boy))(VP (VBG laying)(S(VP (VB face)(PRT (RP down))(PP (IN on)(NP (DT a) (NN skateboard)))))))(VP (VBZ is)(VP (VBG being)(VP (VBN pushed)(PP (IN along)(NP (DT the) (NN ground)))(PP (IN by)(NP (DT another) (NN boy))))))(. .)))"""

    cdef Tree tree
    cdef Tree t

    tree = from_string(test_str)

    print(tree)

    for t in tree:
        print(t.gorn, t.symbol, t.head)


    tree = from_string(test_str2)

    print(tree)

    for t in tree:
        print(t.gorn, t.symbol, t.head)

if __name__ == "__main__":
    run_tests()
