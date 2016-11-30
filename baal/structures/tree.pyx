from __future__ import print_function
import re

cdef enum OPTYPE:
    INSERT_LEFT, INSERT_RIGHT, SUBSTITUTE, LEXICAL, SPINE


cdef set tag_exceptions = set(["-LRB-", "-RRB-", "-LSB-",
                                "-RSB-", "-LCB-", "-RCB-"])


cdef class Tree:
    cdef public str symbol
    cdef public str head
    cdef list children
    cdef int spine_index
    cdef int iterator_index
    cdef list iterator_object
    cdef Tree parent
    cdef OPTYPE op
    cdef dict _addressbook
    cdef tuple _gorn



    def __init__(self, str symbol, list children=None, Tree parent=None,
                       OPTYPE op=SUBSTITUTE):
        self.symbol = symbol
        self.parent = parent
        self.op = op
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
    def lex(cls, str symbol, Tree parent):
        return cls(symbol, [], parent, LEXICAL)

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

    def __str__(self):
        if self.op == LEXICAL:
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



if __name__ == "__main__":
    cdef str test_str = """(ROOT(S(NP (NNP Man))(VP (VBD dressed)(PP (IN in)(NP(NP (NN leather) (NNS chaps))(CC and)(NP (JJ purple) (NN -NONE-) (NNS stands))))(PP (IN in)(NP(NP (NN front))(PP (IN of)(NP (NNS onlookers))))))(. .)))"""

    tree = from_string(test_str)

    print(tree)

    for t in tree:
        print(t.gorn, t.symbol, t.head)
