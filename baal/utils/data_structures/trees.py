from baal.utils.sugar import memoized
from baal.utils.general import cprint, bcolors, cformat, nonstr_join
from baal.utils import config
import re, types, logging
try:
    from nltk.tree import Tree as TKTree
except ImportError:
    print("Warning: You don't have NLTK. One method won't work in Tree")

logger = logging.getLogger("treedebugging")

def from_string(in_str):
    """
    modeled from NLTK's version in their class.
    Assuming () as open and close patterns
    e.g. (S (NP (NNP John)) (VP (V runs)))

    TODO: we want parent to be insertion, foot to be foot. fix.

    """
    tree_starting = lambda x: x[0] == "("
    tree_ending = lambda x: x[0] == ")"
    token_re = re.compile("\(\s*([^\s\(\)]+)?|\)|([^\s\(\)]+)")
    stack = [(None, [])]
    for match in token_re.finditer(in_str):
        token, semantic = match.group(), ""
        #if "-" in token and token[0] != '-':
        #    token, semantic = token.split("-")[0], token.split("-")[1]

        # Case: tree/subtree starting. prepare structure
        if tree_starting(token):
            stack.append((token[1:].lstrip(), [], semantic))
        # Case: tree/subtree is ending. make sure it's buttoned up
        elif tree_ending(token):
            label, children, semantic = stack.pop()
            stack[-1][1].append(Tree(symbol=label,
                                     children=children,
                                     parent=stack[-1][0],
                                     semantic=semantic))
        # Case: leaf node.
        else:
            stack[-1][1].append(token)

    try:
        assert len(stack) == 1
        assert len(stack[0][1]) == 1
        assert stack[0][0] is None
    except AssertionError as e:
        print stack
        print in_str
        raise AssertionError

    resulting_tree = stack[0][1][0]
    if isinstance(resulting_tree, types.ListType):
        resulting_tree = resulting_tree[0]

    assert isinstance(resulting_tree, Tree)

    return clean_tree(resulting_tree, [0], {tuple([0]):resulting_tree})


def clean_tree(root_tree, address, addressbook):
    """
        Clean the tree. This is called by from_string
        From_String doesn't annotate the tree, it just makes makes the structure
        this is to annotate with the relevant information
    """
    logger = logging.getLogger('trees')
    logger.debug(root_tree)

    if "*" in root_tree.symbol:
        root_tree.adjunct = True
        root_tree.direction = "right" if root_tree.symbol[0] == "*" else "left"
        root_tree.symbol = root_tree.symbol.replace("*", "")

    logger.debug('starting child iter for %s' % root_tree.symbol)

    for c_i, child in enumerate(root_tree.children):
        next_address = address+[c_i]
        if isinstance(child, Tree):

            if len(child.children) > 0:
                # Interior node
                logger.debug('diving into child')
                logger.debug('specifically: %s' % child)
                child, addressbook = clean_tree(child, next_address, addressbook)
                root_tree.children[c_i] = child

                if child.head is not None:
                    root_tree.head = child.head
                    root_tree.spine_index = c_i
                #else:
                #    raise AlgorithmicException, "Only making initial trees here"

            else:
                # Substitution point
                child.complement = True

        else:
            # Found the head
            child = child.lower()
            child = Tree(symbol=child, parent=root_tree.symbol)
            child.lexical = True
            root_tree.children[c_i] = child
            head = child
            head.lexical = True
            root_tree.head = head.symbol
            root_tree.spine_index = c_i

        child.parent = root_tree.symbol
        addressbook[tuple(next_address)] = child


    if all([child.complement for child in root_tree.children]):
        logger.warning("Condition: this subtree has no lexical items. "+
                       "This condition should be indicating co-heads. FIX")
        root_tree.head = None


    # Debugging stuff
    try:
        logger.debug(root_tree.head)
        logger.debug(root_tree.head is None)
        logger.debug(type(root_tree.head))
        logger.debug(root_tree.symbol)
        logger.debug(addressbook)
        assert root_tree.head is None or len(root_tree.head) > 0, type(root_tree)
    except AttributeError as e:
        logger.debug(root_tree)
        raise e

    return root_tree, addressbook

def strings2entries(tree_strings):
        ret = []
        for tree_string in tree_strings:
            if "RB-" in tree_string: continue
            ret.append(Entry.make(bracketed_string=tree_string,
                                  correct_root=True))
        return ret

class Tree(object):
    def __init__(self, symbol, children=[], parent="", semantic=None):
        self.symbol = symbol
        self.children = children
        self.parent = parent
        self.head = ""
        self.spine_index = -1
        self.complement = False
        self.adjunct = False
        self.lexical = False
        self.depth = 0
        self.direction = ""
        self.semantictag = semantic
        self.substitutions = []
        self.adjuncts = []
        self.saved_repr = None

    @classmethod
    def make(cls, tree=None, bracketed_string=None, correct_root=False):
        """
            Instantiation check order: bracketed_string, lexical_item, root.
            so leave bracketed_string and lexical_item empty if root.
            lexical_item will only work if you have a lexicon
            and even then, it expects the lexicon to yield bracketed strings
        """
        if not tree and not bracketed_string:
            raise TypeError("tree.instantiate takes either an existing tree"
                            + " or a bracketed string structure to convert")

        if bracketed_string:
            new_tree, addressbook = from_string(bracketed_string)
        elif tree:
            #I don't think I'll ever use this, but just in case
            new_tree, addressbook = copy(tree)
        else:
            raise TypeError("I don't know how you got here, but this is wrong")

        if correct_root:
            if len(new_tree.symbol) == 0:
                assert len(new_tree.children) == 1
                new_tree, addressbook = new_tree.children[0].clone()

        new_tree.correct_projections()

        return new_tree, addressbook

    def correct_projections(self):
        if len(self.children) == 1:
            if self.children[0].symbol == self.symbol:
                self.children = self.children[0].children
        for child in self.children:
            child.correct_projections()

    def to_nltk_tree(self):
        def f(x):
            if len(x.children) == 1 and x.children[0] is None:
                return "SUB({})".format(x.symbol)

            return [child.symbol if child.lexical else child.to_nltk_tree()
                                 for child in x.children if child is not None]

        return TKTree(self.symbol, f(self))
        #if self.lexical:
        #    return TKTree(self.symbol, [self.])


    def clone(self, address=None, addressbook=None,prefab_children=None,child_offset=0):
        """ Copy the tree and make the addressbook along the way """
        #print("Cloning myself: {}".format(self))
        address = address or [0]
        addressbook = addressbook or {}

        if prefab_children is not None:
            new_children = prefab_children
        else:
            new_children = []
            for c_i, child in enumerate(self.children):
                new_child, addressbook = child.clone(address+[c_i+child_offset], addressbook)
                new_children.append(new_child)
        new_tree = Tree(self.symbol, children=new_children, parent=self.parent)
        new_tree.head = self.head
        new_tree.spine_index = self.spine_index
        new_tree.complement = self.complement
        new_tree.adjunct = self.adjunct
        new_tree.lexical = self.lexical
        new_tree.direction = self.direction

        addressbook[tuple(address)] = new_tree

        return new_tree, addressbook

    def insert_into(self, op_tree, op_addr, recur_addr=[0], addrbook={}):
        """
            Input: an insertion tree:
                    the root symbol matches a frontier interior symbol
                        (an ancestor of the further left/right lexical)
        """

        # /// Check for base case
        if len(op_addr) == 0:
            if op_tree.direction == "left":
                # adding to the left side of the node
                other_children = []
                for c_i, child in enumerate(op_tree.children):
                    new_child, addrbook = child.clone(recur_addr+[c_i], addrbook)
                    other_children.append(new_child)
                    new_child.parent = self.symbol
                new_tree, addrbook = self.clone(recur_addr, addrbook, child_offset=len(op_tree.children))

                new_tree.children = other_children + new_tree.children

                if new_tree.spine_index >= 0:
                    new_tree.spine_index += len(op_tree.children)

            else:
                # Adding to the right side of the node
                new_tree, addrbook = self.clone(recur_addr, addrbook)
                other_children = []
                b_i = len(self.children)
                for c_i, child in enumerate(op_tree.children):
                    new_child, addrbook = child.clone(recur_addr+[b_i+c_i], addrbook)
                    new_child.parent = new_tree.symbol
                    other_children.append(new_child)
                new_tree.children += other_children
            return new_tree, addrbook


        # /// Recursive case. Clone children, recurse on operation child
        next_addr, new_op_addr = op_addr[0], op_addr[1:]
        new_children = []
        for c_i, child in enumerate(self.children):
            if c_i == next_addr:
                new_child, addrbook = child.insert_into(op_tree, new_op_addr, recur_addr+[c_i], addrbook)
                new_child.parent = self.symbol
                new_children.append(new_child)
            else:
                new_child, addrbook = child.clone(recur_addr+[c_i], addrbook)
                new_children.append(new_child)

        # /// Return ourself cloned
        return self.clone(recur_addr, addrbook, new_children)

    def substitute_into(self, op_tree, op_addr, recur_addr=[0], addrbook={}):
        """
            Input: an substitution tree:
                    the root symbol matches a frontier symbol
                        (a frontier symbol just beyond left/right lexical)
        """
        # /// Check for base case
        if len(op_addr) == 0:
            new_tree, addrbook = op_tree.clone(recur_addr, addrbook)
            #new_tree.complement = False
            return new_tree, addrbook

        # /// Recursive case. Clone children, recurse on operation child
        next_addr, op_addr = op_addr[0], op_addr[1:]
        new_children = []
        for c_i, child in enumerate(self.children):
            if c_i == next_addr:
                new_child, addrbook = child.substitute_into(op_tree, op_addr,
                                                            recur_addr+[c_i], addrbook)
                new_child.parent = self.symbol
                new_children.append(new_child)
            else:
                new_child, addrbook = child.clone(recur_addr+[c_i], addrbook)
                new_children.append(new_child)
        # /// Return ourself cloned
        return self.clone(recur_addr, addrbook, new_children)

    def excise_substitutions(self):
        # print "inside excise substitutions"
        new_subtrees = []
        for subst in self.substitutions:
            # print "excsising %s" % subst
            new_subst = subst
            new_subtrees.append(new_subst)

            ind = self.children.index(subst)
            self.children[ind] = Tree(subst.symbol)
            self.children[ind].complement = True
            self.children[ind].parent = self.symbol

        self.substitutions = []
        return new_subtrees

    def excise_adjuncts(self):
        new_subtrees = []
        for adj_tree in self.adjuncts:
            ind = self.children.index(adj_tree)
            self.children[ind] = None
            adj_wrapper = Tree(symbol=self.symbol, children=[adj_tree])
            adj_wrapper.adjunct = True
            adj_wrapper.direction = "left" if ind < self.spine_index else "right"
            new_subtrees.append(adj_wrapper)

        self.adjuncts = []
        return new_subtrees

    def __getitem__(self, k):
        if k>=len(self.children):
            return None
        return self.children[k]

    def __str__(self):
        if config.ChartSettings().verbose:
            return self.verbose_string()
        else:
            return "(%s" % self.symbol + ("-> %s" % self.children if len(self.children) > 0 else ")")
        return "From: %s; I am %s, with %s children: %s" % (self.parent, self.symbol, len(self.children), "\n".join([str(child) for child in self.children]))

    def verbose_string(self,depth=1):
        # return self._simple_str()
        p = lambda x: bcolors.OKBLUE+x+bcolors.ENDC
        b = lambda x: bcolors.BOLD+x+bcolors.ENDC
        g = lambda x: bcolors.OKGREEN+x+bcolors.ENDC
        r = lambda x: bcolors.FAIL+x+bcolors.ENDC

        if self.complement:
            this_symbol = "%s(%s)" % (p("Subst"), g(self.symbol))
        elif self.adjunct:
            d = "[->]" if self.direction == "right" else "[<-]"
            this_symbol = "%s(%s)" % (p(d+"Ins"), g(self.symbol))
        elif self.lexical:
            this_symbol = "%s(%s)" % (p("lex"), g(self.symbol))
        else:
            this_symbol = self.symbol

        d = depth
        s = lambda x: ("{:%d}" % (x*d)).format("")
        if len(self.children) > 0:
            tree_print = "\n"+s(4)+b("%s{"%d)
            tree_print += "{:-<3}->\n".format(this_symbol)
            tree_print += s(5) + r("[")
            tree_print += (",\n"+s(6)).join([x.verbose_string(depth+1) for x in self.children])
            tree_print += r("]") + "\n"
            tree_print += s(4)+b("}")
            return tree_print
        else:
            return this_symbol

    def __repr__(self):
        if self.saved_repr is not None:
            return self.saved_repr
        if self.lexical:
            return "lex/" + self.symbol
        if self.adjunct:
            sym = "<" if self.direction=="left" else ">"
            sym = "ins/"+sym+"/"+self.symbol
        elif self.complement:
            sym = "sub/"+self.symbol
        elif self.parent is None:
            sym = "root/"+self.symbol
        else:
            sym = self.symbol

        this_str = "(%s" % sym
        for child in self.children:
            if child is None:
                continue
            this_str += " %s" % repr(child)
        this_str+=")"
        self.saved_repr = this_str
        return this_str

    def save_str(self):
        if self.lexical:
            return self.symbol.lower()

        if self.adjunct:
            sym = (self.symbol, "*") if self.direction=="left" else ("*", self.symbol)
            sym = "{}{}".format(*sym)
        else:
            sym = self.symbol

        this_str = "(%s" % sym
        for child in self.children:
            if child is None:
                continue
            this_str += " %s" % child.save_str()
        this_str+=")"
        return this_str


    def lexical_string(self):
        if self.lexical:
            return self.symbol.lower()
        else:
            return " ".join([child.lexical_string() for child in self.children])

class Entry(object):
    def __init__(self, tree, subst_points, adjoin_points,
                 lexical, addressbook, derived):
        self.tree = tree
        self.lexical = lexical
        self.addressbook = addressbook
        self.derived = derived
        self.symbol = self.tree.symbol

        subst_points = sorted(subst_points)
        self.left_subs = []
        self.right_subs = []
        for address,subtree in subst_points:
            if self.isleft(address):
                self.left_subs.append((address, subtree))
            elif self.isright(address):
                self.right_subs.append((address, subtree))
            #else:
            #    print("CHECKING")
        #self.subst_points = left_subs+right_subs
        self.subst_points = subst_points
        self.adjoin_points = [point for point in adjoin_points
                              if self.leftfrontier(point[0])
                              or self.rightfrontier(point[0])]

    @classmethod
    def make(cls, bracketed_string, correct_root=False):
        """ Initial make. Combine will copy, not make """
        tree, addressbook = Tree.make(bracketed_string=bracketed_string)
        if correct_root:
            if len(tree.symbol) == 0:
                assert len(tree.children) == 1
                tree, addressbook = tree.children[0].clone()
        addressbook = sorted(addressbook.items())
        adjoin_points = []
        subst_points = []
        lexical = []
        for address, subtree in addressbook:
            if subtree.lexical:
                lexical.append((address, subtree))
            elif subtree.complement:
                subst_points.append((address,subtree))
            elif subtree.spine_index >= 0: # and not subtree.adjunct:
                # spine index >= 0 means it's on the spine
                adjoin_points.append((address, subtree))

        deriv_sym = "%s(%s)" % (tree.symbol, tree.head)
        derived = [(("initial"), [0], deriv_sym)]
        return cls(tree, subst_points, adjoin_points, lexical, addressbook, derived)

    @classmethod
    def sub_generate(cls, this, op, address):
        tree = this.tree
        op_tree = op.tree
        new_tree, addressbook = tree.substitute_into(op_tree, address, [0], {})
        addressbook = sorted(addressbook.items())
        adjoin_points = []
        subst_points = []
        lexical = []
        for address, subtree in addressbook:
            if subtree.lexical:
                lexical.append((address, subtree))
            elif subtree.complement:
                subst_points.append((address,subtree))
            elif subtree.spine_index >= 0:
                adjoin_points.append((address, subtree))
        deriv_sym = "%s(%s)" % (op_tree.symbol, op_tree.head)
        derived = this.derived + [("substitute", address, deriv_sym)] + \
                  [("subtree_derived", op.derived)]


        return cls(new_tree, subst_points, adjoin_points, lexical, addressbook, derived)

    @classmethod
    def ins_generate(cls, this, op, address):
        tree = this.tree
        op_tree = op.tree
        new_tree, addressbook = tree.insert_into(op_tree, address, [0], {})

        addressbook = sorted(addressbook.items())
        adjoin_points = []
        subst_points = []
        lexical = []
        for address, subtree in addressbook:
            if subtree.lexical:
                lexical.append((address, subtree))
            elif subtree.complement:
                subst_points.append((address,subtree))
            elif subtree.spine_index >= 0:
                adjoin_points.append((address, subtree))
        deriv_sym = "%s(%s)" % (op_tree.symbol, op_tree.head)
        derived = this.derived + [("insert", address, deriv_sym)] + \
                  [("ins_derived", op.derived)]
        #if "Two" in str(lexical):
        #    print("Inside generate: {}".format(new_tree.symbol))
        #    print new_tree.children
        #elif "Two" in str(addressbook):
        #    print addressbook
        #    print lexical
        #    print op_tree, op_tree.direction
        #    print tree
        return cls(new_tree, subst_points, adjoin_points, lexical, addressbook, derived)

    def get_lexical(self):
        return " ".join(subtree.symbol for _, subtree in sorted(self.lexical))


    def isleft(self, address):
        try:
            return self._isleft(tuple(address), tuple(self.lexical[0][0]))
        except IndexError as e:
            logger.debug(self.lexical)
            logger.debug(self.tree)
            raise IndexError

    def isright(self, address):
        return self._isright(tuple(address), tuple(self.lexical[-1][0]))

    def leftfrontier(self, address):
        return self._frontier(tuple(address), tuple(self.lexical[0][0]))

    def rightfrontier(self, address):
        return self._frontier(tuple(address), tuple(self.lexical[-1][0]))


    #@memoized
    def _isleft(self, addressone, addresstwo):
        for ind in range(min(len(addressone), len(addresstwo))):
            if addressone[ind] < addresstwo[ind]:
                return True
            elif addressone[ind] > addresstwo[ind]:
                return False
        return False

    #@memoized
    def _isright(self, addressone, addresstwo):
        for ind in range(min(len(addressone), len(addresstwo))):
            if addressone[ind] < addresstwo[ind]:
                return False
            elif addressone[ind] > addresstwo[ind]:
                return True
        return False

    #@memoized
    def _frontier(self, addressone, addresstwo):
        for ind in range(min(len(addressone), len(addresstwo))):
            if addressone[ind] < addresstwo[ind]:
                return False
            elif addressone[ind] > addresstwo[ind]:
                return False
        return True

    def combine(self, other_entry, edge_conditionals=(True,True), genning=True):
        left_edge, right_edge = edge_conditionals
        other_tree = other_entry.tree
        sym_match = lambda x,y: x.symbol == y.symbol
        if other_tree.adjunct:
            logger.debug("Looking at the adjunct: {}".format(other_tree))
            for address, subtree in self.adjoin_points:
                if sym_match(other_tree, subtree):
                    logger.debug("symbols match at subtree: {} \n\twith address {}".format(subtree, address))
                    # left insert says we are inserting on the left side
                    # of this (self) guy
                    # so, this means that left edge must true
                    left_insert = other_tree.direction == "left"
                    if (left_edge and (self.leftfrontier(address) or genning)
                        and left_insert):
                        logger.debug("should be yielding")
                        yield Entry.ins_generate(self, other_entry, address[1:])
                    elif (right_edge and (self.rightfrontier(address) or genning)
                          and not left_insert):
                        logger.debug("should be right yielding")
                        yield Entry.ins_generate(self, other_entry, address[1:])
        else:
            for address, subtree in self.subst_points:
                if sym_match(other_tree, subtree):
                    if left_edge and self.isleft(address):
                        yield Entry.sub_generate(self, other_entry, address[1:])
                    elif right_edge and self.isright(address):
                        yield Entry.sub_generate(self, other_entry, address[1:])

    def __str__(self):
        for addr,item in self.addressbook:
            item.depth = len(addr)
        return "Entry<%s>" % str(self.tree)

    def print_derived(self):
        print "======\nDerived Operations\n-----"
        for deriv in self.derived:
            print deriv
        print "======"

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(repr(self.tree))

"""
0 the 1 dog 2 is 3 in 4 a 5 fight 6
the -> 0, 1
dog -> 1, 2
is -> 2, 3
in -> 3, 4
a -> 4, 5
fight -> 5, 6
"""


def tests():
    tree_strs = ["(S (NP) (VP (V loves) (NP)))",
                 "(NP Chris)",
                 "(NP Sandy)",
                 "(VP* (ADVP madly))",
                 "(*VP (ADVP madly))"]

    for tree_str in tree_strs:
        print "--------------------------"
        grammar_entry = Entry.make(tree_str)
        for address, tree in grammar_entry.addressbook:
            print "-\n---"
            print "Address %s for tree_str: %s" % (address, tree_str)
            #tree.debug_string()
            print "---\n-"
        print "--------------------------\n"

    print "no early errors"

if __name__ == "__main__":
    tests()


