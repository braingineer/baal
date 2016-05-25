from baal.utils.general import cprint, bcolors, cformat, nonstr_join
from baal.utils import config
from copy import copy, deepcopy
import re, types, logging
try:
    from nltk.tree import Tree as TKTree
except ImportError:
    print("Warning: You don't have NLTK. One method won't work in ConstituencyTree")

logger = logging.getLogger("treedebugging")

import baal
#baal.utils.loggers.turn_on("trees", "debug")

class ConstituencyTree(object):
    def __init__(self, symbol, children=None, parent="", semantic=None):
        self.symbol = symbol
        self.children = children or []
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
    def make(cls, bracketed_string=None,  tree=None, correct_root=False):
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

        #new_tree.correct_projections()
        new_tree, addressbook = new_tree.clone()
        return new_tree, addressbook

    def correct_projections(self):
        while (len(self.children) == 1 and not self.children[0].lexical and
               self.children[0].symbol == self.symbol):
            # if things suck and keep sucking.
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


    def clone(self, address=None, addressbook=None,prefab_children=None,
                    child_offset=0):
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
        new_tree = ConstituencyTree(self.symbol, children=new_children, parent=self.parent)
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
            #new_tree.is_argument = True
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
            self.children[ind] = ConstituencyTree(subst.symbol)
            self.children[ind].complement = True
            self.children[ind].parent = self.symbol

        self.substitutions = []
        return new_subtrees

    def excise_adjuncts(self):
        new_subtrees = []
        for adj_tree in self.adjuncts:
            ind = self.children.index(adj_tree)
            self.children[ind] = None
            adj_wrapper = ConstituencyTree(symbol=self.symbol, children=[adj_tree])
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

    def verbose_string(self,depth=1, verboseness=float('inf'), spacing=1):
        # return self._simple_str()
        p = lambda x: bcolors.OKBLUE+x+bcolors.ENDC
        b = lambda x: bcolors.BOLD+x+bcolors.ENDC
        g = lambda x: bcolors.OKGREEN+x+bcolors.ENDC
        r = lambda x: bcolors.FAIL+x+bcolors.ENDC

        if self.complement:
            this_symbol = "%s(%s)" % (p("Subst"), g(self.symbol))
        elif self.adjunct:
            direction = "[->]" if self.direction == "right" else "[<-]"
            this_symbol = "%s(%s)" % (p(direction+"Ins"), g(self.symbol))
        elif self.lexical:
            this_symbol = "%s(%s)" % (p("lex"), g(self.symbol))
        else:
            this_symbol = self.symbol

        d = depth
        s = lambda x: ("{:%d}" % (x*d)).format("")
        if verboseness > 0:
            if len(self.children) > 0:
                tree_print =  "\n"+s(spacing)+b("%s{"%d)
                tree_print += "{:-<1}->\n".format(this_symbol)
                tree_print += s(spacing) + r("[")
                tree_print += " "+("\n "+s(spacing)).join([x.verbose_string(depth+1, verboseness-1) for x in self.children])
                tree_print += r("]") + "\n"
                tree_print += s(spacing)+b("}")
                return tree_print
            else:
                return this_symbol
        else:

            this_symbol = self.symbol if self.lexical else ""
            mysep,childsep, spacers = " ", "; ", " "
            if len(self.children) > 0:
                tree_print = this_symbol
                tree_print += ("|").join([x.verbose_string(depth+1, verboseness-1) for x in self.children])
                return tree_print
            else:
                return this_symbol


    def __repr__(self):
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
        #if "-" in token and token[0] != '-' and in_str[match.start()-1] != " ": 
        #    token, semantic = token.split("-")[0], token.split("-")[1]

        # Case: tree/subtree starting. prepare structure
        if tree_starting(token):
            stack.append((token[1:].lstrip(), [], semantic))
        # Case: tree/subtree is ending. make sure it's buttoned up
        elif tree_ending(token):
            label, children, semantic = stack.pop()
            stack[-1][1].append(ConstituencyTree(symbol=label,
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

    assert isinstance(resulting_tree, ConstituencyTree)

    return clean_tree(resulting_tree, [0], {tuple([0]):resulting_tree})


def clean_tree(root_tree, address, addressbook):
    """
        Clean the tree. This is called by from_string
        From_String doesn't annotate the tree, it just makes makes the structure
        this is to annotate with the relevant information
    """
    logger = logging.getLogger('trees')
    logger.debug(root_tree)
    tag_exceptions = set("-LRB- -RRB- -RSB- -RSB- -LCB- -RCB-".split(" "))
    if root_tree.symbol == "" and len(root_tree.children)==1:
        root_tree.symbol = "ROOT"
    if "*" in root_tree.symbol:
        root_tree.adjunct = True
        root_tree.direction = "right" if root_tree.symbol[0] == "*" else "left"
        root_tree.symbol = root_tree.symbol.replace("*", "")
    if "-" in root_tree.symbol and root_tree.symbol not in tag_exceptions:
        root_tree.symbol = root_tree.symbol.split("-")[0]
    if "=" in root_tree.symbol:
        root_tree.symbol = root_tree.symbol.split("=")[0]
    if "|" in root_tree.symbol:
        root_tree.symbol = root_tree.symbol.split("|")[0]
        
    if root_tree.symbol == "":
        print(root_tree, addressbook)
        return None, addressbook

    logger.debug('starting child iter for %s' % root_tree.symbol)
    bad_offset = 0
    marked_bad = []
    #### NOTES ON BAD OFFSET
    # to fix when bad children happen (aka, -None-)
    # I want to remove their tree. but this is tricky because its coupled
    # with the spine_index 
    # so, i keep track of the bad offset, set the spine index to the correct val using it
    # and then at the end, I update the list to be correct 
    for c_i, child in enumerate(root_tree.children):
        next_address = address+[c_i - bad_offset]
        if isinstance(child, ConstituencyTree):
            logger.debug("child is a constituency tree")
            logger.debug(child)
            if '-NONE-' in child.symbol:
                marked_bad.append(c_i)
                bad_offset += 1
                continue 

            if len(child.children) > 0:
                ### SPECIAL CASE:
                ### basically, this can take out the None chidlren that happen in WSJ
                if ( len(child.children) == 1 and 
                     isinstance(child.children[0], ConstituencyTree) and 
                     "-NONE-" in child.children[0].symbol):
                    marked_bad.append(c_i)
                    bad_offset += 1
                    continue
                # Interior node
                logger.debug('diving into child')
                logger.debug('specifically: %s' % child)
                child, addressbook = clean_tree(child, next_address, addressbook)
                if child is None:
                    marked_bad.append(c_i)
                    bad_offset += 1
                    continue
                
                root_tree.children[c_i] = child

                if child.head is not None:
                    root_tree.head = child.head
                    root_tree.spine_index = c_i - bad_offset
                #else:
                #    raise AlgorithmicException, "Only making initial trees here"

            else:
                # Substitution point
                logger.debug("child was a complement")
                child.complement = True

        else:
            if "*" in child:
                marked_bad.append(c_i)
                bad_offset += 1
                continue
            # Found the head
            child = child.lower()
            child = ConstituencyTree(symbol=child, parent=root_tree.symbol)
            child.lexical = True
            root_tree.children[c_i] = child
            head = child
            head.lexical = True
            root_tree.head = head.symbol
            root_tree.spine_index = c_i - bad_offset
        try:
            addressbook[tuple(next_address)] = child
        except TypeError as e:
            import pdb
            pdb.set_trace()

        child.parent = root_tree.symbol

    root_tree.children = [child for c_i, child in enumerate(root_tree.children) if c_i not in marked_bad]
    if len(root_tree.children) == 0:
        return None, addressbook

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

