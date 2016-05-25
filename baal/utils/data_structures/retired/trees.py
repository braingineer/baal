"""  Tree Objects """
import types
import re
import sys
import logging
from collections import deque
from baal.nlp.lexicon import get_entries
from copy import copy
from baal.utils.general import cformat, nonstr_join
from baal.utils.hlf import gensym, unboundgensym
from baal.utils.data_structures.nodes import *
# logging tips:
#   http://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout


class enforcetype:
    def __init__(self, *dargs):
        self.enforce_args = dargs

    def __call__(self, func):
        def inner(*args, **kwargs):
            if len(args[1:]) != len(self.enforce_args):
                raise TypeError("Passed wrong number of parameters " +
                                "to either the decorator or the function")
            for arg, darg in zip(args[1:], self.enforce_args):
                if not isinstance(arg, darg):
                    raise TypeError("'%s' instance was type '%s'.  "
                                    "Class '%s' wants type '%s'" %
                                    (arg, type(arg).__name__,
                                     type(args[0]).__name__, darg.__name__))
            return func(*args, **kwargs)
        return inner


class Frontier(object):
    def __init__(self, iterable, *args, **kwargs):
        super(Frontier, self).__init__(*args, **kwargs)
        self.items = deque(iterable)
        self.span = [0, 0]
        self.logger = logging.getLogger('trees')
        self.initialize_span()

    def initialize_span(self):
        """
        Initialize the span.
        span[0] should point at the most left lexical item
        span[1] should point at the most right lexical item
        Will break down if there is a gap between lexical items

        Procedure:
            The span marks the boundary edges of the lexical material
            First, we iterate over the non lexical items to find the left boundary
            both span numbers will equal the index of the first lexical item
            if this is the case (span[0] + 1 == len(self.items)), exit
            Then, we want the right span number to point at the rightmost unit
        """
        span = [0, 0]
        for s_i, item in enumerate(self.items):
            # find the left boundary
            if not isinstance(item, LexicalNode):
                span[0] += 1
                span[1] += 1
            else:
                break
        if span[0] + 1 == len(self.items):
            # single lexical item case
            self.span = span
            return
        for s_i, item in enumerate(list(self.items)[span[0]+1:]):
            # find the right boundary
            if not isinstance(item, LexicalNode):
                break
            span[1] += 1
        self.span = span

    def update_span(self):
        """ Move the span to incorporate new lexical material """
        while self.span[0] > 0 and  \
                isinstance(self.items[self.span[0]-1], LexicalNode):
            self.span[0] -= 1
        while self.span[1] < len(self.items)-1 and \
                isinstance(self.items[self.span[1]+1], LexicalNode):
            self.span[1] += 1

    @property
    def complete(self):
        return self.span[0] == 0 and self.span[1] == len(self.items) - 1

    @property
    def nextitems(self):
        """
        return the items just beyond the lexical material
        will return none for the span boundary that has no material beyond
        """
        ret = []
        if self.span[0] > 0:
            ret.append(self.items[self.span[0]-1])
        else:
            ret.append(None)
        if self.span[1] < len(self.items)-1:
            ret.append(self.items[self.span[1]+1])
        else:
            ret.append(None)

        return ret

    @property
    def currentitems(self):
        """ Return the items at the span boundaries (should be lexical) """
        return self.items[self.span[0]], self.items[self.span[1]]

    def update(self, f_i, other_tree):
        """ Updates the span for substitution operations """
        if f_i:  # the right case.
            ind = self.span[1]+1
        else:
            ind = self.span[0]-1
        address = self.items[ind].get_address()
        self._insert(ind, other_tree.frontier.items,False)
        self.update_span()
        return tuple(address)

    def spineiter(self, node):
        """ used for insertions. move up the spine to see if it can insert """
        while node.parent is not None:
            yield node.parent
            node = node.parent

    def insert(self, subtree, direction):
        insert_func = {"left":self.insertleft, "right":self.insertright}[direction]
        insert_func(subtree)

    def insertleft(self, subtree):
        """ Leftward insertion: add items to left side """
        self._insert(self.span[0], subtree.frontier.items, True)
        self.update_span()

    def insertright(self, subtree):
        """ rightward insertion: add items to right side """
        self._insert(self.span[1], subtree.frontier.items, True)
        self.update_span()

    def _insert(self, ind, iterable,include=True):
        """
        Insert the new material in. Used as helper functions, don't call
                directly
        Note: we don't the things that were at that spot if it's a
            substitution.
        """
        if not include:
            end = ind+1
        else:
            end = ind
        item_list = list(self.items)
        self.items = deque(item_list[:ind]+list(iterable)+item_list[end:])

    def copy(self,node_dict):
        """ Making a __copy__ is too hard because I need a node dictionary """
        self.logger.debug("\n Frontier's node dict")
        for v in node_dict.values():
            self.logger.debug(repr(v))
        try:
            new_frontier = [node_dict[hash(n)] for n in self.items]
        except KeyError as e:
            self.logger.debug(cformat("Culprit: %s" % str(n), "b2"))
            self.logger.debug("\n")
            self.logger.debug(cformat("nodedict", "f"))
            for v in node_dict.values():
                if v.symbol not in [x.symbol for x in self.items]:
                    continue
                self.logger.debug(cformat(str(v),"1"))
                self.logger.debug("Symbol: %s" % v.symbol)
                self.logger.debug("Parent symbol: %s " % v.parent.symbol)
                self.logger.debug("Children: %s" % v.children)
                self.logger.debug("Type: %s" % v.node_type)
                self.logger.debug("Parent Type: %s" % v.parent.node_type)
                self.logger.debug("Hash: %s" % hash(v))
            self.logger.debug("\n")
            self.logger.debug(cformat("items","f"))
            for v in self.items:
                self.logger.debug(cformat(str(v),"1"))
                self.logger.debug("Symbol: %s" % v.symbol)
                self.logger.debug("Parent symbol: %s " % v.parent.symbol)
                self.logger.debug("Children: %s" % v.children)
                self.logger.debug("Type: %s" % v.node_type)
                self.logger.debug("Parent Type: %s" % v.parent.node_type)
                self.logger.debug("Hash: %s" % hash(v))
            raise e
        new_frontier = Frontier(new_frontier)
        assert new_frontier.span == self.span, (
            "Things have broken!",
            "Possible thing: Frontiers!",
            "Old frontier: %s" % self.items,
            "   New frontier: %s" % new_frontier.items,
            "Other things",
            "Old span: %s" % self.span,
            "New span: %s" % new_frontier.span
            )
        return new_frontier

    def __str__(self):
        return "Frontier: %s" % ",".join([str(x) for x in self.items])

    def __repr__(self):
        return "Frontier: %s" % [repr(x) for x in self.items]

    def __hash__(self):
        """
        if frontier is list of nodes ala defined below
            then hash the symbols to avoid duplicates in charts
        else
            hash the list itself
        """
        if len(self.items) > 0 and isinstance(self.items[0], Node):
            return tuple([n.symbol for n in self.items])
        return tuple(self.items).__hash__()

    @enforcetype(int)
    def __getitem__(self, k):
        """ going to assume k is an integer """
        return self.items[k]

    @enforcetype(int, Node)
    def __setitem__(self, k, v):
        self.items[k] = v

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, k):
        if len(self.items) > 0 and isinstance(self.items[0], Node):
            return tuple([n.symbol for n in self.items])
        return k in set([])

    def __len__(self):
        return len(self.items)

    def __getslice__():
        raise NotImplementedError("Slices not supported in Frontiers")

    def __setslice__():
        raise NotImplementedError("Slices not supported in Frontiers")

    def __delslice__():
        raise NotImplementedError("Slices not supported in Frontiers")

    def __missing__(self):
        raise KeyError("There no such thing here")


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
        token = match.group()
        # Case: tree/subtree starting. prepare structure
        if tree_starting(token):
            stack.append((token[1:].lstrip(), []))
        # Case: tree/subtree is ending. make sure it's buttoned up
        elif tree_ending(token):
            label, children = stack.pop()
            stack[-1][1].append(Node(symbol=label,
                                     children=children,
                                     parent=stack[-1][0]))
        # Case: leaf node.
        else:
            stack[-1][1].append(token)

    assert len(stack) == 1
    assert len(stack[0][1]) == 1
    assert stack[0][0] is None

    resulting_tree = stack[0][1][0]
    if isinstance(resulting_tree, types.ListType):
        resulting_tree = resulting_tree[0]

    assert isinstance(resulting_tree, Node)

    return clean_tree(resulting_tree)


def clean_tree(root_node):
    """
    We need to:
        1. mark nodes as substitution or insertion.
        2. find the head node.
        3. convert string labels to symbol objects?
    """
    is_insertion = False
    insert_direction = ""
    logger = logging.getLogger('trees')
    head = ""
    for c_i, child in root_node:
        if isinstance(child, Node):
            if len(child) > 0:
                root_node[c_i] = child = clean_tree(child)
                root_node.head = child.head
            elif child.symbol[-1]=="*" and child.symbol[:-1] == root_node.symbol:
                is_insertion = True
                child.symbol = child.symbol[:-1]
                assert c_i == 0 or c_i == len(root_node)-1, (
                                 "root node at error: %s. " % root_node.symbol,
                                 "Sometimes happens from badly formed bracket"
                                    )
                insert_direction = {0: "right",
                                    (len(root_node)-1): "left"}[c_i]
                root_node[c_i] = child = FootNode.from_node(child)
            else:
                root_node[c_i] = child = SubstitutionNode.from_node(child)

        else:
            # Found the head
            root_node[c_i] = head = child = LexicalNode(symbol=child,
                                                        parent=root_node)

            root_node.head = head
        child.parent = root_node
    try:
        assert len(root_node.head.symbol) > 0, type(root_node)
    except AttributeError as e:
        logger.debug(root_node)
        raise e
    if is_insertion:
        root_node = InsertionNode.from_node(root_node,
                                            direction=insert_direction)

    return root_node

# ////////////////////////////////////////////
#  Tree and its descendants
# ////////////////////////////////////////////


class Tree(object):
    """
  A generic tree object for tree grammars
    based on http://www.nltk.org/_modules/nltk/tree.html
    and
    https://github.com/tomerfiliba/tau/blob/master/tag/tig.py#L270
    Modified for Grounded Tree Grammars

  Properties:
    1. head --- Terminal node, represents core lexical item
    2. structure ---

    An initial tree is a spine plus offshoots at particular spots for
    substitution

    an auxillary tree is a spine plus an off shoot for the foot.

    so how do we want to represent this?

    we pick an initial tree. we then pick a second tree which substitutes.
    the top level structure should have a pointer to both trees
    but there should be a representation that says tree 2 substitutes
    into tree 1.

    I think a tree should have a finger on a root node.
    It should also have a finger on its frontier nodes.
    a node is an object

  Terminology from an amalgamation of places:
    1. Elementary Trees,T_E, consist of
        initial, T_I, and auxillary, T_A, trees
        ala original TAG
    2. V_T is the set of Terminal Categories
      ala (stone,2002)
    3. V_NT is the set of Non-Terminal Categories
      ala (stone,2002)
    4. V_T, in natural language, are words or production units
    5. All V in T_E are in V_T.union(V_NT)
    6. In lexicalized trees, exactly 1 frontier node is in V_T
    7. In lexicalized trees, the V_T node is the head
      anchor in (stone,2002), head else
    8. The SPINE is the path from the root to the head
      ala (stone,2002).
      Note: Schabes calls this a trunk, and reserves spine for the path
      from the root to the foot
      In general, I will be trying to keep schabes' spine to ply-1, so
      spine is used for root-head.
    9. V_NT on the frontier in T_i are substitution nodes and usually
      required (although, in theory, fragments can make sense
                especially if the referent is pragmatically salient)
    10. Substitution is one of the two operations allowed on trees
    11. The other goes by a variety of names:
      modification, adjunction, left/right-adjunction, insertion,
      forward/backward sister adjunction
    12. I will use the term "insertion"
    13. I will restrict insertions as in (Shindo et al, 2011) & (stone,2002)
        + Insertions are left-right only.
        + No simultaneous left-right insertions.
        + Auxillary trees will be kept 1-ply from root to foot
    14. I will assume that, on input, any non-spine, non-lexicalized nodes
        are marked for substitution
    """

    def __init__(self, root_node, parent, head, frontier,derived={}):
        """
        A tree will track:
            its root node
            all subtrees that have combined with it
            if its a subtree, where it attaches
            its frontier and the left-right indices of what it lexicalizes
                in the input sentence
        """
        self.logger = logging.getLogger("trees")
        self.debug = False

        self.root = root_node
        self.head = head
        self.frontier = frontier
        self.parent = parent
        self.derived = derived
        self.hlf_form = None
        self.terms = None
        self.yielded = None

        self.logger.debug("\n------------------------\n")
        self.logger.debug("Node with root %s and head %s has been created" % (self.root.symbol, self.head))

        self.logger.debug("\t%s" % self.frontier)
        self.logger.debug("\t Span(%s,%s)" % tuple(self.frontier.span))
        self.logger.debug("\t Parent: %s" % self.parent)
        self.logger.debug("\t Self type: %s" % self.root.node_type)
        self.logger.debug("\n------------------------\n")

    @property
    def next_frontier(self):
        return self.frontier.nextitems

    @property
    def current_frontier(self):
        return self.frontier.currentitems

    @classmethod
    def instantiate(cls, root=None, parent=None, i=-1,
                    bracketed_string=None, lexical_item=None):
        """ Instantiation check order: bracketed_string, lexical_item, root.
            so leave bracketed_string and lexical_item empty if root.
            lexical_item will only work if you have a lexicon
            and even then, it expects the lexicon to yield bracketed strings


            TODO Note:
                  I imagine that one day I will have generic forms for
                  parts of speech. For instance, a PP has attachment options
                  and a PP can modify different properties of the thing it
                  attaches to. So, in the case that I don't know what is
                  going on, or when I don't have the right tree for a word
                  I instantiate all possible forms, and see if I can use that
                  to infer what is going on.

                  But, this also requires a bit of ground truth.
                  So, for example, for INCA, I will have built in a
                  "I don't get it. program me" with a more rigid syntax
                  to select building blocks and connections.
                  Thus, INCA can incrementally learn which things refer
                  to other things.

                  What would be the shortcut to this?
                  We combinatorically sample from our function set
                  we ask people on mechanical turk to generate descriptions
                  we feed that into our parsing model and attempt to
                  build parse trees. we verify on another set of people.
            """
        if not root and not lexical_item and not bracketed_string:
            raise TypeError("tree.instantiate takes either a root node"
                            + " or a lexical item to make into a root node"
                            + " or a bracketed string structure to convert")

        if bracketed_string:
            root = from_string(bracketed_string)
        elif lexical_item:
            root = from_string(get_entries(lexical_item))
        head,frontier = root.initialize_frontier()
        return cls(root,parent,head,Frontier(frontier))

    @property
    def complete(self):
        return self.frontier.complete

    def __copy__(self):
        """
            Extend the copy operation to include deep copies of nodes

            Note: we only want to deep copy nodes because of child structure
                  we want to maintain as minimal repeated objects as possible
                  so, in derived, since we aren't extending it, we only copy
                  and the copy will return back references to the objects found within
        """
        newroot, node_dict = type(self.root).clone(self.root)
        return Tree(newroot, self.parent, newroot.head, self.frontier.copy(node_dict),
                    copy(self.derived))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        addressbook = sorted(self.root.get_addressbook().items())

        # Debugging
        for address,node in addressbook:
            self.logger.debug(cformat(str(address),'wu'))
            self.logger.debug(cformat("%s :: %s" % (node.symbol, node.ops), "b"))

        if not self.yielded:
            self._format_derived(addressbook)

        # Debugging
        for address,node in addressbook:
            self.logger.debug(cformat(str(address),'wu'))
            self.logger.debug(cformat("%s :: %s" % (node.symbol, node.ops), "b"))

        border = cformat("\n{:_^30}\n".format(""),"fu")
        d_fun = lambda (l,x): ("%s" % (l,))
        if len(self.derived) > 0:
            derived_str = "\n\t".join(map(d_fun, self.derived.items()))
            st_str = cformat("\nSubtree root labels:","wu") + \
                "\n\t%s" % derived_str
        else:
            st_str = ""
        return border+cformat("\nDerivation Tree:","wu")+"%s" % self.root + \
            "%s" % st_str + \
            cformat("\nYield:", "wu")+" %s" % self.yielded + \
            cformat("\nHLF Form:", "wu")+" %s" % self.hlf_form + border

    def _format_derived(self, addressbook=None):
        """ Purpose: use the addressbook to make the derived tree. """

        if addressbook is None:
            addressbook = sorted(self.root.get_addressbook().items())
        form_addr = lambda addr: ",".join(map(str,addr[1:])) if addr[0] == -1 \
                                 else ",".join(map(str,addr))

        headbook = {}
        for address,node in addressbook:

            address_str = form_addr(address)

            # node.address = list(address)
            if node.head is not None:
                headbook[node] = gensym(node.head.symbol,form_addr(node.head.get_address()))
            elif isinstance(node, LexicalNode):
                headbook[node] = gensym(node.symbol, form_addr(node.get_address()))
            else:
                headbook[node] = unboundgensym(address_str)
        parent_node_address,parent = addressbook[0]
        head = headbook[parent]
        #addressbook = addressbook[1:]
        last_address, last_node = addressbook[0]
        terms = {head:[head]}

        # a major assumption: when we enter children, it's the first time we
        #       see the parent node. this dictates how we treat heads
        #       i.e. a head's address is when we see it for the first time
        # Enter the set of children of a node
        enter_child_cond = lambda addr, last_addr: len(addr) > len(last_addr)
        # exit the set of children of a node
        exit_child_cond = lambda addr, last_addr: len(addr) < len(last_addr)

        stack = deque()
        yielder = []
        derived = {}
        for address,node in addressbook:
            address_str = ",".join(map(str,address[1:]))
            if isinstance(node,LexicalNode):
                yielder.append(node.symbol)

            if enter_child_cond(address, last_address):
                self.logger.debug("pushing %s" % parent)
                stack.append(parent)
                parent, head = last_node, headbook[last_node]
                terms.setdefault(head,[head])

            elif exit_child_cond(address, last_address):
                try:
                    parent = stack.pop()
                    head = headbook[parent]
                    self.logger.debug("popping %s" % parent)
                except IndexError as e:
                    self.logger.debug(address, node)
                    self.logger.debug(len(address), len(last_address))
                    self.logger.debug(last_node)
                    for address in addressbook:
                        self.logger.debug(address)
                    raise e
            else:
                try:
                    assert len(stack) == len(address) - 2, \
                       "Stack: %s, Address: %s" % (stack, address)
                except:
                    self.logger.debug(cformat('THE STACK IS NOT CORRECT... =(', 'f'))
                    self.logger.debug(cformat('Stack: %s, address: %s' % (stack, address),'b'))
                    headbook_str = "\n".join([str(x) for x in headbook.items()])
                    self.logger.debug(cformat('Also, check the headbook: %s' % headbook_str, 'b'))
            if len(node.ops) == 0 and node.node_type == "substitution":
                assert not node.complete
                terms[head] = terms.setdefault(head,[head]) + \
                    [unboundgensym(address_str)]

            # Hobbsian Logical Form calculations
            for opname,ophead in node.ops:
                if opname == "substitution":
                    # we assume a subbed symbol is an argument for the head
                    #    word but also that it has a function itself
                    key = "sub(%s)@%s" % (ophead.symbol, address_str)
                    # reminder about update sym:
                    #    It both adds and returns the symbol
                    #    Call help(dict().setdefault) in terminal
                    ophead = headbook[ophead]
                    sub_syms = terms.setdefault(ophead,[ophead])
                    sub_sym = sub_syms[0]
                    terms[head].append(sub_sym)

                elif opname.split("-")[0] == "insertion":
                    # Insertions are functions on the head word
                    opname,direction = opname.split("-")
                    headsym = "%s->%s" % (ophead.symbol,node.symbol) \
                        if direction == "left" \
                        else "%s<-%s" % (node.symbol,ophead.symbol)
                    key = "ins(%s)@%s" % (headsym,address_str)
                    ophead = headbook[ophead]
                    terms[ophead] = terms.setdefault(ophead,[ophead]) + \
                        [headbook[node]]
                if key in derived.keys():
                    key = key+"x"
                    self.logger.warning("Duplicate derived keys")
                derived[key] = node

            self.logger.debug('end deriv. loop. node: %s, address: %s' %
                              (node,address))
            last_node = node
            last_address = address


        self.logger.debug("Total: %s" % len(addressbook))


        hlf_make = lambda func,variables: "%s(%s)" % \
                                          (func.headword, nonstr_join(variables, ','))
        hlf_expr = " & ".join([hlf_make(f,vs) for f,vs in terms.items()])

        self.derived = derived
        self.yielded = " ".join(yielder)
        self.hlf_form = hlf_expr
        self.terms = terms

    def get_head(self):
        return self.head

    def _compatible_subst(self, node, other_tree):
        """
            Test for compatibility
            Other things to add:
                feature unification
                semantic unification
                contextual unification (just semantics?)

            Going to pass in the contextual span.
        """
        if not node:
            return False
        same_type = node.symbol == other_tree.root.symbol
        is_subst_site = isinstance(node, SubstitutionNode)
        return same_type and is_subst_site

    def _compatible_insert(self, node, other_tree):
        """
        Same things as in subst but with different semantics probably
        """
        if not node:
            return False
        same_type = node.symbol == other_tree.root.symbol
        return same_type

    def update_derived(self):
        new_derived = {}
        for key,value in self.derived.items():
            new_derived[tuple(value.root.get_address())] = value
        self.derived = new_derived

    def get_node_by_address(self, address):
        if address[0] == -1:
            address = address[1:]
        return self._get_node_by_address(self.root, address)

    def _get_node_by_address(self, node, address):
        if len(address) is 0:
            return node
        elif len(address) is 1:
            return node.children[address[0]]
        else:
            try:
                return self._get_node_by_address(node.children[address[0]],address[1:])
            except IndexError as e:
                print node.children
                print address
                raise e

    def combine(self, other_tree, edge_conditionals=(True, True)):
        """ We assume the other tree is the inserter or substituter """
        self.logger.debug("Inside the combine.")
        self.logger.debug("My symbol: %s" % self.root.symbol)
        self.logger.debug("My frontier with span %s: %s" % (self.frontier.span,
                                                        repr(self.frontier)))
        self.logger.debug("Other symbol: %s" % other_tree.root.symbol)
        if isinstance(other_tree.root, InsertionNode):
            self.logger.debug("Found an insertion node")
            left, right = zip(self.current_frontier, edge_conditionals)
            direction = other_tree.root.direction

            spine_leaf,edge_bool = right if direction == "right" else left

            for frontier_spine in self.frontier.spineiter(spine_leaf):

                if self._compatible_insert(frontier_spine, other_tree) and \
                                                                     edge_bool:
                    newtree = copy(self)
                    new_other_tree = copy(other_tree)
                    insert_address = frontier_spine.get_address()
                    insertee_node = newtree.get_node_by_address(insert_address)
                    tree_index = insertee_node.insert_into(direction,
                                                           new_other_tree.root)
                    assert tuple(insert_address) == tree_index
                    newtree.frontier.insert(new_other_tree, direction)
                    newtree._format_derived()
                    yield newtree

        elif isinstance(other_tree.root, Node):
            left, right = self.next_frontier
            self.logger.debug("In the substitution condition")
            # a note about limiting subs
            # This basically allows us to restrict which substitution site will
            #       get looked at. This is useful when the lex material is
            #       surrounded by the same substitution site symbol (e.g. two NP)
            #       the incoming other tree is only compatible with one of them
            #       as determined by the edge indices. But we don't want to
            #       complicate the logic here with edge indices, so we just pass
            #       in the verdict on whether an edge substitution can happen or not
            limiting_subs = zip([left,right],edge_conditionals)
            for f_i,(frontier_item,edge_bool) in enumerate(limiting_subs):
                if self._compatible_subst(frontier_item, other_tree) \
                                                      and edge_bool:
                    self.logger.debug("self: %s substituting %s in" %
                                   (repr(self.root),repr(other_tree.root)))
                    self.logger.debug("Complete other tree: %s" % other_tree)
                    newtree = copy(self)
                    new_other_tree = copy(other_tree)
                    self.logger.debug("Substituting into %s" % newtree.next_frontier[f_i])
                    newtree.next_frontier[f_i].substitute_into(new_other_tree.root)
                    new_other_tree.root = newtree.next_frontier[f_i]
                    tree_index = newtree.frontier.update(f_i,new_other_tree)
                    new_other_tree.parent = newtree
                    newtree._format_derived()
                    yield newtree


def tests(debug=False):
    """
    testing procedure:
    make trees.
    combine them.
    print output.
    """
    frombstr = lambda x: Tree.instantiate(bracketed_string=x)
    debug_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=debug_level, stream=sys.stdout)
    base_string = "(S (NP) (VP (V loves) (NP)))"
    base_tree = frombstr(base_string)

    comp_strings = ["(NP Chris)",
                    "(NP Sandy)"]
    comptrees = [frombstr(x) for x in comp_strings]
    insert_strings = ["(VP (ADVP madly) (VP*))",
                      "(VP (VP*) (ADVP madly))"]
    inserttrees = [frombstr(x) for x in insert_strings]

    insertion_tests(*([base_tree]+inserttrees))

    composition_test(*([base_tree]+comptrees))

    print "\nMaking 'Chris loves Sandy madly'\n"
    for newt1 in base_tree.combine(comptrees[1]):
        for newt2 in newt1.combine(comptrees[0]):
            print newt2
            for newt3 in newt2.combine(inserttrees[1]):
                print "Result: %s" % newt3


def insertion_tests(base_tree, inserttree1,inserttree2):
    print "Base tree: %s \n" % base_tree
    print "Insertion"
    for newt in base_tree.combine(inserttree1):
        for newt2 in newt.combine(inserttree2):
            print "Result of first combine: %s" % newt
            print "Result of second combine: %s" % newt2


def composition_test(base_tree, comptree1, comptree2):
    print "Base tree: %s \n" % base_tree
    print "Composition"
    for newt in base_tree.combine(comptree1):
        for newt2 in newt.combine(comptree2):
            print "Result of first combine: %s" % newt
            print "Result of second combine: %s" % newt2
    # print "Post composition, base tree unchanged %s" % base_tree
#    print "\n---\n"
#    print "Composition 2"
#    for newt in test_trees[1].combine(test_trees[2]):
#        print newt
#    print "Post Comp"
#    print test_trees[1]


if __name__ == "__main__":
    print "Running tests"
    tests()
