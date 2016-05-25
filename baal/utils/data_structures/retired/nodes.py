from baal.utils.general import cprint, bcolors, cformat, nonstr_join
from collections import deque
from copy import copy
import logging


def unify(feat_a, feat_b):
    return {}


class Node(object):
    __slots__ = ['symbol', 'children', 'parent', 'features',
                 'head', 'node_type', 'node_type_short','address', 'ops',
                 'logger']

    def __init__(self, symbol=None, children=[],
                 parent=None, features={}, head=None,
                 address=-1, ops=[],*args, **kwargs):
        self.symbol = symbol
        self.children = children
        self.parent = parent
        self.features = features
        self.head = head
        self.node_type = self.node_type_short = "interior"
        self.address = address
        self.ops = copy(ops)
        self.logger = logging.getLogger('trees')

    @classmethod
    def anonymous_copy(cls, other, *args, **kwargs):
        if isinstance(other, InsertionNode):
            kwargs['direction'] = other.direction
        return cls(symbol=other.symbol, features=other.features,
                   ops=other.ops, children=[], parent=None, *args, **kwargs)

    @classmethod
    def from_node(cls, other, *args, **kwargs):
        """ Does not create deeply copied clone """
        if isinstance(other, InsertionNode):
            kwargs['direction'] = other.direction
        return cls(symbol=other.symbol, children=other.children,
                   parent=other.parent, features=other.features,
                   head=other.head,ops=other.ops,
                   *args, **kwargs)

    @classmethod
    def clone(cls, other, *args, **kwargs):
        logger = logging.getLogger('trees')
        node_types = {"interior": Node,
              "insertion": InsertionNode,
              "substitution": SubstitutionNode,
              "lexical": LexicalNode,
              "foot": FootNode}
        ntype = lambda node: node_types[node.node_type]
        addresses = other.get_addressbook()
        sorted_addresses = sorted(addresses.items(), key=lambda x:x[0])

        copied_addressbook = []

        lexical_heads = {}
        head_tracker = []

        for address,node in sorted_addresses:
            new_node = ntype(node).anonymous_copy(node)
            head = node.head.symbol if node.head else None
            if head is not None:
                lexical_heads.setdefault(head,None)
            if new_node.symbol in lexical_heads.keys():
                lexical_heads[new_node.symbol] = new_node
            head_tracker.append(head)
            copied_addressbook.append([address,new_node])

        logger.debug("Copied addressbook")
        for item in copied_addressbook:
            logger.debug("Item: %s" % item)

        assert copied_addressbook[0][0] == (-1,), copied_addressbook
        parent_address, parent = copied_addressbook[0]
        logger.debug("First parent: %s" % parent)
        logger.debug("Head tracker: %s" % head_tracker)

        last_address = copied_addressbook[1][0]
        if head_tracker[0]:
            parent.head = lexical_heads[head_tracker[0]]

        stack = deque()

        for head,(address,node) in zip(head_tracker[1:],copied_addressbook[1:]):
            if len(address) > len(last_address):
                logger.debug("%s promoted to parent; %s chilling on ice" %
                             (last_node, parent))
                stack.append((parent,parent_address))
                parent = last_node
                parent_address = last_address
            elif len(address) < len(last_address):
                parent,parent_address = stack.pop()
                try:
                    while len(address) <= len(parent_address):
                        parent,parent_address = stack.pop()
                except IndexError as e:
                    print "Something went wrong with the balancing of parents"
                    print "This message written when it was noticed multiple"
                    print "pops needed to happen for a jump from a deep child"
                    print "back to a shallow child"
                logger.debug("%s coming off the stack" % parent)
            else:
                assert len(stack) == len(address)-2

            parent.add_child(node)
            logger.debug("Asserting that %s is a child of %s" % (node, parent))
            node.parent = parent
            if head is not None:
                node.head = lexical_heads[head]

            last_node = node
            last_address = address

        root = copied_addressbook[0][1]
        node_dict = {hash(v):v for k,v in copied_addressbook}

        return cls.from_node(root), node_dict


    @classmethod
    def clone_v1(cls, other, *args, **kwargs):
        logger = logging.getLogger('trees')
        node_types = {"interior": Node,
                      "insertion": InsertionNode,
                      "substitution": SubstitutionNode,
                      "lexical": LexicalNode,
                      "foot": FootNode}
        ntype = lambda node: node_types[node.node_type]
        addresses = other.get_addressbook()
        sorted_addresses = sorted(addresses.items(), key=lambda x:x[0])
        logger.debug("\naddresses in clone")
        for k,v in sorted_addresses:
            logger.debug(repr(v))
        logger.debug("end addreses in clone")

        parent_node_address,parent_node = sorted_addresses[0]
        parent_node = ntype(parent_node).from_node(parent_node)
        parent_node.children = []
        head_node = ntype(parent_node.head).from_node(parent_node.head)

        sorted_addresses = sorted_addresses[1:]

        last_address, last_node = sorted_addresses[0]

        last_node = ntype(last_node).from_node(last_node)
        last_node.children = []
        if last_node.head:
            last_node.head = ntype(last_node.head).from_node(last_node.head)


        parnode_stack = deque()
        parnode_stack.append(parent_node)

        node_dict = {}
        all_nodes = []

        for cur_address,cur_node in sorted_addresses:
            if len(cur_address) > len(last_address):
                # we are encountering children condition
                # push current parent to stack
                # make last node current parent
                # every child should be added to parent as its seen
                # make a funciton that also sets the child's address
                # every child should also get the parent node passed into it
                parnode_stack.append((parent_node,head_node))
                parent_node = last_node
                head_node = last_node.head

            elif len(cur_address) < len(last_address):
                # we have finished with children condition
                # children should have had parent pushed in
                # parent should had children being set as they were seen
                # pop parent off stack and set current parent to it
                parent_node,head_node = parnode_stack.pop()

            # normal case.  here we just do the child.parent = parent
            #                           and the parent.children.append
            new_node = ntype(cur_node).from_node(cur_node)
            new_node.parent = parent_node
            new_node.children = []
            parent_node.add_child(new_node)
            last_node = new_node
            last_address = cur_address
            if new_node == head_node:
                head_node.parent = parent_node
            all_nodes.append(new_node)

        if len(parnode_stack) > 0:
            parent_node = parnode_stack.popleft()
        logger.debug("\nbefore clone end")
        logging.debug(parnode_stack)

        for node in [parent_node]+all_nodes:
            node_dict[hash(node)] = node
            logger.debug(repr(node))


        return cls.from_node(parent_node), node_dict

    def add_child(self, node):
        self.children.append(node)
        node.address = len(self.children)-1

    def depth(self):
        up = self
        i = 0
        while up is not None and isinstance(up, Node):
            up = up.parent
            i += 1
        return i

    def insert_into(self, direction, node):
        new_children = [child for child in node.children
                        if not isinstance(child, FootNode)]

        if direction == "right":
            self.children.extend(new_children)
        if direction == "left":
            self.children = new_children+self.children

        for c_i,child in enumerate(self.children):
            child.address = c_i
            child.parent = self

        if self.parent:
            for c_i, child in enumerate(self.parent.children):
                child.address = c_i

        node.address = self.address
        node.parent = self.parent
        self.ops.append(["insertion-"+direction, node.head])

        return tuple(self.get_address())

    def __eq__(self, other):
        """ booleans verbose with variables to facilitate reading """
        if not isinstance(other, Node):
            return False

        # Check the self!
        symbols_match = self.symbol == other.symbol

        if not symbols_match:
            return False

        # Check the parent!
        if self.parent:
            if other.parent:
                parent_match = self.parent.symbol == other.parent.symbol
            else:
                parent_match = False
        elif other.parent:
            parent_match = False
        else:
            parent_match = True

        if not parent_match:
            return False

        # Check the children!
        if len(self.children) > 0:
            if len(other.children) > 0:
                children_match = all(
                                [True if x.symbol == y.symbol else False
                                 for x,y in zip(self.children,other.children)])
            else:
                children_match = False
        elif len(other.children) > 0:
            children_match = False
        else:
            children_match = True

        if not children_match:
            return False

        return True

    def get_address(self):
        address = []
        if self.parent and isinstance(self.parent,Node):
            for c_i,child in enumerate(self.parent.children):
                child.address = c_i
            if self.parent.address >= 0:
                address = self.parent.get_address()
        address.extend([self.address])
        return address

    def get_addressbook(self):
        addressbook = {}
        addressbook[tuple([self.address])] = self
        for child in self.children:
            addressbook = child._recursive_addressbook(addressbook,
                                                       [self.address])
        return addressbook

    def _recursive_addressbook(self, addressbook,parentaddress):
        # self.logger.debug("In addressbook recursion")
        # self.logger.debug("Self: %s" % repr(self))
        # self.logger.debug("Parent: %s" % repr(self.parent))
        # self.logger.debug("Children: %s" % repr(self.children))
        # self.logger.debug("Address so far: %s" % repr(parentaddress))
        # self.logger.debug("My address: %s" % repr(self.address))

        address = parentaddress+[self.address]
        addressbook[tuple(address)] = self
        for child in self.children:
            addressbook = child._recursive_addressbook(addressbook,
                                                       address)
        return addressbook

    def initialize_frontier(self):
        frontier = list()
        for c_i, child in self:
            child.address = c_i
            if isinstance(child, SubstitutionNode):
                frontier.append(child)
            # elif isinstance(child, FootNode):
            #    frontier.append(None)
            elif isinstance(child, LexicalNode):
                frontier.append(child)
            else:  # interior node case
                c_head, c_frontier = child.initialize_frontier()
                frontier.extend(c_frontier)
        return self.head, frontier

    def complete(self):
        raise NotImplementedError("This is not a substitution tree")

    def __copy__(self):
        """
            Return a deep copy of the node and its descendants
        """
        node,node_dict = Node.clone(self)
        return node

    def __hash__(self):
        if not self.parent:
            p = ""
        else:
            p = self.parent.symbol
        h1 = hash("%s%s" % (p, self.symbol))
        h2 = hash(self.address)
        h3 = sum([hash(n.symbol) for n in self.children])
        return h1 + h2 + h3

    def __nonzero__(self):
        return True

    def __setitem__(self, k, v):
        assert isinstance(k, int)
        assert isinstance(v, Node)
        self.children[k] = v

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        for c_i, child in enumerate(self.children):
            yield c_i, child

    def _simple_str(self):
        return "%s(%s)" % (self.node_type_short, self.symbol)

    def __str__(self):
        # return self._simple_str()
        p = lambda x: bcolors.OKBLUE+x+bcolors.ENDC
        b = lambda x: bcolors.BOLD+x+bcolors.ENDC
        g = lambda x: bcolors.OKGREEN+x+bcolors.ENDC
        r = lambda x: bcolors.FAIL+x+bcolors.ENDC

        if self.node_type != "interior":
            this_symbol = "%s(%s)" % (p(self.node_type_short), g(self.symbol))
        else:
            this_symbol = self.symbol
        d = self.depth()
        s = lambda x: ("{:%d}" % (x*d)).format("")

        this_symbol += "@%s" % self.get_address()

        if len(self.children) > 0:
            tree_print = "\n"+s(4)+b("%s{"%d)
            tree_print += "{:-<3}->\n".format(this_symbol)
            tree_print += s(5) + r("[")
            tree_print += (",\n"+s(6)).join([str(x) for x in self.children])
            tree_print += r("]") + "\n"
            tree_print += s(4)+b("}")
            return tree_print
        else:
            return this_symbol

    def __repr__(self):
        if self.parent:
            par_repr = "%s->" % self.parent.symbol
        else:
            par_repr = "X"
        if self.node_type != "interior":
            this_symbol = "%s(%s)" % (self.node_type_short, self.symbol)
        else:
            this_symbol = self.symbol
        this_symbol += "[@%s]" % self.get_address()
        d = self.depth()
        repr_str = par_repr+this_symbol+" with %s children" % len(self.children)
        return repr_str


class SubstitutionNode(Node):
    __slots__ = []

    def __init__(self, *args, **kwargs):
        super(SubstitutionNode, self).__init__(*args, **kwargs)
        self.node_type = "substitution"
        self.node_type_short = "sub"
        if self.complete:
            self.node_type_short = "sub'd"

    @property
    def complete(self):
        return len(self.children) > 0

    def substitute_into(self, other):
        logger = logging.getLogger('trees')
        if self.complete:
            logging.getLogger('trees').debug("me: %s, other: %s" % (self, other))
            raise ValueError(
                "You tried to substitute into a non-substitutable node")
        assert other.symbol == self.symbol
        logger.debug("My parent: %s; my Head: %s" % (repr(self.parent), repr(self.head)))
        logger.debug("my children: %s" % self.children)
        logger.debug("other's children: %s" % other.children)
        self.children = other.children

        for child in self.children:
            child.parent = self
        logger.debug("Other's parent: %s" % other.parent)
        logger.debug("My parent: %s" % self.parent)
        other.parent = self.parent
        self.head = other.head
        self.ops += other.ops
        # this is done assuming that the tree will unify on the way
        # back up the tree
        self.features = unify(self.features, other.features)
        self.node_type_short = "sub'd"
        self.ops.append(["substitution",other.head])


class InsertionNode(Node):
    __slots__ = ['direction']

    def __init__(self, direction="left", *args, **kwargs):
        super(InsertionNode, self).__init__(*args, **kwargs)
        self.node_type = "insertion"
        self.node_type_short = "insert_"+direction
        self.direction = direction


class LexicalNode(Node):
    __slots__ = []

    def __init__(self, *args, **kwargs):
        super(LexicalNode, self).__init__(*args, **kwargs)
        self.node_type = "lexical"
        self.node_type_short = "lex"

    @property
    def gorn_rank(self):
        backward_enumerate = lambda iterable: ((i,list(iterable)[i]) for i in range(len(list(iterable))-1,-1,-1))
        address = []
        if self.parent and isinstance(self.parent,Node):
            for c_i,child in enumerate(self.parent.children):
                child.address = c_i
            if self.parent.address >= 0:

                address = self.parent.get_address()

        parent_list = []
        p = self.parent
        while p is not None and isinstance(p,Node):
            parent_list.append(p)
            p = p.parent
        parent_list.reverse()

        for v_i, (val,addr) in backward_enumerate(zip(parent_list,address)):
            if val.head.symbol != self.symbol:

                return tuple(address[:v_i+2])

        return tuple(address[0:1])


class FootNode(Node):
    __slots__ = []

    def __init__(self, *args, **kwargs):
        super(FootNode, self).__init__(*args, **kwargs)
        self.node_type = self.node_type_short = "foot"
