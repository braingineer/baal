"""
Derivation and Elementary Trees live here.
"""

from __future__ import print_function
from baal.structures import Entry, ConstituencyTree, consts
from baal.semantics import Predicate, Expression
from collections import deque
from copy import copy, deepcopy
from math import floor, ceil

try:
    input = raw_input
except:
    pass


def prn_pairs(phead, thead):
    pairs = [("-LRB-", "-RRB-"), ("-RSB-", "-RSB-"), ("-LCB-", "-RCB-"), 
             ("--", "--"), (",", ",")]
    return any([left.lower()==phead.lower() and right.lower()==thead.lower() for left,right in pairs])


class AttachmentPoint(object):
    def __init__(self, free, pos_symbol, gorn, type, seq_index):
        self.free = free
        self.pos_symbol = pos_symbol
        self.gorn = gorn
        self.type = type
        self.seq_index = seq_index
        self.hlf_symbol = None
        self.frontier_increment = 0.01
        self.frontier = (-1,0)

    def __repr__(self):
        return "{}@{}".format(self.pos_symbol,self.gorn)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @classmethod
    def from_tree(cls,  tree, address, seq_index, tree_type):
        new_point = cls(True, tree.symbol, address, tree_type, seq_index)
        if tree.spine_index >= 0:
            new_point.frontier = (tree.spine_index, tree.spine_index)
        return new_point

    @property
    def left_frontier(self):
        l, r = self.frontier
        self.frontier = (l-self.frontier_increment, r)
        assert self.frontier[0] > floor(self.frontier[0]) 
        return self.frontier[0]

    @property
    def right_frontier(self):
        l, r = self.frontier
        self.frontier = (l, r+self.frontier_increment)
        assert self.frontier[1] < ceil(self.frontier[1])
        return self.frontier[1]

    def sibling_increment(self, left=True):
        l, r = self.frontier
        if left:
            self.frontier = (ceil(l) - 1.0, r)
        else:
            self.frontier = (l, floor(r) + 1.0)


    def match(self, op):
        pos_match = self.pos_symbol == op.target['pos_symbol']
        gorn_match = ((self.gorn == op.target['target_gorn'])
                       or op.target['target_gorn'] is None)
        hlf_match = self.hlf_symbol == op.target['target_hlf']
        type_match = self.type == op.type
        fail = []
        if not pos_match:
            f = "failure because pos:"
            f += "self: {}; op: {}".format(str(self.pos_symbol),
                                           str(op.target['pos_symbol']))
            fail.append(f)
        if not gorn_match:
            f = "failure because gorn:"
            f += "self: {}; op: {}".format(str(self.gorn),
                                           str(op.target['target_gorn']))
            fail.append(f)
        if not hlf_match:
            f = "failure because hlf:"
            f += "self: {}; op: {}".format(str(self.hlf_symbol),
                                           str(op.target['target_hlf']))
            fail.append(f)
        #if len(fail) > 0:
        #    print(" & \n".join(fail))
        #else:
        #    print("Success!")
        return self.free and pos_match and gorn_match and hlf_match and type_match

    def set_path_features(self, hlf_symbol):
        self.hlf_symbol = hlf_symbol

    def clone(self):
        ret = AttachmentPoint(self.free, self.pos_symbol, self.gorn,
                              self.type, self.seq_index)
        ret.hlf_symbol = self.hlf_symbol
        ret.frontier = self.frontier
        return ret

class AttachmentOperation(object):
    """Represents an elementary tree operation

    Used by DerivationTrees when trying to find where an elementary tree should attach
    There are two modes to the operation:
        1. Use it as a general attachment. In this case it needs to know
           the permissable attachments via the pos_symbol (and direction if insertion)
        2. Use it in specific attachment.  In this case it needs to know
           identifying information about the tree it should be attaching to.
           Current ideas: hlf_symbol, tree_id, argument_number, gorn_address
           Thoughts: gorn_address won't work (for obvious reasons as the tree grows)
                     tree_id won't work because there might be duplicates
                     hlf_symbol could work, as long as this semantic form remains
                     argument_number requires planning, which CSG and others might handle
     """
    def __init__(self, target, type):
        """Pass in the already made parameters to make the operation.

        Args:
            target: dict with keys 'pos_symbol' and 'parameter'
                    'pos_symbol' is the part of speech this operation looks for
                    'parameter' is direction for insertions, and argument number
                        for substitutions
            type:   the type of operation this is: consts.INSERTION or consts.SUBSTITUTION

        Notes:
            insertion direction: left means it inserts on the left side
                                 e.g. (NP* (DT a)) inserts left.
                                      the asterisk denotes the attachment point
                                 right means it inserts on the right side
                                 e.g. (*S (. .)) inserts right
                                      the asterisk denotes the attachment point
        """
        self.target = target
        self.type = type

    @property
    def is_insertion(self):
        return self.type == consts.INSERTION

    @property
    def direction(self):
        if not self.is_insertion:
            raise Exception("Not an insertion tree")
        else:
            return self.target['attach_direction']

    def clone(self):
        return AttachmentOperation(self.target, self.type)

    def set_path_features(self, target_gorn, target_hlf):
        if target_hlf is not None:
            self.target['target_hlf'] = target_hlf
        if target_gorn is not None:
            self.target['target_gorn'] = tuple(target_gorn)

    @classmethod
    def from_tree(cls, tree):
        """Calculate the parameters for the operation from a parse tree

        Args:
            tree: A ConstituencyParse instance
        """
        if tree.adjunct:
            target = {'pos_symbol': tree.symbol, 'attach_direction': tree.direction,
                      'target_gorn': None, 'target_hlf': None}
            type = consts.INSERTION
        else:
            target = {'pos_symbol': tree.symbol, 'attach_direction': "up",
                      'target_gorn': None, 'target_hlf': None}
            type = consts.SUBSTITUTION

        return cls(target, type)

class ElementaryTree(object):
    """represent a tree fragment, its operations, and its internal addresses
    """
    def __init__(self, op, head, head_address, head_symbol, bracketed_string,
                       substitution_points, insertion_points,
                       hlf_symbol=None, tree_id=None, last_type=None, last_index=-1):
        self.tree_operation = op
        self.head = head
        self.head_address = head_address
        self.substitution_points = substitution_points
        self.insertion_points = insertion_points
        self.address = (0,)
        self.last_type = last_type
        self.last_index = last_index
        self.hlf_symbol = hlf_symbol
        self.bracketed_string = bracketed_string
        self.tree_id = tree_id
        self.head_symbol = head_symbol

    @classmethod
    def from_full_parse_tree(cls, parse_tree):
        if parse_tree.symbol == "" and len(parse_tree.children) == 1:
            parse_tree.symbol = "ROOT"
        _, addressbook = parse_tree.clone()

    @classmethod
    def from_single_parse_tree(cls, parse_tree):
        if parse_tree.save_str().upper() == "(ROOT ROOT)":
            return cls.root_tree()
        

        _, addressbook = parse_tree.clone()


        head = None
        head_address = None
        substitution_points = list()
        insertion_points = list()

        sorted_book = sorted(addressbook.items())
        _, root = sorted_book[0]
        root_sym = root.symbol

        for address, tree in sorted_book:
            #if tree.symbol == "ROOT":
            #    head = "ROOT"
            #    new_point = AttachmentPoint.from_tree(tree, address, 0, consts.SUBSTITUTION)
            #    substitution_points.append(new_point)
            if tree.lexical:
                if head is None:
                    head = tree.symbol
                    head_address = address
                    head_parent = tree.parent
                else:
                    assert prn_pairs(head, tree.symbol)
            elif tree.complement:
                new_point = AttachmentPoint.from_tree(tree,
                                                      address,
                                                      len(substitution_points),
                                                      consts.SUBSTITUTION)
                substitution_points.append(new_point)
            elif tree.spine_index >= 0:
                new_point = AttachmentPoint.from_tree(tree,
                                                      address,
                                                      len(insertion_points),
                                                      consts.INSERTION)
                insertion_points.append(new_point)
            else:
                print(address, tree)
                print("Then what is it?")

        op = AttachmentOperation.from_tree(parse_tree)

        assert (head is not None and head_address is not None) or head is "ROOT"
        return cls(op, head, head_address, head_parent, parse_tree.save_str(),
                   substitution_points, insertion_points)

    @classmethod
    def from_bracketed_string(cls, bracketed_string):
        parse_tree, _ = ConstituencyTree.make(bracketed_string=bracketed_string)
        return cls.from_single_parse_tree(parse_tree)

    @classmethod
    def root_tree(cls):
        root_op = AttachmentOperation({'pos_symbol': 'ROOT', 'attach_direction': None,
                                       'target_gorn': None, 'target_hlf':None},
                                      consts.SUBSTITUTION)
        root_subpoint = AttachmentPoint(True, 'ROOT', (0,), consts.SUBSTITUTION, 0)
        root_subpoint.hlf_symbol = "g-1"
        return cls(root_op, "", None, None, "(ROOT)",
                   [root_subpoint], [], hlf_symbol="g-1")


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    ################### INSERTION OPERATION
    ########################################
    def insert(self, op_tree):
        new_tree = deepcopy(self)#.clone()
        address = new_tree.mark_insertion(op_tree.tree_operation)
        op_tree = deepcopy(op_tree)#.clone()
        op_tree.address = address
        return new_tree, op_tree

    def mark_insertion(self, op):
        assert self.last_match is not None
        assert self.last_match.match(op)
        if op.target['attach_direction'] == "left":
            op_index = self.last_match.left_frontier
        else:
            op_index = self.last_match.right_frontier
        return self.last_match.gorn + (op_index,)

    def matches_inspoint(self, op):
        self.last_type = None
        self.last_index = -1
        for index, point in enumerate(self.insertion_points):
            if point.match(op):
                self.last_index = index
                self.last_type = consts.INSERTION
                return True
        return False

    ################### SUBSTITUTION OPERATION
    ###########################################
    def substitute(self, op_tree):
        """update open substitution spots.

        Args:
            op_tree: an ElementaryTree instance
        Notes:
            accepts an op_tree that needs to substitute here.
            raises an Exception if it can't
        """
        new_tree = deepcopy(self)#self.clone()
        address = new_tree.mark_substituted(op_tree.tree_operation)
        op_tree = deepcopy(op_tree)#.clone()
        op_tree.address = address
        return new_tree, op_tree

    def mark_substituted(self, op):
        assert self.last_match is not None
        assert self.last_match.match(op)
        self.last_match.free = False
        match_gorn = self.last_match.gorn 
        is_left = match_gorn < self.head_address
        for point in self.insertion_points:
            if point.gorn == match_gorn[:-1]:
                point.sibling_increment(is_left)
        return match_gorn

    def matches_subpoint(self, op):
        """check to see if operation matches anything on this tree

        Args:
            op: AttachmentOperation instance
        Returns:
            True, False
        """
        self.last_type = None
        self.last_index = -1
        for index, point in enumerate(self.substitution_points):
            if point.match(op):
                self.last_type = consts.SUBSTITUTION
                self.last_index = index
                return True
        return False

    ##################### UTILITY METHODS
    #####################################

    def point_iterator(self, ignore_taken=False):
        for pt_type, points in zip(['SUB', 'INS'], [self.sub_points, self.ins_points]):
            for point in points:
                if ignore_taken and not point.free:
                    continue
                yield pt_type, point

    @property
    def ins_points(self):
        return self.insertion_points

    @property
    def sub_points(self):
        return self.substitution_points

    @property
    def root_pos(self):
        return self.tree_operation.target['pos_symbol']

    @property
    def last_match(self):
        if self.last_index < 0:
            return None
        elif self.last_type == consts.SUBSTITUTION:
            return self.substitution_points[self.last_index]
        else:
            return self.insertion_points[self.last_index]

    @property
    def is_insertion(self):
        return self.tree_operation.is_insertion

    @property
    def pos_symbol(self):
        return self.tree_operation.target['pos_symbol']

    def set_path_features(self, target_gorn=None, target_hlf=None,
                                self_hlf=None, tree_id=None):
        """Set the variables needed to reconstruct paths.

        Args
            target_gorn: the gorn address of the target operation node
            target_hlf: the target hlf symbol of the target operation tree
            self_hlf:  this tree's hlf symbol
        Notes:
            The gorn address will identify where in the target tree
            The target_hlf will identify which tree; especially important for duplicates
        """
        if self_hlf:
            for point in self.substitution_points + self.insertion_points:
                point.set_path_features(self_hlf)
            self.hlf_symbol = self_hlf

        if target_gorn or target_hlf:
            self.tree_operation.set_path_features(target_gorn, target_hlf)

        if tree_id:
            self.tree_id = tree_id

    def expand_address(self, incoming):
        self.expanded_address = incoming
        for _, point in self.point_iterator():
            point.expanded_address = incoming + point.gorn[1:]


    """ a soft deletion to see if i can get rid of this code
    def refresh_points(self):
        self.tree_operation = self.tree_operation.clone()
        self.substitution_points = [sub.clone() for sub in self.substitution_points]
        self.insertion_points = [ins.clone() for ins in self.insertion_points]

    def clone(self):
        new_tree = ElementaryTree(self.tree_operation, self.head,
                                  self.head_address, self.bracketed_string,
                                  self.substitution_points,
                                  self.insertion_points)
        new_tree.refresh_points()
        if self.last_match:
            new_tree.last_type = self.last_type
            new_tree.last_index = self.last_index
        if self.hlf_symbol:
            new_tree.hlf_symbol = self.hlf_symbol
        new_tree.address = self.address
        new_tree.tree_id = self.tree_id
        return new_tree
    """
    def __str__(self):
        return self.bracketed_string

    def __repr__(self):
        substr = ", ".join("{}{}@{}".format(sub.pos_symbol,
                                            "-FREE" if sub.free else "-FILLED",
                                            sub.gorn)
                           for sub in sorted(self.substitution_points,
                                             key=lambda x: x.gorn))
        instr = ", ".join("{}@{}".format(ins.pos_symbol, ins.gorn)
                           for ins in sorted(self.insertion_points,
                                             key=lambda x: x.gorn))
        if self.tree_operation.is_insertion:
            typestr = "{}*" if self.tree_operation.direction == "left" else "*{}"
        else:
            typestr = "^{}^"
        typestr = typestr.format(self.head)
        return "<{}; sub=[{}], ins=[{}]>".format(typestr, substr, instr)

class DerivationTree(object):
    """represent a tree of ElementaryTrees and their attachment addresses.
    """
    def __init__(self, elem_tree, children, predicate=None, suppress_predicate=False):
        self.elem_tree = elem_tree
        self.children = children
        self.predicate = predicate
        if not suppress_predicate and predicate is None:
            self.predicate = self.instantiate_semantics()

    @classmethod
    def root_tree(cls):
        E = ElementaryTree.root_tree()
        P = Predicate(name='ROOT', valence=1, hlf_symbol='g-1')
        return cls(E, [], P)

    @classmethod
    def from_single_parse_tree(cls, tree):
        elem_tree = ElementaryTree.from_single_parse_tree(tree)
        return cls(elem_tree, [])

    @classmethod
    def from_bracketed(cls, bracketed_string, **kwargs):
        elem_tree = ElementaryTree.from_bracketed_string(bracketed_string)
        #parse_tree, _ = ConstituencyTree.make(bracketed_string=bracketed_string)
        return cls(elem_tree, [], **kwargs)

    @property
    def E(self):
        """ shortcut alias for shorter lines """
        return self.elem_tree

    @property
    def is_insertion(self):
        return self.elem_tree.is_insertion

    @property
    def direction(self):
        if self.is_insertion:
            return self.E.tree_operation.target['attach_direction']
        else:
            return "up"

    @property
    def bracketed(self):
        return self.E.bracketed_string

    @property
    def head(self):
        return self.E.head

    @property
    def supertag(self):
        return (self.E.root_pos, self.E.head_symbol, self.direction)

    @property
    def superindex(self):
        return (self.head, self.supertag)

    @property
    def is_root(self):
        return "ROOT" in self.E.bracketed_string

    @property
    def num_children(self):
        return sum([child.num_children+1 for child in self.children])

    @property
    def lexical(self):
        out = [self.E.head]
        for child in self.children:
            out.extend(child.lexical)
        return out

    def accepts_op(self, other_tree):
        other_target = other_tree.E.tree_operation.target['pos_symbol']
        if other_tree.is_insertion:
            points = self.E.insertion_points
        else:
            points = self.E.substitution_points
        for point in points:
            if point.pos_symbol == other_target:
                return True
        return False

    def expand_address(self, incoming=None):
        incoming = incoming or (0,)
        self.E.expand_address(incoming)
        self.expanded_address = incoming
        for child in self.children:
            child_address = incoming + child.E.address[1:]
            child.expand_address(child_address)

    def all_points(self):
        points = list(self.E.point_iterator())
        for child in self.children:
            points.extend(child.all_points)
        return points

    def get_spine(self):
        tree, _ = ConstituencyTree.make(bracketed_string=self.bracketed)
        annotate = lambda t: (t.symbol, ("SUB" if t.complement 
                                               else ("INS" if t.adjunct 
                                                           else "SPINE")))
        not_lex = lambda t: not tree.lexical
        spine = [[(tree.symbol, self.direction)]]
        while not_lex(tree):
            if len(tree.children) == 1 and tree.children[0].lexical:
                break
            spine.append([annotate(c) for c in tree.children if not_lex(c)])
            tree = tree.children[tree.spine_index]
        return spine

    def roll_features(self, parent_head="ROOT"):
        """assumes 1 head.. more thought needed for other forms"""

        spine = self.get_spine()
        out_ch = [child.head for child in self.children]
        out = [(self.head, parent_head, self.bracketed, spine, out_ch)]
        for child in self.children:
            out.extend(child.roll_features(self.head))
        return out

    def modo_roll_features(self, parent_head="ROOT", parent_spine=None):
        """v2. mother-daughter roll features

        roll up the tree; get the mother-daughter quadruples 
        """
        parent_spine = parent_spine or ((("ROOT", "SUB"),),)
        tree, _ = ConstituencyTree.make(bracketed_string=self.bracketed)
        safety = 0
        annotate = lambda t: (t.symbol, ("SUB" if t.complement 
                                               else ("INS" if t.adjunct 
                                                           else "SPINE")))
        filter_ch = lambda c: c.E.head_symbol in [",", ":", ".", "``","''", "--"]
        not_lex = lambda t: not tree.lexical
        spine = [[(tree.symbol, self.direction)]]
        while not_lex(tree):
            if len(tree.children) == 1 and tree.children[0].lexical:
                break
            spine.append([annotate(c) for c in tree.children if not_lex(c)])
            tree = tree.children[tree.spine_index]
            safety += 1
            if safety == 100:
                raise Exception("loop issue")


        out = [(self.head, parent_head, self.bracketed, spine, parent_spine)]
        for child in self.children:
            out.extend(child.modo_roll_features(self.head, spine))

        return out


    def dcontext_roll_features(self):
        """v3. mother-daughter roll features

        roll up the trees; get the node+daughter head context
        """
        tree, _ = ConstituencyTree.make(bracketed_string=self.bracketed)
        annotate = lambda t: (t.symbol, ("SUB" if t.complement 
                                               else ("INS" if t.adjunct 
                                                           else "SPINE")))
        filter_ch = lambda c: c.E.head_symbol in [",", ":", ".", "``","''", "--"]
        not_lex = lambda t: not tree.lexical
        spine = [[(tree.symbol, self.direction)]]
        while not_lex(tree):
            if len(tree.children) == 1 and tree.children[0].lexical:
                break
            spine.append([annotate(c) for c in tree.children if not_lex(c)])
            tree = tree.children[tree.spine_index]

        hlf_info = (self.E.hlf_symbol, self.E.tree_operation.target['target_hlf'])
        child_heads = [child.head for child in self.children]
        out = [(self.head, spine, child_heads, self.bracketed, hlf_info)]
        for child in self.children:
            out.extend(child.dcontext_roll_features())

        return out

    def rollout_learning_features(self):
        tree, _ = ConstituencyTree.make(bracketed_string=self.bracketed)
        safety = 0
        annotate = lambda t: (t.symbol, ("SUB" if t.complement 
                                               else ("INS" if t.adjunct 
                                                           else "SPINE")))
        not_lex = lambda t: not tree.lexical
        spine = [[(tree.symbol, self.direction)]]
        while not_lex(tree):
            if len(tree.children) == 1 and tree.children[0].lexical:
                break
            spine.append([annotate(c) for c in tree.children if not_lex(c)])
            tree = tree.children[tree.spine_index]
            safety += 1
            if safety == 100:
                raise Exception("loop issue")

        return [self.head, spine], self.bracketed
        

    def to_constituency(self):
        raise Exception("dont use this yet")
        import pdb
        #pdb.set_trace()
        tree, _ = ConstituencyTree.make(bracketed_string=self.bracketed)
        for child in sorted(self.children, key=lambda c: c.E.address):
            print("*******\n**********")
            print("starting child {}".format(child.supertag))
            ct = child.to_constituency()
            print("----------------------------")
            print("finished to constituency for ct")
            print("tree is currently {}".format(tree))
            print("child's ct: {}".format(ct))
            print("-------------------")
            print(self.bracketed)
            print(child.E.address)
            print(str(child))

            print("attaching {} to {}".format(child.bracketed, self.bracketed))
            self.attach_at(tree, ct, list(child.E.address)[1:])
        return tree

    def attach_at(self, node, op, address):
        raise Exception("dont use this yet")
        while len(address) > 1:
            node = node.children[address.pop(0)]
        if not hasattr(node, "bookkeeper"):
            node.bookkeeper = {}
        opid = address.pop(0)
        assert len(address) == 0
        if isinstance(opid, int):
            node.children[opid].__dict__.update(op.__dict__)
        elif isinstance(opid, float):
            if opid > 0: 
                node.children.extend(op.children)
            else:
                node.children = op.children + node.children
                node.spine_index += len(op.children)
        else:
            raise Exception("sanity check")

    def __str__(self):
        if self.E.bracketed_string == "(ROOT)" and len(self.children) == 0:
            return "<empty root>"
        lexical = self.in_order_lexical()
        return " ".join(lexical)

    def __repr__(self):
        if self.E.bracketed_string == "(ROOT)" and len(self.children) == 0:
            return "<empty root>"
        descs = self.in_order_descriptive()
        return  " ".join(descs)

    def _check_heads(self, child_prep, next_word, stk_idx, sf_stk, avail_pos):
        for (head,hlf), child in child_prep.items():
            if head == next_word:
                import pdb
                #pdb.set_trace()
                w_size = child.num_children + 1
                low,high = stk_idx, stk_idx+w_size
                while high >= stk_idx and low >= 0:
                    possible = sf_stk[low:high]
                    if sorted(possible) == sorted(child.lexical):
                        child_prep.pop((head, hlf))
                        pos = avail_pos.pop()
                        return child, pos, low
                    else:
                        low -= 1
                        high -= 1

        return None, None, None

    def _sort_by_surface_form(self, sf_list, children, positions, left=True):
        """assign spine-out indices that agrees with surface form list (sf_list)

        positions start from 0 and go negative when left, positive when right
        we want to associate things closer to 0 with words closer to head
        """

        #my_possible_positions = [i for i,x in enumerate(sf_list) if x==self.E.head]
        #if "down" in [c.E.head for c in children]:
        #    import pdb
        #    pdb.set_trace()
        #for possible_position in my_possible_positions:
            #print("===")
        child_prep = {(child.E.head,child.E.hlf_symbol):child for child in children}
        pairing = []
        avail_pos = sorted(positions)
        sf_stk = sf_list[:]
        if not left:
            avail_pos = avail_pos[::-1]
            sf_stk = sf_stk[::-1]

        # if the position is so bad that it cuts off the words, just skip it
        if not all([(word in sf_stk) for c in children for word in c.lexical]):
            raise Exception()
        stk_idx = len(sf_stk) - 1
        #print("xxx")
        domain = set([w for child in children for w in child.lexical])
        import pdb
        #pdb.set_trace()
        while len(avail_pos) > 0 and stk_idx >= 0:

        #while len(sf_stk) > 0 and len(pairing)<len(children):
            #print("---", possible_position, child_prep.keys(), sf_stk, stk_idx)
            next_word = sf_stk[stk_idx]
            if next_word not in domain:
                #print("trashpop", next_word)
                sf_stk.pop()
            else:
                child, pos, low = self._check_heads(child_prep, next_word, stk_idx, sf_stk, avail_pos)
                if child is not None:
                    stk_idx = low
                    sf_stk = sf_stk[:low]
                    pairing.append((child,pos))

            stk_idx -= 1

        try:
            assert len(avail_pos) == 0
            yield pairing
        except:
            raise Exception()
            #try:
            #    assert len(my_possible_positions) > 1
            #except:
            print("available positions weren't exausted. why?")
            print("I thought i had it figured out; multiple of this head word")
            print("it partitions string too much.. but i was wrong?")
            print("debugging. inspect now.")
            import pdb
            pdb.set_trace()


    def sort_by_surface_form(self, sf_list, children, positions, left=True):

        #import pdb
        #pdb.set_trace()
        #try:
            #if self.E.head == "iii":
            #    import pdb
            #    pdb.set_trace()
        all_pairings = list(self._sort_by_surface_form(sf_list, children, positions, left))
        #except IndexError as e:
        #    print("tried to pop from an empty list... what should I do")
        #    import pdb
        #    pdb.set_trace()

        if len(all_pairings) == 1:
            return all_pairings[0]
        else:
            #try: 
            key = lambda item: (item[1], (item[0].E.head, item[0].E.hlf_symbol))
            same = lambda p1, p2: tuple(map(key,p1))==tuple(map(key,p2))
            if all([same(p1,p2) for p1 in all_pairings for p2 in all_pairings]):
                #print("all same anyway, returning")
                return all_pairings[0]
            else:
                dt_check = lambda diffs: any([item[0].E.head_symbol == "DT" for pair in diffs for item in pair])
                dt_key = lambda pairing: sum([abs(p) for c,p in pairing if c.E.head_symbol=="DT"])
                differences = [(p1,p2) for i,p1 in enumerate(all_pairings) 
                                       for j,p2 in enumerate(all_pairings) 
                                       if not same(p1,p2) and i<j]
                differences = [(x,y) for diff_item in differences for x,y in zip(*diff_item) if x!=y]
                if len(differences) == 2 and dt_check(differences):
                    #print("shortcutting")
                    out_pairing =  max(all_pairings, key=dt_key)
                    #print("hopefully works: ", out_pairing)
                    return out_pairing
                #return all_pairings[0]
                print("Not sure what to do.  not all pairings are the same. inspect please")
                import pdb
                pdb.set_trace()
            #except Exception as e:
            #    print("not exactly sure what is breaking")
            #    import pdb
            #    pdb.set_trace()

    def surface_index(self, sf_list, num_left):
        for i,w in enumerate(sf_list):
            if w == self.E.head and i >= num_left:
                return i
        return -1


    def align_gorn_to_surface(self, surface_form):
        if len(self.children) == 0:
            return

        sf_list = surface_form.split(" ")
        if self.E.head == "as" and "much" in sf_list:
            import pdb
            #pdb.set_trace()


        left_of = lambda x,me: x.elem_tree.address < me.elem_tree.head_address
        left_children = [child for child in self.children if left_of(child, self)]
        organizer = {}
        num_left = sum([child.num_children+1 for child in left_children])
        boundary = max(num_left, self.surface_index(sf_list, num_left))
        left_form = " ".join(sf_list[:boundary])
        right_form = " ".join(sf_list[boundary+1:])
        #### LEFT CHILDREN
        for child in left_children:
            addr = child.elem_tree.address
            level, position = addr[:-1], addr[-1]
            organizer.setdefault(level, []).append((child, position))
        for level, items in organizer.items():
            if len(items) == 1:
                continue
            children, positions = [x[0] for x in items], [x[1] for x in items]
            pairing = self.sort_by_surface_form(sf_list[:boundary], children, positions, True)
            for child,position in pairing:
                assert child.E.address[:-1] == level
                child.E.address = child.E.address[:-1] + (position,)

        #### RIGHT CHILDREN
        organizer = {}
        right_children = [child for child in self.children if not left_of(child, self)]
        for child in right_children:
            addr = child.elem_tree.address
            level, position = addr[:-1], addr[-1]
            organizer.setdefault(level, []).append((child, position))
        for level, items in organizer.items():
            if len(items) == 1:
                continue
            children, positions = [x[0] for x in items], [x[1] for x in items]
            pairing = self.sort_by_surface_form(sf_list[boundary+1:], children, positions, False)
            for child,position in pairing:
                assert child.E.address[:-1] == level
                child.E.address = child.E.address[:-1] + (position,)


        
        for child in left_children:
            child.align_gorn_to_surface(left_form)
        for child in right_children:
            child.align_gorn_to_surface(right_form)

    def align_gorn_to_surface_deprecated_march30(self, surface_form):
        left_of = lambda x,me: x.elem_tree.address < me.elem_tree.head_address
        surface_index = lambda child: surface_form.find(child.elem_tree.head)
        left_children = [child for child in self.children if left_of(child, self)]
        organizer = {}
        #### LEFT CHILDREN
        for child in left_children:
            addr = child.elem_tree.address
            level, position = addr[:-1], addr[-1]
            organizer.setdefault(level, []).append((child, position))
        for level, items in organizer.items():
            if len(items) == 1:
                continue
            child_list = sorted([c for c,p in items], key=surface_index)
            pop_q = deque(sorted([p for c,p in items]))
            assert [x!=y for x in pop_q for y in pop_q]
            for child in child_list:
                addr = child.elem_tree.address
                child.elem_tree.address = addr[:-1] + (pop_q.popleft(), )

        #### RIGHT CHILDREN
        organizer = {}
        right_children = [child for child in self.children if not left_of(child, self)]
        for child in right_children:
            addr = child.elem_tree.address
            level, position = addr[:-1], addr[-1]
            organizer.setdefault(level, []).append((child, position))
        for level, items in organizer.items():
            if len(items) == 1:
                continue
            child_list = sorted([c for c,p in items], key=surface_index)
            pop_q = deque(sorted([p for c,p in items]))
            for child in child_list:
                addr = child.elem_tree.address
                child.elem_tree.address = addr[:-1] + (pop_q.popleft(), )

        for child in self.children:
            child.align_gorn_to_surface(surface_form)


    def align_gorn_to_surface_old(self, surface_form):
        ins_children = [child for child in self.children if child.is_insertion]
        sub_children = [child for child in self.children if not child.is_insertion]
        surface_index = lambda child: surface_form.find(child.elem_tree.head)
        organizer = {}
        for child in ins_children:
            addr = child.elem_tree.address
            new_addr = addr[:-1] + ((1,) if addr[-1] > 0 else (-1,))
            organizer.setdefault(addr, []).append(child)
        for proxy_addr, child_list in organizer.items():
            if len(child_list) == 1:
                continue
            offset = min([c.elem_tree.address[-1] for c in child_list])
            for i, child in enumerate(sorted(child_list, key=surface_index),0):
                last_bit = i+offset
                child.elem_tree.address = proxy_addr[:-1] +(last_bit,)

        for child in self.children:
            child.align_gorn_to_surface(surface_form)
        #left_ins = [child for child in ins_children if child.elem_tree.address[-1]<0]
        #right_ins = [child for child in ins_children if child.elem_tree.address[-1]>0]
        #surface_index = lambda child: surface_form.find(child.elem_tree.head)
        #sort_key = lambda ch: ch.elem_tree.address[:-1]+()


    def gorn_in_order(self, include_empty=False):
        items = [(child.elem_tree.address, child) for child in self.children]
        if len(self.E.head) > 0:
            items.append((self.elem_tree.head_address, self))
        if include_empty:
            for point in self.elem_tree.substitution_points:
                if all([addr!=point.gorn for addr, _ in items]):
                    items.append((point.gorn, None))
        sorted_items = sorted(items)
        return sorted_items

    def gorn_pre_order(self, merged=True):
        """Return children sorted by gorn. Use for pre-order walks. 
           Will also return from inside out. 
        """
        left_of = lambda x,me: x.elem_tree.address < me.elem_tree.head_address
        left_children = [child for child in self.children if left_of(child, self)]
        right_children = [child for child in self.children if not left_of(child, self)]
        sorted_left = sorted(left_children, key=lambda x: x.elem_tree.address, reverse=True)
        #for i,left in enumerate(sorted_left):
        #    print(i,left.elem_tree.bracketed_string)
        #    print(i,left.elem_tree.address)
        sorted_right = sorted(right_children, key=lambda x: x.elem_tree.address)
        #for i,right in enumerate(sorted_right):
        #    print(i,right.elem_tree.bracketed_string)
        #    print(i,right.elem_tree.address)
        #sorted_children = sorted(self.children, key=lambda x: x.elem_tree.address)
        if merged:
            return sorted_left + sorted_right
        else:
            return sorted_left, sorted_right

    def learning_features(self, *args):
        """make learning features. currently for dual attender model. 

        output: features and annotations for pairs (parent, child)
        """
        feature_output = []

        f1  = "head={}".format(self.E.head)
        f2  = "template={}".format(self.E.bracketed_string.replace(self.E.head, ""))
        if self.is_root:
            my_feats = (f2,)
        else:
            my_feats = (f1, f2)

        for child_type, side in zip(self.gorn_pre_order(False), ("left", "right")):
            for i, child in enumerate(child_type):
                anno = []
                anno.append("dist-from-spine: {}".format(i))
                anno.append("dist-from-frontier: {}".format(len(child_type)-i-1))
                anno.append("spine-side: {}".format(side))
                if child.is_insertion:
                    anno.append("type=ins")
                else:
                    anno.append("type=sub")
                    for j, pt in enumerate(self.E.substitution_points):
                        if pt.gorn == child.E.address:
                            anno.append("argument-{}".format(j))
                child_feats, pairs_below = child.learning_features()
                feature_output.extend(pairs_below)
                feature_output.append((my_feats, child_feats, tuple(anno)))
        return my_feats, feature_output


    def _old_learning_features(self, flat=False):
        raise Exception("don't use this function anymore")
        f1 = "head={}".format(self.elem_tree.head)
        f2 = "template={}".format(self.elem_tree.bracketed_string.replace(self.elem_tree.head, ""))
        #f4 = "surface=[{}]".format(str(self))
        #fulllex = self.in_order_lexical(True)
        #f5 = "surface_with_empties=[{}]".format(fulllex)
        myfeats = {"f1":f1,"f2":f2,"f3": []}
                 #"f4":f4,"f5":f5}
        allfeats = [myfeats]
        first_ins = lambda child: (child.E.address < self.E.head_address and
                                   all([child.E.address < other_child.E.address
                                        for other_child in self.children
                                         if other_child.E.address != child.E.address]))
        last_ins = lambda child: (child.E.address > self.E.head_address and
                                   all([child.E.address > other_child.E.address
                                        for other_child in self.children
                                         if other_child.E.address != child.E.address]))

        for child in self.children:
            # if child is insertion, find out whether it's furthest left or furthest right
            # if child is substitution, find out which of the substitution poitns it corresponds to
            if first_ins(child):
                pass


            arrow = "<-" if child.is_insertion else "->"
            f3 = "{}{}{}".format(self.elem_tree.head, arrow, child.elem_tree.head)
            myfeats['f3'].append(f3)
            allfeats.extend(child.learning_features())
        
        if flat:
            final_list = []
            for featset in allfeats:
                for featval in featset.values():
                    if isinstance(featval, list):
                        final_list.extend(featval)
                    else:
                        final_list.append(featval)
            return final_list
        return allfeats


    def path_reconstruction_features(self):
        return (self.E.bracketed_string, self.E.hlf_symbol,
                self.E.tree_operation.target['target_hlf'],
                self.E.tree_operation.target['target_gorn'])

        #return (self.elem_tree.tree_id, self.elem_tree.head)

    def pre_order_features(self):
        feat_list = [self.path_reconstruction_features()]# for now, just id
        for child in self.gorn_pre_order():
            feat_list.extend(child.pre_order_features())
        return tuple(feat_list)

    def pre_order_descriptive(self):
        descs = [str(self.elem_tree)]
        sorted_children = sorted(self.children, key=lambda x: x.elem_tree.address)
        for tree in sorted_children:
            descs.extend(tree.pre_order_descriptive())
        return descs

    def in_order_descriptive(self):
        descs = []
        for address, tree in self.gorn_in_order():
            if tree == self:
                descs.append(str(self.elem_tree))
            else:
                descs.extend(tree.in_order_descriptive())
        return descs

    def in_order_treeids(self):
        treeids = []
        for address, tree in self.gorn_in_order():
            if tree == self:
                treeids.append(tree.elem_tree.tree_id)
            else:
                treeids.extend(tree.in_order_treeids())
        return treeids

    def pre_order_lexical(self):
        pass

    def in_order_lexical(self, include_empties=False):
        lexical = []
        for address, tree in self.gorn_in_order(include_empties):
            if include_empties and tree is None:
                lexical.append("<open-sub-point>")
            elif tree.elem_tree.head is None:
                continue
            elif tree == self:
                lexical.append(self.elem_tree.head)
            else:
                lexical.extend(tree.in_order_lexical())
        return lexical


    def expanded_by_hlf(self, book=None):
        if book is None:
            self.expand_address()
            book = {}
        book[self.E.hlf_symbol] = self.expanded_address
        for child in self.children:
            book = child.expanded_by_hlf(book)
        return book


    def make_expression(self, top=True):
        expr = []
        for i, (address, tree) in enumerate(self.gorn_in_order()):
            if tree == self:
                expr.append(self.predicate)
            else:
                expr.extend(tree.make_expression(False))
        if top:
            return Expression.from_iter(expr)
        return expr

    def lookup_insert(self, index):
        return self.elem_tree.insertion_points[index].gorn

    def lookup_sub(self, index):
        return self.elem_tree.substitution_points[index].gorn

    def set_path_features(self, instantiate_semantics=True, *args, **kwargs):
        self.elem_tree.set_path_features(*args, **kwargs)
        if instantiate_semantics:
            self.predicate = self.instantiate_semantics()

    def set_insertion_argument(self, arg):
        if not self.is_insertion:
            raise Exception("Don't call this if it's not insertion..")
        self.predicate.substitute(arg, 0)

    def instantiate_semantics(self):
        num_arguments = len(self.elem_tree.substitution_points)
        if self.is_insertion:
            num_arguments += 1
        predicate = Predicate(self.elem_tree.head,
                                   num_arguments,
                                   self.elem_tree.hlf_symbol)
        if self.elem_tree.hlf_symbol is None:
            self.elem_tree.set_path_features(self_hlf=predicate.hlf_symbol)
        return predicate

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def clone(self):
        children = [child.clone() for child in self.children]
        pred = self.predicate.clone()
        return self.__class__(self.elem_tree.clone(), children)

    def handle_insertion(self, operative, in_place):
        """Check if my elementary tree is the insertion point; if not, recurse

        Args:
            op_tree: ElementaryTree instance
        """
        ThisClass = self.__class__
        op_tree = operative.elem_tree
        op = op_tree.tree_operation
        if self.elem_tree.matches_inspoint(op):
            # do the insertting; making new elem tree copies; updating addresses
            new_elem_tree, new_op_tree = self.elem_tree.insert(op_tree)
            # start making the new composed tree
            # create a new clone of the op dtree
            if in_place:
                new_operative = operative
                new_operative.elem_tree = new_op_tree
                new_children = self.children
            else:
                #new_children = [child.clone() for child in self.children]
                new_children = deepcopy(self.children)
                new_operative = ThisClass.replicate(operative, new_op_tree)
            # since it's an insertion, this pred is an argument to the op
            new_pred = deepcopy(self.predicate)
            # put the predicate into the op
            new_operative.set_insertion_argument(new_pred)
            # finish off the children
            new_children.append(new_operative)
        else:
            new_elem_tree = deepcopy(self.elem_tree)
            new_children = [child.operate(operative, in_place) for child in self.children]
            new_pred = deepcopy(self.predicate)

        if in_place:
            self.elem_tree = new_elem_tree
            self.children = new_children
            self.predicate = new_pred
            return self
        else:
            return ThisClass(new_elem_tree, new_children)

    def handle_substitution(self, operative, in_place=False):
        """Check if my elementary tree is the subpoint; if not, recurse on children

        Args:
            op_tree: ElementaryTree instance
        """
        ThisClass = self.__class__
        op_tree = operative.elem_tree
        op = op_tree.tree_operation
        if self.elem_tree.matches_subpoint(op):

            # the purpose of the substitute is to give the op_tree an address
            # that adddress is the location of its substituion
            # this is important for when we want to order our derived children via gorn
            new_elem_tree, new_op_tree = self.elem_tree.substitute(op_tree)

            ##### HANDLE IN-PLACE-TYPE VS FACTORY-TYPE OPERATION
            # the thing coming in is copied
            if in_place:
                new_operative = operative
                new_operative.elem_tree = new_op_tree
                new_children = self.children
            else:
                new_children = deepcopy(self.children)#[child.clone() for child in self.children]
                new_operative = ThisClass.replicate(operative, new_op_tree)

            new_children.append(new_operative)

            ##### HANDLE LOGIC STUFF
            new_pred = deepcopy(self.predicate)#.clone()
            # we put it into its correct spot
            if self.is_insertion:
                pred_arg_index = new_elem_tree.last_index + 1
            else:
                pred_arg_index = new_elem_tree.last_index
            # abusing terms. substitute here is not a tree substitute, but a logic substitute
            # find a better term....................
            new_pred.substitute(new_operative.predicate, pred_arg_index)
        else:
            new_elem_tree = deepcopy(self.elem_tree)#.clone()
            new_pred = deepcopy(self.predicate)#.clone()
            new_children = [child.operate(operative, in_place) for child in self.children]


        if in_place:
            self.elem_tree = new_elem_tree
            self.children = new_children
            self.predicate = new_pred
            return self
        else:
            return ThisClass(new_elem_tree, new_children)

    def operate(self, operative, in_place=False):
        """handle the possible operations incoming to this derived tree.

        Args:
            operative: a DerivationTree instance
        Returns:
            a new DerivationTree that results from operation

        Notes:
            An intended operation would know what tree it wants to operate on
            and where it wants to do it.
            E.G:
                (NP* (DT a)) knows it wants to attach to the tree (NP (NN dog))
                which is substituted into (S (NP) (VP finds) (NP))
                The DerivationTree should know that (NP (NN dog)) was substituted into
                the first substitution spot.

                Temp QUD:
                    what is the best way to represent this intended operation?
                    we could have the DT tree know it wants to attach to tree id X
                    but that tree id X could be in the tree twice (either NP)
                    it could know the predicate then?
        """
        if operative.elem_tree.tree_operation.type == consts.INSERTION:
            return self.handle_insertion(operative, in_place)
        elif operative.elem_tree.tree_operation.type == consts.SUBSTITUTION:
            return self.handle_substitution(operative, in_place)

    @classmethod
    def replicate(cls, old_inst, new_elem_tree=None, new_children=None, new_pred=None):
        """ this is basically clone but allows etrees, childre, and preds rather than just straight cloning """
        new_elem_tree = new_elem_tree or deepcopy(old_inst.elem_tree)#.clone()
        new_children = new_children or deepcopy(old_inst.children) #[child.clone() for child in old_inst.children]
        new_pred = new_pred or deepcopy(old_inst.predicate)#.clone()
        return cls(new_elem_tree, new_children)


def test():
    parse = """(ROOT(S(NP(NP (DT The) (NN boy))(VP (VBG laying)(S(VP (VB face)(PRT (RP down))(PP (IN on)(NP (DT a) (NN skateboard)))))))(VP (VBZ is)(VP (VBG being)(VP (VBN pushed)(PP (IN along)(NP (DT the) (NN ground)))(PP (IN by)(NP (DT another) (NN boy))))))(. .)))"""
    tree_cuts = tree_enrichment.string2cuts(parse)
    tree_strings = [cut.save_str() for cut in tree_cuts]
    derived_trees = [DerivationTree.from_bracketed(tree_string) for tree_string in tree_strings]
    derived_trees[2].elem_tree.insertion_points[0].hlf_symbol = 'g0'
    derived_trees[1].elem_tree.tree_operation.target['target_hlf'] = 'g0'
    derived_trees[1].elem_tree.tree_operation.target['target_gorn'] = (0,)
    #derived_two = [DerivationTree.from_parse_tree(tree) for tree in tree_cuts]
    return derived_trees


if __name__ == "__main__":
    test()
