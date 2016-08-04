from baal.utils.general import cprint, bcolors, cformat, nonstr_join
from baal.utils import config
import re, types, logging
from baal.structures import ConstituencyTree



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
    def strings2entries(cls, tree_strings):
        ret = []
        for tree_string in tree_strings:
            if "RB-" in tree_string: continue
            ret.append(cls.make(bracketed_string=tree_string,
                                  correct_root=True))
        return ret

    @classmethod
    def make(cls, bracketed_string, correct_root=False):
        """ Initial make. Combine will copy, not make """
        tree, addressbook = ConstituencyTree.make(bracketed_string=bracketed_string)
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
    

    def _isleft(self, addressone, addresstwo):
        for ind in range(min(len(addressone), len(addresstwo))):
            if addressone[ind]< addresstwo[ind]:
                return True
            elif addressone[ind] > addresstwo[ind]:
                return False
    
    def _isright(self, addressone, addresstwo):
        for ind in range(min(len(addressone), len(addresstwo))):
            if addressone[ind] < addresstwo[ind]:
                return False
            elif addressone[ind] > addresstwo[ind]:
                return True
    
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
        print("======\nDerived Operations\n-----")
        for deriv in self.derived:
            print(deriv)
        print("======")

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(repr(self.tree))
