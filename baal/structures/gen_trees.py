"""GIST trees used for generation purposes

Summary
-------

basically, I need a tree structure that is more useful to generation.
I need the following operations with described qualities:

Operation Desiderata
^^^^^^^^^^^^^^^^^^^^
1. Substitution
    - The root point needs to match the sub point
        + Previously, the op also needed to know the gorn and the hlf
        + Now, we let it be agnostic to both
2. Insertion
    - The root points need to match the insertion point
    - The insertion needs to affect a target index
        + this means affecting the ElementaryTree itself, right?
        + as the elementary tree attaches the insertion point,
            it usually gives it either the left or right frontier as its next place
        + this was done so that the address remains the same for that substitution node
        + so when we save the sub point later, it has the same address no matter what
        + so how do we modify the gorn so that it allows for insertion?
        + do we shift everything over and the sub node now has a new address?
        + this is how I did it before, but it never really worekd all that well
        + we could save it as a continuous number instead
        + such that the numbe always falls between the actual structured nodes
        + this could work, i think
        + the comaprisons would remain true
        + gorn order would come out the same

How things are breaking down:
1. The GTree manages the elem tree
2. the op will need to know its address
3. the things being copied are the target links
4. it doens't really matter if we copy the  elem tree forward
5. the actual address for the insertion node should be at the first point where nothing has been
6. in other words, moving outward from the spine

"""

from .gist_trees import ElementaryTree, DerivationTree, consts
from copy import copy, deepcopy
import time
from nltk.stem import PorterStemmer


class GenerationTree(DerivationTree):
    def generate(self, operative):
        if operative.elem_tree.tree_operation.type == consts.INSERTION:
            return self.handle_insert_generation(operative)
        elif operative.elem_tree.tree_operation.type == consts.SUBSTITUTION:
            return self.handle_sub_generation(operative)

    def _determine_gorn(self, gorn, is_left):
        if is_left:
            op_index = 0
        else:
            op_index = 0
        for ins_point in self.elem_tree.insertion_points:
            if ins_point.gorn == gorn:
                continue
            if ins_point.gorn[:len(gorn)] == gorn:
                if is_left:
                    op_index = min(op_index, ins_point.gorn[len(gorn)]-1)
                else:
                    op_index = max(op_index, ins_point.gorn[len(gorn)]+1)

        for subpoint in self.elem_tree.substitution_points:
            if subpoint.gorn[:len(gorn)] == gorn and not subpoint.free:
                if is_left:
                    op_index = min(op_index, subpoint.gorn[len(gorn)]-1)
                else:
                    op_index = max(op_index, subpoint.gorn[len(gorn)]+1)

        return op_index


    def handle_insert_generation(self, operative):
        op_tree = operative.elem_tree
        op_point = op_tree.tree_operation
        # check the insertion points
        for point in self.elem_tree.insertion_points:
            # see if it matches; our only criterion
            if self.matches(point, op_point):
                # it does, so clone the elem tree
                new_elem = deepcopy(self.elem_tree)#.clone()
                is_left = op_point.target['attach_direction'] == 'left'
                op_index = self._determine_gorn(point.gorn, is_left)
                address = point.gorn+(op_index, )
                try:
                    if is_left:
                        assert address < self.elem_tree.head_address
                    else:
                        assert address > self.elem_tree.head_address
                except:
                    print("assertion failure")
                    print(address)
                    print(self.elem_tree.head_address)

                new_op = deepcopy(operative)#.clone()
                new_op.elem_tree.address = address
                new_children = deepcopy(self.children) + [new_op]#[child.clone() for child in self.children] + [new_op]

                #new_pred = self.predicate.clone()
                # put the predicate into the op
                #new_op.set_insertion_argument(new_pred)

                yield GenerationTree(new_elem, new_children)

        for i, child in enumerate(self.children):
            for new_child in child.handle_insert_generation(operative):
                new_children = self.children[:i] + [new_child] + self.children[i+1:]
                elem_tree = deepcopy(self.elem_tree)#.clone()
                #new_pred = self.predicate.clone()
                yield GenerationTree(elem_tree, new_children)

    def matches(self, target_point, op_point, sub=False):
        pos_match = target_point.pos_symbol == op_point.target['pos_symbol']
        if target_point.hlf_symbol is None or op_point.target['target_hlf'] is None:
            hlf_match = True
        else:
            hlf_match = target_point.hlf_symbol == op_point.target['target_hlf']
        return pos_match and hlf_match and (target_point.free or not sub)

    def handle_sub_generation(self, operative):

        op_tree = operative.elem_tree
        op_point = op_tree.tree_operation
        #print("here")
        #import pdb
        #pdb.set_trace()
        for i, point in enumerate(self.elem_tree.substitution_points):
            if self.matches(point, op_point, sub=True):
                new_elem = deepcopy(self.elem_tree)#.clone()
                new_elem.substitution_points[i].free = False
                gorn = new_elem.substitution_points[i].gorn

                new_op = deepcopy(operative)#.clone()
                new_op.elem_tree.address = gorn

                new_children = deepcopy(self.children) #[child.clone() for child in self.children]
                new_children.append(new_op)

                new_pred = deepcopy(self.predicate)#.clone()
                # adjust for the insertion (0th) argument
                if self.is_insertion:
                    new_pred.substitute(new_op.predicate, i+1)
                else:
                    new_pred.substitute(new_op.predicate, i)

                yield GenerationTree(new_elem, new_children, new_pred)

        for i, child in enumerate(self.children):
            for new_child in child.handle_sub_generation(operative):
                new_children = self.children[:i] + [new_child] + self.children[i+1:]
                elem_tree = deepcopy(self.elem_tree)#.clone()
                new_pred = deepcopy(self.predicate)#.clone()
                yield GenerationTree(elem_tree, new_children)

class Lattice(object):
    def __init__(self, expression, grammar, language_model):
        self.stm = PorterStemmer()
        self.subtrees = {key:LatticeSubtree(pred,self) for key,pred in expression.items()}
        self.expr = expression
        self.op_map = self.process_predicates(expression)
        self.head_map = self.process_grammar(grammar)
        self.language_model = language_model

    def process_predicates(self, expression):
        """ make the operand map by walking the expression graph 
            the operand map encodes all incoming connections.
            this way, we can move bottom up on the graph.
        """
        predbyhlf = {pred.hlf_symbol:pred for pred in expression.values()}
        self.root = self.subtrees[predbyhlf['g0'].dict_key]
        frontier = [predbyhlf['g0']]
        operand_map = {}
        while len(frontier) > 0:
            pred = frontier.pop()
            # get this pred's substitutions, set as operand map
            args = pred.arguments[:]

            if len(args) > 0 and args[0].hlf_symbol in operand_map:
                args = args[1:]
            operand_map[pred.hlf_symbol] = args
            for arg in args:
                arg.set_target(pred.hlf_symbol)
            frontier.extend(args)
            # check for insertions, add to frontier and to this pred's operand map
            for other_pred in expression.values():
                if (pred.hlf_symbol in [arg.hlf_symbol 
                                        for arg in other_pred.arguments] and
                    other_pred.hlf_symbol not in operand_map):
                    other_pred.set_target(pred.hlf_symbol, True)
                    operand_map[pred.hlf_symbol].append(other_pred)
                    frontier.append(other_pred)
        assert len(operand_map) == len(expression)
        return {predbyhlf[sym].dict_key:ops for sym,ops in operand_map.items()}

    def __str__(self):

        if self.root.done:
            out = "Completed.\n"
            for option in sorted(self.root.tree_struct, key=lambda x: x['score']):
                out += "\tScore: {}. \n\t\tSentence: {}\n".format(option['score'], option['result_tree'])
        else:
            out = "Not finished."
        return out

    def process_grammar(self, grammar):
        """ make the a map from stems to trees """
        head_map = {}
        for g in grammar:
            head_map.setdefault(g.E.head,[]).append(g)
        return head_map

    def get_subtree(self, pred):
        """ for a single predicate, return its latticesubtree """
        return self.subtrees[pred.dict_key]

    def get_trees(self, pred):
        """find trees that match pred in stem and arity. 
           also, assigns to them the proper HLF symbols (self and target)
        """
        out = []
        assert pred.target is not None or pred.hlf_symbol == "g0"
        for tree in self.head_map[pred.name]:
            if tree.is_insertion != pred.is_insertion: continue
            if tree.E.head_symbol != pred.pos_symbol: continue

            arity = len(tree.E.substitution_points) + (1 if tree.is_insertion else 0)
            if arity != pred.arity:  continue

            tree = deepcopy(tree)
            tree.set_path_features(self_hlf=pred.hlf_symbol, target_hlf=pred.target)
            out.append(tree)
        return out

    def get_operands(self, pred):
        """return the operands for the predicate from the operand map"""
        return [self.get_subtree(pred) for pred in self.op_map[pred.dict_key]]

    def walk(self):
        start = time.time()
        while any([st.ready and not st.done for st in self.subtrees.values()]):
            for subtree in self.subtrees.values():
                if subtree.ready and not subtree.done:
                    subtree.step()

        print("finished in {} seconds".format(time.time()-start))

class LatticeSubtree(object):
    def __init__(self, predicate, ref):
        self.pred = predicate
        self.ref = ref
        self.tree_struct = []

    @property
    def trees(self):
        out = []
        for item in self.tree_struct:
            out.append(item['result_tree'])
        return out

    @property
    def ready(self):
        """ all incoming edges must be done """
        operands = self.ref.get_operands(self.pred)
        return all([operand.done for operand in operands])

    @property
    def done(self):
        return len(self.tree_struct) > 0

    def collect(self, result_tree, score):
        self.tree_struct.append({'result_tree': result_tree,
                                 'score': score})

    def step(self):
        # i have a head word
        trees = self.ref.get_trees(self.pred)
        # tricky thing for get_operands
        # to make things easier, we need to have insertions come in this route
        # this way, the operate stays consistent
        # we can trace through the pred structure at first and form a map
        # the map will assume any the children from root are correct subs
        # other preds that have as children these nodes, will not be marked as parents
        # instead, they will be marked as operands of that child node
        # so really, it's an operation graph
        operands = self.ref.get_operands(self.pred)
        #flat_ops = [op_tree for operand in operands for op_tree in operand.trees]
        for tree in trees:
            # want to keep the best option for each of this subtree's options
            # we keep a second data struct (bucket) so resultants doesn't grow inside the loop
            resultants, bucket = [(tree, ([],[tree.E.bracketed_string]))], []
            # go over each the tree options so far (multiple loops in case of multiple operands)
            #genset = [tree]
            #for operand in operands:
            #    if operand.is_insertion: continue
            #    genset = [newtree for op_tree in operand.trees 
            #                      for rtree in genset 
            #                      for newtree in rtree.generate(op_tree)]
            #for operand in operands:
            #    if operand.
            for _ in range(len(operands)):
                for result_tree, (op_list,b_list) in resultants:
                    # loop over operands. in other words, find all ways and all order that this 
                    # tree can be expanded
                    for operand in operands:
                        if operand.pred.hlf_symbol in op_list:
                            # op list maintains the list of hlf symbols already used
                            # used when there are multiple operands
                            continue
                        # iterate over the things that will be subbing into this
                        # / the things that will be inserting into this
                        # for insertion, order matters. 
                        # so, I've moved the operand loop inside the resultant loop
                        # this insures that all possible orderings will occur.
                        for op_tree in operand.trees:
                            for new_tree in result_tree.generate(op_tree):
                                new_oplist = copy(op_list) + [operand.pred.hlf_symbol]
                                new_blist = copy(b_list) + [op_tree.E.bracketed_string]
                                bucket.append((new_tree, (new_oplist, new_blist)))   
                    # conditional here in case we're in base case and there are no operands
                resultants = bucket if len(bucket) > 0 else resultants
            resultants = [(t,(o,b)) for t,(o,b) in resultants if len(o) == len(operands)]
            dups = set()
            out = []
            for t,(o,b) in resultants:
                key = (tuple(sorted(b)), str(t))
                if key in dups: continue
                out.append(t)
                dups.add(key)
            if len(resultants) > 0:
                #print("RESULTANTS PRIOR TO RANKING")
                #for r in resultants:
                #    print(r)
                #print("THE ONES MAKING IT OUT")
                #for o in out:
                #    print(o)
                best_result, score = self.ref.language_model.rank(out, False)
                self.collect(best_result, score)
