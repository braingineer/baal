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

