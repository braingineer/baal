"""
Finding Dependencies and Heads in Derivation Trees

- For 1-ply subtrees, we have a root and a set of children.
- In tree-annotated corpora, it's not indicated:
    - which of the children are the head,
    - which of the children are the complements
    - which of the children are the adjuncts

[Wagerman95] used a head percolation table to propogate heads upward
from the leaves to the root.  He was interested in this information for a
PCFG

[Collins1999] used these tables in his dissertation to further head-driven
PCFG parsing.  Collins extended this to finding the constituents of the heads
so that they can be distinguished from the adjuncts.

Required datastructure (see data_structures for wrappers):
    tree(object):
        head_word = ""
        head_index = -1
        adjuncts = []
        substitutions = []


@author bcmcmahan
"""
from __future__ import print_function, division
import baal
from baal.utils import AlgorithmicException
from baal.utils.general import backward_enumeration, flatten, SimpleProgress, cformat
from baal.semantics import simple_hlf
from collections import defaultdict
import argparse
from functools import wraps
from copy import deepcopy
import logging
try:
    import cPickle as pickle
except:
    import pickle

try:
    input = raw_input
except:
    pass

__all__ = ["parse2derivation", "annotation_cut", "populate_annotations", 
           "string2cuts", "transform_tree", "get_trees"]


logger = baal.utils.loggers.duallog("treecut", "debug", disable=True)
logger.setLevel(50)

def debugrule(func):
    @wraps(func)
    def debug_and_call(*args,**kwargs):
        if len(args)==2:
            children, headlist = args
        else:
            return func(*args, **kwargs)
        fname = cformat(func.func_name, 'w')
        cname = "; ".join([cformat("{}".format(child.symbol), 'f')+str(child) for child in children])
        logger.debug("inside {} search with children: {} and headlist: {}".format(fname,
                                                                           cname,
                                                                           headlist))
        ind, suc = func(*args, **kwargs)
        if ind is not None:
            csel = cformat("{}".format(children[ind]),'f')
        else: 
            csel = cformat("None", "f")
        logger.debug("selecting {} at index {}; successful={}".format(csel, ind, suc))
        return ind, suc
    return debug_and_call

class CollinsMethod(object):
    headruleset = {
            "ROOT": [("right",)],
            "ADJP": [("left", "NNS", "QP", "NN", "$", "ADVP", "JJ",
                      "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT",
                      "FW", "RBR", "RBS", "SBAR", "RB")],
            "ADVP": [("right", "RB", "RBR", "RBS", "FW", "ADVP",
                       "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN")],
            "CONJP": [("right", "CC", "RB", "IN")],
            "FRAG": [("right",)],
            "INTJ": [("left",)],
            "LST": [("right", "LS", ":")],
            "NAC": [("left", "NN", "NNS", "NNP", "NNPS", "NP", "NAC",
                      "EX", "$", "CD", "QP", "PRP", "VBG", "JJ", "JJS",
                      "JJR", "ADJP", "FW")],
            "NX": [("left",)],
            "PP": [("right", "IN", "TO", "VBG", "VBN", "RP", "FW"),
                   ("left", "PP")],
            "PRN": [("left",)],
            "PRT": [("right", "RP")],
            "QP": [("left", "$", "IN", "NNS", "NN", "JJ", "RB", "DT",
                    "CD", "NCD", "QP", "JJR", "JJS")],
            "RRC": [("right", "VP", "NP", "ADVP", "ADJP", "PP")],
            "S": [("left", "TO", "IN", "VP", "S", "SBAR",
                   "ADJP", "UCP", "NP")],
            "SBAR": [("left", "WHNP", "WHPP", "WHADVP", "WHADJP", "IN",
                      "DT", "S", "SQ", "SINV", "SBAR", "FRAG")],
            "SBARQ": [("left", "SQ", "S", "SINV", "SBARQ", "FRAG")],
            "SINV": [("left", "VBZ", "VBD", "VBP", "VB", "MD", "VP",
                      "S", "SINV", "ADJP", "NP")],
            "SQ": [("left", "VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ")],
            "UCP": [("right",)],
            "VP": [("left", "TO", "V", "VBD", "VBN", "MD", "VBZ", "VB", "VBG",
                    "VBP", "AUX", "AUXG", "VP", "ADJP", "NN", "NNS", "NP")],
            "WHADJP": [("left", "CC", "WRB", "JJ", "ADJP")],
            "WHADVP": [("right", "CC", "WRB")],
            "WHNP": [("left", "WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP")],
            "WHPP": [("right", "IN", "TO", "FW")],
            "X": [("right",)],
            "NP": [("cc_context", "CC"),
                   ("rightdis", "NN", "NNP", "NNPS", "NNS",
                    "NX", "POS", "JJR"),
                   ("left", "NP"),
                   ("rightdis", "$", "ADJP", "PRN"),
                   ("right", "CD"),
                   ("rightdis", "JJ", "JJS", "RB", "QP"),
                   ('right',)],
            "TYPO": [("left",)],
            "EDITED": [("left",)],
            "XS": [("right", "IN")],
            "ROOT": [("left",)],
            "": [('left',)],
                    }


    class CMWrapper:
        @staticmethod
        @debugrule
        def cc_context(children, headlist, parent):
            all_cc = {i:child for i,child in enumerate(children) if child.symbol=="CC"}
            if len(all_cc) == 0: 
                return 0, False
            
            cc_idx = sorted(all_cc.keys())[-1]
            if cc_idx + 1 == len(children) or cc_idx == 0: 
                return 0,False

            L = lambda j: children[cc_idx+j] 
            if L(1).symbol != parent.symbol: 
                return 0, False
            if (L(-1).symbol == L(1).symbol) or (L(-1).head in (",",";") and
                                                 L(-2).symbol == L(1).symbol):
                return cc_idx, True

            return 0, False

        @staticmethod
        @debugrule
        def left(children, headlist, parent):
            """ head symbol preference then forward order of children preference """
            for head_symbol in headlist:
                for c_i, child in enumerate(children):
                    if child.symbol == head_symbol:
                        return c_i, True
            return 0, (False or len(headlist) == 0)


        @staticmethod
        @debugrule
        def right(children, headlist, parent):
            """ head symbol preference then backward order of children preference """
            for head_symbol in headlist:
                for c_i, child in backward_enumeration(children):
                    if child.symbol == head_symbol:
                        return c_i, True
            last_ind = len(children) - 1
            return last_ind, (False or len(headlist) == 0)


        @staticmethod
        @debugrule
        def leftdis(children, headlist, parent):
            """ forward order of children preference """
            for c_i, child in children:
                if child.symbol in headlist:
                    return c_i, True
            return 0, (False or len(headlist) == 0)

        @staticmethod
        @debugrule
        def rightdis(children, headlist, parent):
            """ backward order of children preference """
            for c_i, child in backward_enumeration(children):
                if child.symbol in headlist:
                    return c_i, True
            last_ind = len(children) - 1
            return last_ind, (False or len(headlist) == 0)

        @staticmethod
        @debugrule
        def leftexcept(children, headlist, parent):
            """ excluding symbols, forward order of children preference """
            for c_i, child in enumerate(children):
                if child.symbol not in headlist:
                    return c_i, True
            return 0, (False or len(headlist) == 0)

        @staticmethod
        @debugrule
        def rightexcept(children, headlist, parent):
            """ excluding symbols, backward order of children preference """
            for c_i, child in backward_enumeration(children):
                if child.symbol not in headlist:
                    return c_i, True
            last_ind = len(children) - 1
            return last_ind, (False or len(headlist) == 0)

    headrule_functions = {"left": CMWrapper.left,
                          "right": CMWrapper.right,
                          "leftdis": CMWrapper.leftdis,
                          "rightdis": CMWrapper.rightdis,
                          "leftexcept": CMWrapper.leftexcept,
                          "rightexcept": CMWrapper.rightexcept,
                          "cc_context": CMWrapper.cc_context}

    complementruleset = {"first_condition":
                         [(set(("NP", "SBAR", "S")), set(["S"])),
                          (set(("NP", "SBAR", "S", "VP")), set(["VP"])),
                          (set(("NP", "S")), set(["SBAR"]))],
                          #(set(("NP", "VP", "PP", "CC", "SBAR", "S")), set(["PRN"]))],
                         "second_condition":
                         set(("ADV", "VOC", "BNF", "DIR", "EXT", "LOC", "MNR",
                              "TMP", "CLR", "PRP"))
                         }


    @staticmethod
    def ccs_are_hard(parent, c_i, children):
        if children[parent.head_index].symbol != "CC": return False
        cc_idx = parent.head_index
        #import pdb
        #pdb.set_trace()

        diff = cc_idx - c_i
        if cc_idx == len(children) - 1 or cc_idx == 0: return False
        if diff not in [-2, -1, 1]: return False
        if diff == -2:
            if children[cc_idx-1].head not in (",", ";"): return False
            if children[cc_idx+1].symbol != children[c_i].symbol: return False
            return True
        if children[cc_idx-1].symbol == children[cc_idx+1].symbol: return True
        return False


    @staticmethod
    def mark_complements(parent, children):
        for c_i, child in enumerate(children):
            if parent.symbol == "PP" and c_i == 1:
                child.complement = True
            elif c_i == parent.head_index:
                continue
            elif (parent.head_index - c_i == 1
                  and parent.symbol == "NP"
                  and children[parent.head_index].symbol == "POS"):
                child.complement = True
            elif CollinsMethod.ccs_are_hard(parent, c_i, children):
                child.complement = True 
            else:
                child.complement = CollinsMethod.is_complement(parent, child)
        return children

    @staticmethod
    def is_complement(parent, child):
        first_cond = CollinsMethod.complementruleset["first_condition"]

        first_condition_bool = []
        for nt_set, pnt_set in first_cond:
            if child.symbol in nt_set and parent.symbol in pnt_set:
                first_condition_bool.append(True)
            else:
                first_condition_bool.append(False)
        first_condition_bool = any(first_condition_bool)

        second_cond = CollinsMethod.complementruleset["second_condition"]
        return first_condition_bool and child.semantictag not in second_cond

def transform_tree(tree):
    transformers = {"cc":baal.hacks.cc_transformer,
                    "nnp": baal.hacks.nnp_transformer,
                    "projection": baal.hacks.projection_transformer}#, 
                    #"nnp": baal.nlp.hacks.nnp_merger}
    for name, transformer in transformers.items():
        before = tree.verbose_string(verboseness=4)
        transformer.process(tree)
        #    print(name, "did something"))
        #   print(before))
        #    print("\n=====\n=====\n"))
        #    print(tree.verbose_string(verboseness=4)))


def populate_annotations(tree):
    """
        Input: a tree with subtree children
        Output: each tree object is annotated with head, adjunct, or substitution
    """
    transform_tree(tree)
    parent = tree
    children = tree.children
    parent = _annotate(parent, children)
    parent, children = select_head(parent, children)
    return parent

def _annotate(parent, children):
    if parent.lexical:
        parent.head = parent.symbol
        return parent


    if len(children)==0:
        # This is the lexical node.
        # Shouldn't be here though...
        children[0].head = children[0].symbol
        raise ValueError, ("Shouldn't be here. tree_enrichment, lexical node",
                           "Unless.. Maybe we don't see only pre-terminals before leaves")

    if len(children)==1 and children[0].lexical:
        children[0].symbol = children[0].symbol.lower()
        head = children[0]
        parent.head_index = 0
        parent.head = children[0].symbol

    else:
        # print(len(children))
        # print(children)
        for child in children:
            child = _annotate(child, child.children)

        parent, children = select_head(parent, children)
        # print("with a parent as %s" % parent.symbol)
        # print("the head is index %d" % parent.head_index)

        adj, subs, children = mark_dependencies(parent, children)
        # print("we found adjuncts: %s" % [a.symbol for a in adj])
        # print("we found substitutions: %s" % [s.symbol for s in subs])
        # print("but overall, we have children: %s" % [(ci, c) for ci, c in enumerate(children)])

        parent.adjuncts, parent.substitutions, parent.children = \
            mark_dependencies(parent, children)

    return parent




def select_head(parent, children):
    """
        Input:
            parent: the parent Non-Terminal
            children: a list of symbols, either Non-Terminal or Terminal

        Procedure:
            Find rule for parent
            Proceed by parameters of rule to select head
            Default if rule matches nothing
            Annotate head
            Return

        Output:
            Returns (parent,children) with head child annotated
    """
    rules = CollinsMethod.headruleset[parent.symbol]
    logger.debug("=======\n starting select head with "+cformat("{}".format(parent.symbol), '2'))
    logger.debug("rules: {}".format(rules))
    # print("parent symbol", parent.symbol)
    # print("rules",  rules)
    # (bmcmahan; 10/14/2015) :::
    #       I am adding the default for each rule to be the last rule's
    #       head direction. aka, in Collins' method, left/right indicates
    #       head-initial or head-final categories
    #       Some categories have more than one rule, so we can't default
    #       on the first rule, so we default on the last rule
    rules = [["cc_context", ("CC",)]] + rules
    for i,rule in enumerate(rules):
        # print(rule)
        search_method, argset = rule[0], rule[1:]
        # print(rules)
        func = CollinsMethod.headrule_functions[search_method]
        head_ind, success = func(children, argset, parent)
        if head_ind is not None and (success or i+1 == len(rules)):
            #head_ind = len(children)-1 if head_ind < 0 else head_ind
            head = children[head_ind]
            head.on_spine = True
            parent.head_index = head_ind
            # print(head)
            parent.head = head.head
            parent.head_symbol = head.symbol

            parent.spine_index = head_ind
            logger.debug("=======")
            return parent, children
    raise AlgorithmicException("We shouldn't be here")

def mark_dependencies(parent, children):
    """
        Input:
            parent: the parent Non-Terminal
            children: a list of symbols, either Non-Terminal or Terminal

        Procedure:
            iterate children
            determine if child meets complement rules
            annotate complement if it does
            annotate adjunct if it does not

        Output:
            returns (parent,children) with

    """
    children = CollinsMethod.mark_complements(parent, children)
    normal_head = lambda parent, c_i: parent.head_index == c_i
    prn_head = lambda parent, c_i: (parent.symbol == "PRN" and 
                                   (parent.head_index == c_i or 
                                              c_i + 1 == len(parent.children)))
    adjunct_condition = lambda child, parent, c_i: (not child.complement and 
                                                    not normal_head(parent, c_i)) 
                                                    #and not prn_head(parent, c_i))
    adjuncts = [child for c_i, child in enumerate(children)
                if adjunct_condition(child, parent, c_i)]
    complements = [child for child in children if child.complement]
    #adjuncts, complements = specialcases(children, adjuncts, complements)
    return adjuncts, complements, children

def specialcases(children, adjuncts, complements):
    if (len(adjuncts) > 1 and
        all([adj.symbol==adjuncts[0].symbol for adj in adjuncts])):
        adjuncts = []
    #for adj in adjuncts:
    #    if adj.symbol == ',' and children.index(adj)
    return adjuncts, complements

def annotation_cut(tree):
    """
        Given an annotated tree, return the forest of subtrees which
            represents the annotated split (head with spine, complements via
            substitution, and adjuncts via insertion)

    """
    return  _recursive_cut(tree) + [tree]


def _recursive_cut(subtree):
    decomposed = []
    recursed = []
    fix_spine(subtree)
    subs = [annotation_cut(t) for t in subtree.excise_substitutions()]
    adjs = [annotation_cut(t) for t in subtree.excise_adjuncts()]
    decomposed.extend(flatten(subs))
    decomposed.extend(flatten(adjs))
    spine = _get_spine(subtree)
    if spine:
        recursed = _recursive_cut(spine)
    return decomposed + recursed


def spine_collapse(tree):
    ##import pdb
    #pdb.set_trace()
    #print("{} -> {} children [{}]".format(tree.symbol, len(tree.children), [c.symbol for c in tree.children]))
    if len(tree.children) > 0:
        SI1 = tree.spine_index   ## SPINE INDEX; LEVEL 1
        SC1 = tree.children[SI1] ## SPINE CHILD; LEVEL 1
        if SC1.lexical: return
        assert SC1.head == tree.head
        if SC1.symbol == tree.symbol:   
            #print("firing yes")
            LC1 = tree.children[:SI1]  ## LEFT CHILDREN ; LEVEL 1
            RC1 = tree.children[1+SI1:]  ## RIGHT CHILDREN ; LEVEL 1
            #if len(SC1.children) > 0:
            #    SI2 = SC1.spine_index   ### SPINE INDEX ; LEVEL 2
            #    SC2 = [SC1.children[SI2]]  ## SPINE CHILD; LEVEL 2
            #    LC2 = SC1.children[:SI2] ## LEFT CHILDREN; LEVEL 2
            #    RC2 = SC1.children[1+SI2:] ## RIGHT CHILDREN; LEVEL 2
            #else: 
            #    LC2, RC2, SC2 = [],[],[]
            #tree.children = LC1 + LC2 + SC2 + RC2 + RC1
            #print("NEW {} -> {} children [{}]".format(tree.symbol, len(tree.children), [c.symbol for c in tree.children]))
            #print("---"*10)
            #for k,v in tree.__dict__.items():
            #    print(k, v)
            #print("---"*10)
            #for k,v in SC1.__dict__.items():
            #    print(k, v)
            tree.children = LC1 + SC1.children + RC1
            tree.spine_index = len(LC1) + SC1.spine_index
            tree.substitutions += SC1.substitutions
            tree.adjuncts += SC1.adjuncts
            #print("---"*10)
            #for k,v in tree.__dict__.items():
            #    print(k, v)
            #print("=="*10) 
    #new_c = []
    for child in tree.children:
        spine_collapse(child)

    #tree.children = new_c
    #return tree

def recursive_spine_fix(tree):
    fix_spine(tree)
    for child in tree.children:
        fix_spine(child)


def _get_spine(subtree):
    if len(subtree.children) > 0:
        return subtree.children[subtree.spine_index]
    else:
        return False

def fix_spine(tree):
    """
        we can't excise our spine.

        The tree should have a spine index. if its spine is in the adjuncts
        or the substitutions, remove it, and update all of the bookkeeping stuff
        inside the fixed spine.
    """
    if len(tree.children) > 0:
        spine = _get_spine(tree)
        # print("tree %s " % tree)
        # print("Has a spine: %s" % spine)
        if spine in tree.substitutions:
            tree.substitutions.remove(spine)
        if spine in tree.adjuncts:
            tree.adjuncts.remove(spine)

hlf_gen = ("g{}".format(i) for i in xrange(10**10))       
def pp(tree):
    if len(repr(tree)) > 10:
        return "{}->{}...".format(tree.symbol, ", ".join([x.symbol for x in tree.children]))
    else:
        return repr(tree) 

def debug_print(*args):
    if baal.OMNI.VERBOSE:
        print(*args)

def travel(tree, hlf=None, gorn=None, is_root=True):
    hlf = hlf or next(hlf_gen)      
    out = [tree] if is_root else []
    left, right = [], []
    gorn = gorn or (0,)
    offset = 0
    delete = []
    desc_str = "***********\n\t{}-fix spine_index={};"
    desc_str += "\n\t|children|={}; \n\t|subs|={}; \n\t|ins|={}"
    debug_print("####TRAVELSTART#####")
    debug_print("Tree={}".format(tree.save_str()))
    debug_print(desc_str.format("Pre", tree.spine_index, 
                          len(tree.children),
                          len(tree.substitutions),
                          len(tree.adjuncts)))
    fix_spine(tree)
    debug_print(desc_str.format("Post", tree.spine_index, 
                          len(tree.children),
                          len(tree.substitutions),
                          len(tree.adjuncts)))
    if len(tree.children) == len(tree.substitutions) + len(tree.adjuncts) and len(tree.children)>0:
        debug_print("THERE IS A PROBLEM WITH THE SPINE")
    tree.self_hlf = hlf
    
    if baal.OMNI.VERBOSE:
        ans = input("Start PDB? : ")
        if "y" == ans:
            import pdb
            pdb.set_trace()
        elif "exit" == ans:
            import sys
            sys.exit(0)


    for i, child in enumerate(tree.children):
        if tree.spine_index == i:
            debug_print("Spine condition")
            tree.spine_index -= len(delete)
        elif child in tree.substitutions:
            debug_print("Substitution Condition")
            subtree = deepcopy(child)
            subtree.target_gorn = gorn+(i-len(delete),)
            subtree.target_hlf = tree.self_hlf
            tree.children[i] = baal.structures.ConstituencyTree(child.symbol)
            tree.children[i].complement = True
            tree.children[i].parent = tree.symbol
            if i < tree.spine_index:
                left.append(subtree)
            else:
                right.append(subtree)
            #subtree, more_out = travel(subtree)
            #out += more_out
        elif child in tree.adjuncts:
            debug_print("Insertion condition")
            subtree = baal.structures.ConstituencyTree(symbol=tree.symbol, 
                                                            children=[deepcopy(child)])
            subtree.adjunct = True
            subtree.spine_index = 0
            subtree.direction = 'left' if i < tree.spine_index else 'right'
            subtree.target_gorn = gorn
            subtree.target_hlf = tree.self_hlf
            delete.append(i)
            if i < tree.spine_index:
                left.append(subtree)
            else:
                right.append(subtree)
            #subtree, more_out = travel(subtree)
            #out += more_out
        

    tree.substitutions = []
    tree.adjuncts = []
    tree.children = [child for j, child in enumerate(tree.children) if j not in delete]

    left.reverse()
    for subtree in left:        
        debug_print("---START------------------")
        debug_print("Left Enum={}".format(subtree.save_str()))
        subtree, more_out = travel(subtree)
        debug_print("Left Enum={}".format(subtree.save_str()))
        debug_print("---END------------------")
        out += more_out  

    if tree.spine_index >= 0:
        debug_print("---START------------------")
        debug_print("Spine descent")
        updated, more_out = travel(tree.children[tree.spine_index],hlf=tree.self_hlf, 
                                   gorn=gorn+(tree.spine_index,),is_root=False)
        debug_print("---END------------------")

        out += more_out

    for subtree in right:
        debug_print("--START-------------------")
        debug_print("Right enum:{}".format(subtree.save_str()))
        subtree, more_out = travel(subtree)
        debug_print("Right enum:{}".format(subtree.save_str()))
        debug_print("--END-------------------")
        out += more_out 

    debug_print("####TRAVELEND#####")

    return tree, out


def make_dt(annotation_tuple):
    treestr, selfhlf, targethlf, targetgorn = annotation_tuple
    tree = baal.structures.DerivationTree.from_bracketed(treestr)
    tree.set_path_features(self_hlf=selfhlf)
    if targethlf:
        tree.set_path_features(target_hlf=targethlf, target_gorn=targetgorn)
    return tree

def rollout(path):
    dtree = make_dt(path[0])
    #print("root: ", dtree))
    for annotation_tuple in path[1:]:
        etree = make_dt(annotation_tuple)
        dtree.operate(etree, True)
    return dtree

def parse2derivation(parse):
    tree, addr = baal.structures.ConstituencyTree.make(bracketed_string=parse)
    populate_annotations(tree)
    spine_collapse(tree)
    final_tree, out = travel(tree)
    subtree_list = []
    for o in out:
        try:
            subtree_list.append((o.save_str(), o.self_hlf, o.target_hlf, o.target_gorn))
        except:
            subtree_list.append((o.save_str(), o.self_hlf, "g-1", (0,)))

    subtree_list = sorted(subtree_list, key=lambda x: int(x[2][1:]))
    dtree = rollout(subtree_list)
    return dtree, subtree_list


def string2cuts(instr):
    tree = baal.structures.Entry.make(bracketed_string=instr).tree
    if tree.symbol == "" and len(tree.children) == 1: 
        tree.symbol = "ROOT"
    populate_annotations(tree)
    cuts = annotation_cut(tree)
    return cuts


def get_trees(infile, use_tqdm=False):
    # trying things WITH root
    #fix = lambda y: [x.replace("\n","").replace("ROOT","").strip() for x in y]
    fix = lambda y: [x.replace("\n","").strip() for x in y]
    with open(infile) as fp:
        capacitor = []
        is_consuming = True
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(enumerate(fp), desc=' lines')
        else:
            it = enumerate(fp)
        for i, line in it:

            # catching shitty noise
            if line[0]!="(" and not capacitor:
                continue

            if line[0]=="(" and capacitor:
                yield "".join(fix(capacitor))
                capacitor = []

            capacitor.append(line)
    if len(capacitor)>0:
        yield "".join(fix(capacitor))
    raise StopIteration

if __name__ == "__main__":
    pass