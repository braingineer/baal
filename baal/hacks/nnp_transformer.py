"""Merge trees which only have NNP children into multi-word expressions

"""
from baal.structures import ConstituencyTree

def nnp_condition(tree):
    for child in tree.children:
        if child.symbol != "NNP": return False
        if len(child.children) != 1: return False
        if not child.children[0].lexical: return False
    return True

def merge(tree):
    #import pdb
    #pdb.set_trace()
    new_mwe = "_".join(child.head for child in tree.children)
    new_child = ConstituencyTree(symbol=new_mwe, parent="NNP")
    new_child.lexical = True
    new_tree = ConstituencyTree(symbol="NNP", children=[new_child])
    new_tree.spine_index = 0
    new_tree.head = new_mwe
    tree.children = [new_tree]
    tree.spine_index = 0
    tree.head = new_mwe

def process(tree):
    did_something = False
    if nnp_condition(tree):
        did_something = True
        merge(tree)
    else:
        for child in tree.children:
            did_something = process(child) or did_something
    return did_something
