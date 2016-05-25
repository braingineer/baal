"""
Fix projections. 

"""
from baal.structures import ConstituencyTree

def projection_condition(tree):
    if len(tree.children) != 1: return False
    if tree.children[0].symbol != tree.symbol: return False
    if tree.children[0].lexical: return False
    return True

def _process(tree):
    tree.spine_index = tree.children[0].spine_index
    tree.children = tree.children[0].children


def process(tree):
    did_something = False
    if projection_condition(tree):
        did_something = True
        _process(tree)
    else:
        for child in tree.children:
            did_something = process(child) or did_something
    return did_something
