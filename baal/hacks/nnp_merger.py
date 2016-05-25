"""
merging those nnps because. [full stop]
"""

from baal.structures import ConstituencyTree

def nnp_condition(tree):
    return tree.symbol == "NP" and all([c.symbol=="NNP" for c in tree.children])

def _process(tree):
    assert all([len(c.children)==1 and c.children[0].lexical for c in tree.children])
    lex_str = "_".join([c.head for c in tree.children])
    newlex = ConstituencyTree(lex_str, parent="NNP")
    newlex.head = lex_str
    newt = ConstituencyTree("NNP", [newlex], parent="NP")
    newt.head = lex_str
    tree.head = lex_str
    print(lex_str)
    tree.children = [newt]

def process(tree):
    if tree is None:
        import pdb
        pdb.set_trace()
    did_something = False
    if nnp_condition(tree):
        did_something = True
        _process(tree)
    else:
        for child in tree.children:
            did_something = did_something or process(child)
    return did_something