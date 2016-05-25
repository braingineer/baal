"""
Inspired by CoreNLP's transformCC

According to them, think about the following things:
// a CC b c ... -> (a CC b) c ...  with b not a DT
// something like "soya or maize oil"
// if stuff after (like "soya or maize oil and vegetables")
// we need to put the tree in another tree
//to handle the case of a comma ("soya and maize oil, and vegetables")

// DT a CC b c -> DT (a CC b) c
/create a new tree to be inserted as second child of t (after the determiner

 // ... a, b CC c ... -> ... (a, b CC c) ...

 // something like "the new phone book and tour guide" -> multiple heads
    // we want (NP the new phone book) (CC and) (NP tour guide)
    // handle the case of a preconjunct (either, both, neither)
    // handle the case of a comma ("GM soya and maize, and food ingredients")
     // handle the case of a comma ("GM soya and maize, and food ingredients")

basically, new tree. 
add preconjunct
add left
add comma
add cc
add right
add comma

comments:
case 1 will never happen to me. the right sibling of a CC will never be a DT

"""
import baal
from baal.structures import ConstituencyTree

def deprecated_case_1(cc_idx, tree):
    """ this is for a corenlp case; but i'm not doing this one for now """
    left = (tree.children[cc_idx] - 1).symbol
    right = (tree.children[cc_idx] - 1).symbol
    
    cc_pos_cond = cc_idx == 1
    left_cond1 = left.symbol in ("DT", "JJ", "RB") or right != "DT"
    left_cond2 = left[:2] != "NP" and left not in ("ADJP", "NNS")

    return cc_pos_cond and left_cond1 and left_cond2 

################
## SIMPLE: X -> X CC X 


def simple_cc_condition(tree):
    all_cc = {i:child for i,child in enumerate(tree.children) if child.symbol=="CC"}
    if len(all_cc) != 1: return False
    if len(tree.children) != 3: return False
    if len(tree.children[1].children) > 1: return False
    if (tree.children[1].symbol == "CC" and tree.children[0].symbol == tree.symbol
                                        and tree.children[2].symbol == tree.symbol):
       return True
    return False

def simple_process(tree):
    tree.children[1].children = [tree.children[0], 
                                 tree.children[1].children[0], 
                                 tree.children[2]]
    tree.children = [tree.children[1]]

#######################3
## COMPLEX: X -> X, CC X
##          X -> EITHER X CC X
## etc

def complex_cc_condition(tree):
    if tree.symbol == "NP":
        import pdb
        #pdb.set_trace()
    all_cc = {i:child for i,child in enumerate(tree.children) if child.symbol=="CC"}
    if len(all_cc) != 1: return False
    if tree.symbol != "NP": return False
    if len(tree.children) < 3: return False
    syms = [c.symbol for c in tree.children]
    cc_idx = all_cc.keys()[0]
    assert syms[cc_idx] == "CC"
    if cc_idx == len(tree.children) - 1: return False
    if syms[cc_idx+1][:2] not in ("NP", "NN", "NX"):
        return False
    return True


def complex_process(tree):
    all_cc = {i:child for i,child in enumerate(tree.children) if child.symbol=="CC"}

    cc_idx, cc_tree = list(all_cc.items())[0]
    NoneTree = ConstituencyTree("None")

    sib = lambda i: tree.children[cc_idx+i] if cc_idx + i < len(tree.children) else NoneTree
    sibs = lambda i,j: [sib(k) for k in range(i,j+1) if sib(k) is not NoneTree]
    if sib(-1).symbol[:2] == sib(1).symbol[:2]:
        lower, upper = -1, 1
    elif sib(-1).head in (",", ";") and sib(1).symbol[:2] == sib(-2).symbol[:2]:
        lower, upper = -2, 1
    else:
        if hasattr(baal.OMNI, "weird_file"):
            with open(baal.OMNI.weird_file, "a") as fp:
                fp.write("=CC.COND2============\n")
                fp.write("unknown cc condition\n")
                fp.write("=CC.COND2.START============\n")
                fp.write(tree.save_str()+"\n")
                fp.write("=CC.COND2.END============\n")
                fp.close()
        return

    if sib(lower-1).head in ("either", "both", "neither"): 
        lower -= 1

    if sib(upper+1).head in (",", ";"):
        upper += 1


    new_tree = ConstituencyTree(symbol="NP", children=sibs(lower,upper), parent="NP")
    tree.children = tree.children[:cc_idx+lower] + [new_tree] + tree.children[cc_idx+upper+1:]


def process(tree):
    return False
    did_something = False
    if complex_cc_condition(tree):
        did_something = True
        complex_process(tree)
    #elif simple_cc_condition(tree):
    #    did_something = True
    #    simple_process(tree)
    else:
        for child in tree.children:
            did_something = process(child) or did_something
    return did_something
