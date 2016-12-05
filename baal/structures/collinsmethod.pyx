from functools import wraps
from tree cimport Tree

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




headrule_functions = {"left": left,
                      "right": right,
                      "leftdis": leftdis,
                      "rightdis": rightdis,
                      "leftexcept": leftexcept,
                      "rightexcept": rightexcept,
                      "cc_context": cc_context}

complementruleset = {"first_condition":
                     [(set(("NP", "SBAR", "S")), set(["S"])),
                      (set(("NP", "SBAR", "S", "VP")), set(["VP"])),
                      (set(("NP", "S")), set(["SBAR"]))],
                      #(set(("NP", "VP", "PP", "CC", "SBAR", "S")), set(["PRN"]))],
                     "second_condition":
                     set(("ADV", "VOC", "BNF", "DIR", "EXT", "LOC", "MNR",
                          "TMP", "CLR", "PRP"))
                     }


cdef ccs_are_hard(Tree tree, int c_i):
    if tree.children[tree.head_index].symbol != "CC": return False
    cc_idx = tree.head_index
    #import pdb
    #pdb.set_trace()

    diff = cc_idx - c_i
    if cc_idx == len(tree.children) - 1 or cc_idx == 0: return False
    if diff not in [-2, -1, 1]: return False
    if diff == -2:
        if tree.children[cc_idx-1].head not in (",", ";"): return False
        if tree.children[cc_idx+1].symbol != tree.children[c_i].symbol: return False
        return True
    if tree.children[cc_idx-1].symbol == tree.children[cc_idx+1].symbol: return True
    return False


cdef mark_complements(Tree tree):
    for c_i, child in enumerate(tree.children):
        if tree.symbol == "PP" and c_i == 1:
            child.complement = True
        elif c_i == tree.head_index:
            print('{}.{} is spine'.format(tree.symbol, child.symbol))
            continue
        elif (tree.head_index - c_i == 1
              and tree.symbol == "NP"
              and tree.children[tree.head_index].symbol == "POS"):
            child.complement = True
        elif ccs_are_hard(tree, c_i):
            child.complement = True
        else:
            child.complement = is_complement(tree, child)
        print('{}.{} is complement: {}'.format(tree.symbol, child.symbol, child.complement))
    return tree.children

cdef is_complement(Tree tree, Tree child):
    first_cond = complementruleset["first_condition"]

    first_condition_bool = []
    for nt_set, pnt_set in first_cond:
        if child.symbol in nt_set and tree.symbol in pnt_set:
            first_condition_bool.append(True)
        else:
            first_condition_bool.append(False)
    first_condition_bool = any(first_condition_bool)

    second_cond = complementruleset["second_condition"]
    return first_condition_bool # and child.semantictag not in second_cond

# debug_on = False

# def debugrule(func):
#     if debug_on:
#         @wraps(func)
#         def debug_and_call(*args,**kwargs):
#             if len(args)==2:
#                 children, headlist = args
#             else:
#                 return func(*args, **kwargs)
#             fname = cformat(func.func_name, 'w')
#             cname = "; ".join([cformat("{}".format(child.symbol), 'f')+str(child) for child in children])
#             logger.debug("inside {} search with children: {} and headlist: {}".format(fname,
#                                                                                cname,
#                                                                                headlist))
#             ind, suc = func(*args, **kwargs)
#             if ind is not None:
#                 csel = cformat("{}".format(children[ind]),'f')
#             else:
#                 csel = cformat("None", "f")
#             logger.debug("selecting {} at index {}; successful={}".format(csel, ind, suc))
#             return ind, suc
#         return debug_and_call
#     else:
#         return func

cdef backward_enumeration(list vals):
    cdef int n = len(vals)
    return [(i, val) for i, val in zip(range(n-1,0),vals[::-1])]

#@debugrule
cdef cc_context(list children, tuple headlist, Tree parent):
    cdef dict all_cc = {i:child for i,child in enumerate(children) if child.symbol=="CC"}
    if len(all_cc) == 0:
        return 0, False

    cdef list cc_idx = sorted(all_cc.keys())[-1]
    if cc_idx + 1 == len(children) or cc_idx == 0:
        return 0,False

    L = lambda j: children[cc_idx+j]
    if L(1).symbol != parent.symbol:
        return 0, False
    if (L(-1).symbol == L(1).symbol) or (L(-1).head in (",",";") and
                                         L(-2).symbol == L(1).symbol):
        return cc_idx, True

    return 0, False

#@debugrule
cdef left(list children, tuple headlist, Tree parent):
    """ head symbol preference then forward order of children preference """
    cdef str head_symbol
    cdef int c_i
    cdef Tree child
    for head_symbol in headlist:
        for c_i, child in enumerate(children):
            if child.symbol == head_symbol:
                return c_i, True
    return 0, (False or len(headlist) == 0)


#@debugrule
cdef right(list children, tuple headlist, Tree parent):
    """ head symbol preference then backward order of children preference """
    cdef str head_symbol
    cdef int c_i
    cdef int last_ind
    cdef Tree child
    for head_symbol in headlist:
        for c_i, child in backward_enumeration(children):
            if child.symbol == head_symbol:
                return c_i, True
    return (len(children) - 1), (False or len(headlist) == 0)


#@debugrule
cdef leftdis(list children, tuple headlist, Tree parent):
    """ forward order of children preference """
    cdef int c_i
    cdef Tree child

    for c_i, child in children:
        if child.symbol in headlist:
            return c_i, True
    return 0, (False or len(headlist) == 0)

#@debugrule
cdef rightdis(list children, tuple headlist, Tree parent):
    """ backward order of children preference """
    cdef int c_i
    cdef Tree child
    for c_i, child in backward_enumeration(children):
        if child.symbol in headlist:
            return c_i, True
    return (len(children) - 1), (False or len(headlist) == 0)

#@debugrule
cdef leftexcept(list children, tuple headlist, Tree parent):
    """ excluding symbols, forward order of children preference """
    cdef int c_i
    cdef Tree child
    for c_i, child in enumerate(children):
        if child.symbol not in headlist:
            return c_i, True
    return 0, (False or len(headlist) == 0)

#@debugrule
cdef rightexcept(list children, tuple headlist, Tree parent):
    """ excluding symbols, backward order of children preference """
    cdef int c_i
    cdef Tree child
    for c_i, child in backward_enumeration(children):
        if child.symbol not in headlist:
            return c_i, True
    return (len(children) - 1), (False or len(headlist) == 0)


