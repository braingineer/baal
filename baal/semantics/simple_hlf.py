from baal.semantics.hlf import gensym, unboundgensym
from baal.utils.general import nonstr_join, cformat
from collections import deque
from baal.utils import loggers


logger = loggers.duallog("hlfdebug", shell_level="debug", disable=True)
logger.setLevel(50)

def reset():
    gensym.reset()
    unboundgensym.reset()

def _preprocess(addressbook):
    if isinstance(addressbook, dict):
        addressbook = addressbook.items()
    return sorted(addressbook)


def prn_pairs(phead, thead):
    pairs = [("-LRB-", "-RRB-"), ("-RSB-", "-RSB-"), ("-LCB-", "-RCB-"), 
             ("--", "--")]
    return any([left.lower()==phead.lower() and right.lower()==thead.lower() for left,right in pairs])



def from_addressbook(addressbook, preprocess=False):
    """
        Take a tree and make it into logical form

        Observations motivating procedure:
            1. The parent's head word is the function over its arguments
            2. Its arguments are marked as complements
            3. Its spine index marks itself
            4. Things that aren't its spine and aren't its complements are adjuncts
            5. Adjuncts are functions on the headword.

        Procedure:
            Iterate through sorted (address, tree) pairs
                1. If address is longer, we have descended into children
                2. if parent's spine index is this address's last index, it's on spine
                3. if this child has a marker "complement", it's an argumetn of parent
                4. If the last two aren't true, it's an adjunct.
            Note: we do nothing for spines, but continue iterating (presumably, the head word won't change)
    """
    if preprocess:
        addressbook = _preprocess(addressbook)

    enter_child_cond = lambda addr, last_addr: len(addr) > len(last_addr)
    # exit the set of children of a node
    exit_child_cond = lambda addr, last_addr: len(addr) < len(last_addr)

    logical_form = dict()
    headbook = dict()

    stack = deque()

    parent_address, parent = addressbook[0]
    #print "PARENT IS {}".format(parent)
    last_address, last_tree = addressbook[1]

    c = lambda x,i: cformat("{}".format(x),i)

    """
    BIG WARNING. 12/20. BCM.
    ========================
    Originally, addressbook in the loop would start from the first: addressbook[1:]
    But, this only worked if it went S -> rest of tree. the reason is because of
    weak valued dictionaries, like that in gensym.
    when it is ROOT -> S -> rest of the tree, there is a loop iteration at the beginning
    where the parent loses its reference if we start with [1:].
    So, we lose the fact that there is a g0.
    If things break down the line because this assumption gets lifted (assumption of ROOT)
    then, things might need to change here.
    """
    skip_next = False
    for address, tree in addressbook:

        logger.debug(c("LOOPSTART:: {}, {}".format(address, tree),'f'))
        #if skip_next:
        #    skip_next = False
        #    continue

        if enter_child_cond(address, last_address):
            # this means we are decending a level.
            logger.debug(c("CONDITION::new-child",'w'))
            logger.debug("pushing parent: {}".format(parent.head))
            logger.debug("new parent: {}".format(last_tree.head))
            logger.debug("current tree: {}".format(tree.head))
            stack.append((parent, parent_address))
            parent, parent_address = last_tree, last_address


        elif exit_child_cond(address, last_address):
            # this means we are moving up a level
            logger.debug(c("CONDITION::exit-child",'w'))
            logger.debug("done with {}".format(parent.head))
            while len(parent_address) >= len(address):
                try:
                    parent, parent_address = stack.pop()
                except:
                    logger.debug("Stack exhaustd, should be at root.")
                    break

            logger.debug("new parent head: {}".format(parent.head))
            logger.debug("current tree: {}".format(tree.head))

        else:
            logger.debug(c("CONDITION::sibling","w"))
            # this means we are at the same level as last iteration
            pass

        # make the parent symbol
        psym = gensym(head=parent.head, address=parent_address, symbol=parent.symbol)
        logger.debug("after parent, gensym instance values: {}".format(gensym._instances.values()))
        # make the child symbol
        csym = (gensym(head=tree.head, address=address, symbol=tree.symbol)
                if len(tree.head) > 0 else
                gensym(head=tree.symbol, address=address, symbol=tree.symbol))
        logger.debug("after kid, gensym instance values: {}".format(gensym._instances.values()))
        #### record the relationship

        ### SPINE (aka, the node upon which the subtree belongs)
        ## parent address is 1 len longer than current address
        ## thus, address[-1] is child index
        ## if spine_index is the same as the child index, then it's the spine
        ## lexical material will always be here
        ## nodes on the spine will also be here, though
        if parent.spine_index == address[-1]:
            logger.debug(c("NODETYPE::spine",'0'))
            logger.debug("parent: {}<{}>".format(parent.symbol, parent.head))
            logger.debug("current: {}<{}>".format(tree.symbol, tree.head))
            logger.debug("psym %s for %s" % (psym, parent.head))
            logger.debug("csym %s for %s" % (csym, tree.head))
            if len(tree.children) == 0:
                #import pdb
                #pdb.set_trace()
                logger.debug("lexical material")
                logical_form.setdefault(csym, [csym]).append( unboundgensym(head=tree.symbol,address=address))
                csym.pos_symbol = parent.symbol

        ### ARGUMETN CONDITION
        ## parent address is one longer
        ## this has been marked as a substituted node
        ## thus, it's an argument of the parent
        elif tree.complement:
            logger.debug(c("NODETYPE::complement/argument",'0'))
            logger.debug("parent: %s<%s>" % (parent.symbol, parent.head))
            logger.debug("current: %s<%s>" % (tree.symbol, tree.head))
            logger.debug("psym %s for %s" % (psym, parent.head))
            logger.debug("csym %s for %s" % (csym, tree.head))

            ## it's an argument, but it lacks its own lexical material
            if len(tree.children) == 0:
                logger.debug("NOCHILDREN::{}".format(tree.symbol))
                csym = unboundgensym(head=tree.symbol,address=address, symbol=tree.symbol)
                logical_form.setdefault(psym, [psym]).append(csym)
            else:
                logger.debug("CHILDREN::{}".format(tree.symbol))
                logical_form.setdefault(psym, [psym]).append(csym)
                logical_form.setdefault(csym, [csym])
        #elif tree.spine_index >=0 and prn_pairs(parent.head, tree.head):
        #    logger.debug("This is a wrapping condition; currently only handling PRNs this way")
        #    skip_next = True
        else:
            logger.debug(c("CONDITION::adjunct", '0'))
            logger.debug("parent: {}<{}>".format(parent.symbol, parent.head))
            logger.debug("current: {}<{}>".format(tree.symbol, tree.head))
            logger.debug("PSYM ({}) is argument to CSYM({}) for HEAD({})".format(psym, csym, tree.head))
            logical_form.setdefault(csym, [csym]).append(psym)
        for k,v in logical_form.items():
            logger.debug("Current logical form: {} -> {}".format(k, v))
        logger.debug("--\n--\n")
        last_address, last_tree = address, tree

    hlf_make = lambda func,variables: "%s(%s)" % \
                                      (func.head, nonstr_join(variables, ','))
    hlf_items = sorted(logical_form.items(), key = lambda x: str(x[1][0]))
    # print hlf_items
    hlf_expr = " & ".join([hlf_make(f,vs) for f,vs in hlf_items])

    # print hlf_expr

    return logical_form

def find_dominant(logical_form):
    """ this is a bad algorithm. fix later. it's basically a bubble sort """
    sorted = {k:i for i,k in enumerate(logical_form.keys())}
    for sym1, args1 in logical_form.items():
        for sym2, args2 in logical_form.items():
            if sym2 in args1 and sorted[sym2]>sorted[sym1]:
                # note, python let's us do one line variable swap
                sorted[sym1], sorted[sym2] = sorted[sym2], sorted[sym1]
            elif sym1 in args2 and sorted[sym1]>sorted[sym2]:
                sorted[sym1], sorted[sym2] = sorted[sym2], sorted[sym1]
    return sorted

def post_process_hlf(logical_form):
    post_processed = {}
    dominant = None
    sorted = find_dominant(logical_form)
    # print 'blah'
    for sym, args in logical_form.items():
        if not any([(sym in v and k.pos_symbol == sym.pos_symbol) for k,v in post_processed.items()]):
            post_processed[sym] = args
        else:
            # print 'here'
            kvs = post_processed.items()
            for k,v in kvs:
                if sym in v and k.pos_symbol == sym.pos_symbol:
                    # print 'del'
                    del post_processed[k]
                    post_processed[sym] = args

    return post_processed


def hlf_format(logical_form):
    hlf_make = lambda func,variables: "%s(%s)" % (func.head, nonstr_join(variables, ','))
    hlf_items = sorted(logical_form.items(), key = lambda x: str(x[1][0]))
    hlf_expr = " & ".join([hlf_make(f,vs) for f,vs in hlf_items])
    return hlf_expr

def pretty_hlf(predicate):
    colorfy = lambda x,i,j=0: ("\x1b[{}m{}\x1b[0m".format(i,x) if 31<=i<=37 else
                           ("\x1b[38;5;{}m{}\x1b[0m".format(j,x) if i==38 else "BAD COLOR"))
    colorproc = lambda variables: ([colorfy(variables[0],38,99)] +
                                   [(colorfy(v,38,214) if v.head!=v.symbol else v)
                                    for v in variables[1:]])
    hlf_make = lambda func,variables: "{}({})".format(colorfy("{}_{}".format(func.head,func.pos_symbol.lower()),38,202),
                                                      nonstr_join(colorproc(variables), ','))
    hlf_items = sorted(predicate.items(), key = lambda x: str(x[1][0]))
    hlf_expr = " &\n\t ".join([hlf_make(f,vs) for f,vs in hlf_items])
    return hlf_expr



def hard_test():
    parse = """(ROOT(S(NP (NNP Man))(VP (VBD dressed)(PP (IN in)(NP(NP (NN leather) (NNS chaps))(CC and)(NP (JJ purple) (NN shirt) (NNS stands))))(PP (IN in)(NP(NP (NN front))(PP (IN of)(NP (NNS onlookers))))))(. .)))"""
    import baal
    entry = baal.nlp.structures.Entry.make(bracketed_string=parse)
    tree, surface = entry.tree, entry.get_lexical()
    baal.nlp.induce.tree_enrichment.populate_annotations(tree)
    baal.nlp.induce.tree_enrichment.recursive_spine_fix(tree)
    _, new_addressbook = tree.clone()
    logger.setLevel(50)
    from_addressbook(new_addressbook, preprocess=True)

if __name__ == "__main__":
    hard_test()