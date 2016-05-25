"""
Overall Motivation
==================

Unify two expressions

Given two hlf expressions,
    return the binding set if they can unify false otherwise

Classes
=======
    1. Predicate - atomic unit
        - an UnboundPredicate as well for unknowns
    2. Expression - multiple predicates
    3. Substituter - helper for unification

Notes
=====
    - the tests function at the bottom of the page shows a usage example
    - I am using this for checking the output of a constraint satifaction generator
"""
from __future__ import print_function
from collections import deque
from copy import copy, deepcopy
try:
    import baal
    use_default = False
except:
    # this is just for handling the case in which I give the code to someone
    # and they don't have baal
    print("baal not on this machine; defaulting")
    import logging
    use_default = True

try:
    xrange(1)
except NameError:
    xrange = range

class consts:
    # set log level here
    LOGLEVEL = ["debug", "warning", "info", "error"][3]
    # turn logger on or off
    LOGON = True
    # this is handling the case in which i give this code to someone and they
    # don't have baal
    if use_default:
        levels={"debug":logging.DEBUG, "info":logging.INFO, "warning":logging.WARNING}
        LOGGER = logging.getLogger("unify")
        ch = logging.StreamHandler()
        ch.setLevel(levels[LOGLEVEL])
        LOGGER.addHandler(ch)
        LOGGER.setLevel(level=levels[LOGLEVEL])
    else:
        LOGGER = baal.utils.loggers.get("unify", level=LOGLEVEL, turn_on=LOGON)

    DEBUG = LOGGER.debug
    WARNING = LOGGER.warning

class Predicate(object):
    classwide_gensym = ('g{}'.format(i) for i in xrange(10**10))
    def __init__(self, name, valence=0, hlf_symbol=None, pos_symbol=None):
        self.name = name
        self.frozen = False
        self.hlf_symbol = hlf_symbol or next(self.classwide_gensym)
        self.arguments = [UnboundPredicate() for _ in range(valence)]
        self.valence = valence
        self.arity = valence
        self.granular = False
        self.pos_symbol = pos_symbol
        self.bound = True
        self.target = None
        self.is_insertion = False
        self.attach_dir = None

    @property
    def full_arguments(self):
        return [self.hlf_symbol] + self.arguments

    @property
    def argument_keys(self):
        return [arg.dict_key for arg in self.arguments]

    @property
    def dict_key(self):
        return self.name+"_"+self.hlf_symbol

    def set_target(self, sym, insertion_target=False):
        self.target = sym
        self.is_insertion = insertion_target

    def update_valence(self, valence):
        # don't let this happen right now
        if self.valence > 0:
            consts.WARNING("Valence already set")
            return
        self.arguments = [UnboundPredicate() for _ in range(valence)]
        self.valence = valence
        self.arity = valence

    def substitute(self, var, position):
        try:
            self.arguments[position] = var
        except IndexError as e:
            consts.DEBUG("Bad position argument")
            raise e

    def standardize(self, var_map):
        self.hlf_symbol = var_map.get(self.hlf_symbol)

    def unify(self, other, var_map):
        for var1, var2 in zip(self.full_arguments, other.full_arguments):
            var_map.check_subst(var1, var2)
        return True

    def valence_match(self, other):
        return len(self.arguments) == len(other.arguments)

    def is_argument(self, other_pred):
        check1 = other_pred in self.arguments
        check2 = other_pred.hlf_symbol in [x.hlf_symbol for x in self.arguments]
        assert check1 == check2
        return check1

    def __str__(self):
        child_symbols = [self.hlf_symbol]
        child_symbols.extend([arg.hlf_symbol for arg in self.arguments])
        return "{}({})".format(self.name, ",".join(child_symbols))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((str(self), len(self.arguments)))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

class UnboundPredicate(Predicate):
    classwide_gensym = ("X{}".format(i) for i in xrange(10**10))
    def __init__(self):
        sym = next(self.classwide_gensym)
        super(UnboundPredicate, self).__init__(sym, 0)
        self.self_var = None
        self.bound = False

    def copy(self):
        return UnboundPredicate()

class Substituter(dict):
    def __init__(self, *args, **kwargs):
        symbol = "C"
        if 'symbol' in kwargs:
            symbol = kwargs.pop('symbol')
        super(Substituter, self).__init__(*args, **kwargs)
        self.gensym = ("{}{}".format(symbol, i) for i in xrange(10**10))
        self.recently_new = []

    def get(self, variable, default=None):
        if variable not in self.keys():
            self.recently_new.append(variable)
            self[variable] = default or next(self.gensym)
        return self[variable]

    def check_subst(self, var1, var2):
        subst1 = self.get(var1)
        subst2 = self.get(var2, subst1)
        if subst1 != subst2:
            return False
        return True

    def commit(self):
        self.recently_new = []

    def revert(self):
        for var in self.recently_new:
            del self[var]

    def copy(self):
        new_copy = Substituter(self.items())
        new_copy.gensym = self.gensym

class Expression(dict):
    def add(self, predicate):
        if predicate.dict_key not in self:
            self[predicate.dict_key] = predicate
        else:
            predicate = self[predicate.dict_key]
        return predicate

    @classmethod
    def from_iter(cls, iterable):
        ret = cls()
        ret.add_many(iterable)
        return ret

    def add_many(self, args):
        return [self.add(arg) for arg in args]

    def standardize_apart(self, other_expr):
        me, them = "V", "W"
        mymap = Substituter(symbol=me)
        theirmap = Substituter(symbol=them)

        for pred in self.values():
            pred.standardize(mymap)

        for pred in other_expr.values():
            pred.standardize(theirmap)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k,v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        for k,v in self.items():
            result[deepcopy(k)] = deepcopy(v)
        return result

    def annotate_relations(self, surface_form):
        # get root by hlf symbol
        root = min(self.values(), key=lambda x: x.hlf_symbol)
        root.set_target("g-1")
        frontier = [root.dict_key]
        closed = set()
        while len(frontier) > 0:
            #print('frontier: ', [(f.name,f.hlf_symbol) for f in frontier])
            pred = self[frontier.pop()]
            #print("-- frontier: ", pred.name, pred.hlf_symbol)
            closed.add(pred.dict_key)
            for other_pred in self.values():
                if (len(other_pred.arguments) > 0 and other_pred.dict_key not in closed and
                       other_pred.dict_key not in frontier and
                       other_pred.arguments[0].hlf_symbol == pred.hlf_symbol):
                    #print('++ frontier via ins: ', other_pred.name, other_pred.hlf_symbol)
                    other_pred.set_target(pred.hlf_symbol, True)
                    assert other_pred.dict_key not in closed and other_pred.dict_key not in frontier
                    frontier.append(other_pred.dict_key)
            if pred.is_insertion:
                iterchildren = pred.arguments[1:]
            else:
                iterchildren = pred.arguments
            for child in iterchildren:
                child.set_target(pred.hlf_symbol)
                assert child.dict_key not in frontier
                #print("++ frontier via sub: ", child.name, child.hlf_symbol)
                frontier.append(child.dict_key)



    def unify(self, other_expr):
        var_map = Substituter()
        graveyard = set()

        self.standardize_apart(other_expr)

        myque = deque(self.keys())
        import random
        random.shuffle(myque)
        otherque = deque(other_expr.keys())

        while len(myque) > 0 and len(otherque) > 0:
            p1, p2 = myque.pop(), otherque.pop()
            consts.DEBUG("popped p1={} and p2={}".format(p1,p2))
            pred1, pred2 = self[p1], other_expr[p2]

            ##### CONDITION ONE: GLITCH IN THE MATRIX
            #####               (we've seen this pair before)
            if (pred1.dict_key,pred2.dict_key) in graveyard:
                consts.DEBUG("{} never found in other_expr".format(p1))
                consts.DEBUG("Failing now")
                return False

            ##### CONDITION TWO: PREDICATE name IS WRONG
            if pred1.name != pred2.name:
                consts.DEBUG("{} is not {}".format(pred1.name, pred2.name))
                myque.append(p1)
                otherque.appendleft(p2)
                graveyard.add((pred1.dict_key,pred2.dict_key))
                continue

            ##### CONDITION THREE: VARIABLES ARE WRONG
            if not pred1.unify(pred2, var_map):
                # failure means get rid of these new substitutions
                var_map.revert()
                myque.append(p1)
                otherque.appendleft(p2)
                graveyard.add((pred1.dict_key,pred2.dict_key))
                continue
            else:
                # success means save the new substitutions
                var_map.commit()

        if len(myque) > 0 or len(otherque) > 0:
            return False

        return var_map

    def __str__(self):
        return " & ".join(str(p) for p in sorted(self.values(), key=lambda x: int(x.hlf_symbol[1:])))

    @classmethod
    def from_args(cls, *args):
        expr = Expression()
        expr.add_many(args)
        return expr

    @classmethod
    def from_derived_hlf(cls, predicates):
        """ from the other hlf class that's induced from parse trees"""
        expr = cls()
        if isinstance(predicates, dict):
            predicates = predicates.items()
        for pred, args in predicates:
            # take all args but the first and the unbound
            args = [arg for arg in args
                        if baal.semantics.hlf.is_bound(arg)][1:]
            #import pdb
            #pdb.set_trace()
            new_pred = expr.add(Predicate(pred.head, valence=len(args),
                                                     pos_symbol=pred.pos_symbol,
                                                     hlf_symbol=pred.symbol))
            new_pred.update_valence(len(args))
            for i, arg in enumerate(args):
                argpred = expr.add(Predicate(arg.head, 
                                             valence=0, 
                                             pos_symbol=arg.pos_symbol,
                                             hlf_symbol=arg.symbol))
                new_pred.substitute(argpred, i)
        expr.root_key = min(expr.items(), key=lambda x: x[1].hlf_symbol)[0]
        return expr

    @classmethod
    def from_derivation_tree(cls, dtree):
        expr = cls()
        frontier = [dtree]
        while len(frontier) > 0:
            dtree = frontier.pop()
            frontier.extend(dtree.children)
            dtree.predicate.pos_symbol = dtree.E.head_symbol
            expr.add(dtree.predicate)
        
        #assert all([p.pos_symbol is not None for p in expr.values()])

        return expr



def test_unification(expr1, expr2):
    varmap = expr1.unify(expr2)
    if varmap:
        consts.DEBUG("We did it!")
        for k,v in varmap.items():
            consts.DEBUG("{}<=>{}".format(k, v))
        return True
    else:
        consts.DEBUG("Failure")
        return False

def success_test():
    kicked = Predicate("kicked", valence=2, hlf_symbol="g0")
    jake = Predicate("Jake", valence=0, hlf_symbol="g1")
    finn = Predicate("Finn", valence=0, hlf_symbol="g2")
    kicked.substitute(jake, 0)
    kicked.substitute(finn, 1)
    expr1 = Expression.from_args(kicked, jake, finn)

    kicked2 = Predicate("kicked", valence=2, hlf_symbol="g5")
    jake2 = Predicate("Jake", valence=0, hlf_symbol="g6")
    finn2 = Predicate("Finn", valence=0, hlf_symbol="g7")
    kicked2.substitute(jake2, 0)
    kicked2.substitute(finn2, 1)
    expr2 = Expression.from_args(kicked2, jake2, finn2)

    return test_unification(expr1, expr2)

def failure_test():
    kicked = Predicate("kicked", valence=2, hlf_symbol="g0")
    jake = Predicate("Jake", valence=0, hlf_symbol="g1")
    finn = Predicate("Finn", valence=0, hlf_symbol="g2")
    kicked.substitute(jake, 0)
    kicked.substitute(finn, 1)
    expr1 = Expression.from_args(kicked, jake, finn)

    kicked2 = Predicate("kicked", valence=2, hlf_symbol="g8")
    cake = Predicate("Cake", valence=0, hlf_symbol="g1")
    fiona = Predicate("Fiona", valence=0, hlf_symbol="g2")
    kicked2.substitute(cake, 0)
    kicked2.substitute(fiona, 1)
    expr2 = Expression.from_args(kicked2, cake, fiona)

    return test_unification(expr1, expr2)

if __name__ == "__main__":
    assert success_test() == True
    assert failure_test() == False
    print("Tests passed")

